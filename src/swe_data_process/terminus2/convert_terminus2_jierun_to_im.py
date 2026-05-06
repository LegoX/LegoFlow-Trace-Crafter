"""
Harbor 作业的 trial 目录名（与 results/result.json 中的 id 一致）由 aggregate result 列出。
默认读取 job-dir/results.json；若不存在则使用 job-dir/result.json（harbor 实际文件名）。
使用 stats.evals[*].reward_stats.reward["1.0"] 中的 trial id，仅转换这些子目录下的 agent/trajectory.json。
"""

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from swe_data_process.terminus2.common import convert_one_record, iter_records, to_lf_record
from swe_data_process.rule_score import score_dataset
from swe_data_process.utils import EXCLUDED_REPOS_FILE, filter_instance_ids_by_repo, load_exclusion_patterns, load_json, print_lf_token_stats, save_jsonl


DEFAULT_JOB_DIR = Path(
    "/home/ywxzml3j/ywxzml3juser30/code/harbor/jobs/"
    "swerebench-filtered-oraclesolved-terminus-2-GLM-5-FP8-20260321014346"
)
DEFAULT_IM_OUTPUT = Path(
    "/home/ywxzml3j/ywxzml3juser57/LLaMA-Factory/data/"
    "jierun_glm5_swerebench_oraclesolved_t2_1k.jsonl"
)
DEFAULT_LF_OUTPUT = Path(
    "/home/ywxzml3j/ywxzml3juser57/LLaMA-Factory/data/"
    "jierun_glm5_swerebench_oraclesolved_t2_1k.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将 terminus2 jierun Harbor job 目录下的 agent/trajectory.json 转为 IM / LF 数据"
    )
    parser.add_argument(
        "--job-dir",
        type=Path,
        default=DEFAULT_JOB_DIR,
        help="Harbor 评测 job 根目录（含 results.json 或 result.json）",
    )
    parser.add_argument(
        "--im-output",
        type=Path,
        default=DEFAULT_IM_OUTPUT,
        help="输出 IM 格式 JSONL",
    )
    parser.add_argument(
        "--lf-output",
        type=Path,
        default=DEFAULT_LF_OUTPUT,
        help="输出 LLaMA-Factory sharegpt 格式 JSON",
    )
    parser.add_argument(
        "--max-instances",
        type=int,
        default=None,
        help="最多成功转换多少条，默认不限制",
    )
    parser.add_argument(
        "--exclude-repos-file", type=lambda s: Path(s) if s else None,
        default=EXCLUDED_REPOS_FILE,
        help="排除 repo 列表文件路径（由 generate_excluded_repos.py 生成）",
    )
    return parser.parse_args()


def resolve_results_json_path(job_root: Path) -> Path:
    preferred = job_root / "results.json"
    fallback = job_root / "result.json"
    if preferred.exists():
        return preferred
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"No results file at {preferred} or {fallback}")


def load_trial_ids_from_results(data: Any) -> list[str]:
    """Collect trial folder names / instance keys from harbor result or simple JSON lists."""
    ids: list[str] = []

    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            key = (
                item.get("instance_id")
                or item.get("trial_id")
                or item.get("id")
                or item.get("folder")
            )
            if key is not None and str(key).strip():
                ids.append(str(key).strip())
        return sorted(set(ids))

    if not isinstance(data, dict):
        return []

    stats = data.get("stats")
    if isinstance(stats, dict):
        evals = stats.get("evals")
        if isinstance(evals, dict):
            for entry in evals.values():
                if not isinstance(entry, dict):
                    continue
                rs = entry.get("reward_stats") or {}
                if not isinstance(rs, dict):
                    continue
                reward = rs.get("reward") or {}
                if not isinstance(reward, dict):
                    continue
                solved = reward.get("1.0")
                if isinstance(solved, list):
                    ids.extend(str(x).strip() for x in solved if x is not None and str(x).strip())
            if ids:
                return sorted(set(ids))

    for list_key in ("instance_ids", "trial_ids", "trials", "results"):
        blob = data.get(list_key)
        if isinstance(blob, list):
            for x in blob:
                if isinstance(x, str) and x.strip():
                    ids.append(x.strip())
                elif isinstance(x, dict):
                    key = x.get("instance_id") or x.get("trial_id") or x.get("id")
                    if key is not None and str(key).strip():
                        ids.append(str(key).strip())
            if ids:
                return sorted(set(ids))

    return sorted(set(ids))


def resolve_trajectory_path(job_root: Path, instance_key: str) -> Path | None:
    """Map result id to agent/trajectory.json (exact dir name or owner__repo-issue__suffix prefix)."""
    direct = job_root / instance_key / "agent" / "trajectory.json"
    if direct.exists():
        return direct

    prefix = instance_key + "__"
    candidates = sorted(
        d
        for d in job_root.iterdir()
        if d.is_dir() and (d.name == instance_key or d.name.startswith(prefix))
    )
    if len(candidates) == 1:
        p = candidates[0] / "agent" / "trajectory.json"
        if p.exists():
            return p
    return None


def iter_trajectory_files(
    root_dir: Path, allowed_instance_keys: Iterable[str] | None = None
) -> Iterable[Path]:
    if not root_dir.exists():
        raise FileNotFoundError(f"Input root not found: {root_dir}")

    allow = None
    if allowed_instance_keys is not None:
        allow = set(allowed_instance_keys)

    for instance_key in sorted(allow) if allow is not None else []:
        path = resolve_trajectory_path(root_dir, instance_key)
        if path is not None:
            yield path

    if allow is None:
        for child in sorted(root_dir.iterdir()):
            if not child.is_dir():
                continue
            trajectory_path = child / "agent" / "trajectory.json"
            if trajectory_path.exists():
                yield trajectory_path


def main() -> None:
    args = parse_args()
    job_dir = args.job_dir
    output_im_path = args.im_output
    output_lf_path = args.lf_output
    max_records: int | None = args.max_instances
    if max_records is not None and max_records <= 0:
        max_records = None

    output_lf_path.parent.mkdir(parents=True, exist_ok=True)

    results_path = resolve_results_json_path(job_dir)
    trial_keys = load_trial_ids_from_results(load_json(results_path))
    if not trial_keys:
        raise ValueError(
            f"No trial / instance ids found in {results_path} "
            "(expected harbor reward['1.0'] or a list of instance_id)."
        )

    exclusion_patterns = load_exclusion_patterns(args.exclude_repos_file)
    if exclusion_patterns:
        trial_keys = filter_instance_ids_by_repo(
            trial_keys, exclusion_patterns, label="t2-jierun",
        )

    missing: list[str] = []
    for key in trial_keys:
        if resolve_trajectory_path(job_dir, key) is None:
            missing.append(key)

    im_records: list[dict[str, Any]] = []
    failures: list[tuple[str, str]] = []

    for trajectory_path in iter_trajectory_files(job_dir, trial_keys):
        try:
            obj = load_json(trajectory_path)
            all_records = list(iter_records(obj))
            if not all_records:
                raise ValueError("No valid record found")

            for r in all_records:
                im_record = convert_one_record(r)
                im_record["_instance_id"] = trajectory_path.parent.parent.name
                im_records.append(im_record)
                if max_records is not None and len(im_records) >= max_records:
                    break
        except Exception as e:
            failures.append((str(trajectory_path), str(e)))

        if max_records is not None and len(im_records) >= max_records:
            break

    print(f"Results file: {results_path}")
    print(f"Trial ids from results: {len(trial_keys)}, missing dirs: {len(missing)}")
    if missing and len(missing) <= 30:
        for m in missing:
            print(f"  [missing] {m}")
    elif missing:
        for m in missing[:20]:
            print(f"  [missing] {m}")
        print(f"  ... and {len(missing) - 20} more missing")

    im_records = score_dataset(im_records, quiet=True)
    lf_records = [to_lf_record(r) for r in im_records]

    save_jsonl(output_im_path, im_records)

    with output_lf_path.open("w", encoding="utf-8") as f:
        json.dump(lf_records, f, ensure_ascii=False, indent=2)

    print_lf_token_stats(lf_records, stats_output_path=output_lf_path.with_suffix(".stats.json"))

    print(f"Done. IM converted={len(im_records)}, LF converted={len(lf_records)}")
    print(f"IM Output: {output_im_path}")
    print(f"LF Output: {output_lf_path}")
    print(f"Failed files: {len(failures)}")
    if failures:
        for path, err in failures[:20]:
            print(f"[Failed] {path}: {err}")
        if len(failures) > 20:
            print(f"... and {len(failures) - 20} more")


if __name__ == "__main__":
    main()
