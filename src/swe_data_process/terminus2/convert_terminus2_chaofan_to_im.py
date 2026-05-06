import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from swe_data_process.terminus2.common import convert_one_record, iter_records, to_lf_record
from swe_data_process.rule_score import score_dataset
from swe_data_process.utils import EXCLUDED_REPOS_FILE, get_resolved_instances, instance_id_matches_excluded_repos, load_exclusion_patterns, load_json, print_lf_token_stats, save_jsonl


DEFAULT_SOURCE_DIR = Path(
    "/home/ywxzml3j/ywxzml3juser23/harbor/jobs-GLM-5-ready/glm5-oraclesolved/jobs-glm5-oraclesolved-t2-1k-success"
)
DEFAULT_IM_OUTPUT = Path(
    "/home/ywxzml3j/ywxzml3juser57/LLaMA-Factory/data/"
    "chaofan_glm5_swerebench_oraclesolved_t2_1k.jsonl"
)
DEFAULT_LF_OUTPUT = Path(
    "/home/ywxzml3j/ywxzml3juser57/LLaMA-Factory/data/"
    "chaofan_glm5_swerebench_oraclesolved_t2_1k.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将 terminus2 chaofan Harbor 目录下的 agent/trajectory.json 转为 IM / LF 数据"
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help="含多个 trial 子目录的根目录（每实例 agent/trajectory.json）",
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


def iter_trajectory_files(
    root_dir: Path,
    exclusion_patterns: list | None = None,
) -> Iterable[Path]:
    if not root_dir.exists():
        raise FileNotFoundError(f"Input root not found: {root_dir}")

    skipped = 0
    for child in sorted(root_dir.iterdir()):
        if not child.is_dir():
            continue
        if exclusion_patterns and instance_id_matches_excluded_repos(child.name, exclusion_patterns):
            skipped += 1
            continue
        trajectory_path = child / "agent" / "trajectory.json"
        if trajectory_path.exists():
            yield trajectory_path
    if exclusion_patterns and skipped:
        print(f"  [repo-filter] (t2-chaofan) excluded {skipped} dirs")


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir
    output_im_path = args.im_output
    output_lf_path = args.lf_output
    max_instances: int | None = args.max_instances

    output_lf_path.parent.mkdir(parents=True, exist_ok=True)

    im_records: list[dict[str, Any]] = []
    failures: list[tuple[str, str]] = []

    exclusion_patterns = load_exclusion_patterns(args.exclude_repos_file)

    result_json_path = source_dir / "result.json"
    if result_json_path.exists():
        resolved_ids: set[str] | None = set(get_resolved_instances(result_json_path))
        print(f"  [result.json] found {len(resolved_ids)} resolved instances")
    else:
        resolved_ids = None
        print(f"  [result.json] not found, treating all as resolved")

    for trajectory_path in iter_trajectory_files(source_dir, exclusion_patterns):
        if max_instances is not None and len(im_records) >= max_instances:
            break

        instance_id = trajectory_path.parent.parent.name
        if resolved_ids is not None and instance_id not in resolved_ids:
            continue

        try:
            obj = load_json(trajectory_path)
            all_records = list(iter_records(obj))
            if not all_records:
                raise ValueError("No valid record found")

            for r in all_records:
                if max_instances is not None and len(im_records) >= max_instances:
                    break
                im_record = convert_one_record(r)
                im_record["_instance_id"] = trajectory_path.parent.parent.name
                im_records.append(im_record)
        except Exception as e:
            failures.append((str(trajectory_path), str(e)))

        if max_instances is not None and len(im_records) >= max_instances:
            break

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
