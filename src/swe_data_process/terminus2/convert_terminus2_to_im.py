import argparse
import json
from pathlib import Path
from typing import Any

from swe_data_process.terminus2.common import convert_one_record, iter_records, to_lf_record
from swe_data_process.rule_score import score_dataset
from swe_data_process.utils import (
    EXCLUDED_REPOS_FILE,
    extract_instance_id_from_config,
    filter_instance_ids_by_repo,
    get_instances_from_job_dir,
    load_exclusion_patterns,
    load_json,
    print_lf_token_stats,
    save_jsonl,
)


DEFAULT_JOB_DIR = Path(
    "/home/ywxzml3j/ywxzml3juser57/code/harbor-dev/jobs/"
    "swebench_multilingual-100-terminus-2-Qwen3-8B-20260511120629"
)
DEFAULT_IM_OUTPUT = Path(
    "/home/ywxzml3j/ywxzml3juser57/LLaMA-Factory/data/"
    "glm5_swerebench_oraclesolved_t2_1k.jsonl"
)
DEFAULT_LF_OUTPUT = Path(
    "/home/ywxzml3j/ywxzml3juser57/LLaMA-Factory/data/"
    "glm5_swerebench_oraclesolved_t2_1k.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将 terminus2 Harbor job 目录下的 agent/trajectory.json 转为 IM / LF 数据"
    )
    parser.add_argument(
        "--job-dir",
        type=Path,
        default=DEFAULT_JOB_DIR,
        help="Harbor 评测 job 根目录（含 result.json）",
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
        "--instance-status",
        choices=("resolved", "unresolved", "all"),
        default="resolved",
        help="选择处理 resolved、unresolved 或全部实例，默认 resolved",
    )
    parser.add_argument(
        "--exclude-repos-file", type=lambda s: Path(s) if s else None,
        default=EXCLUDED_REPOS_FILE,
        help="排除 repo 列表文件路径（由 generate_excluded_repos.py 生成）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    job_dir = args.job_dir
    if not job_dir.exists():
        raise FileNotFoundError(f"Job 目录不存在: {job_dir}")

    output_im_path = args.im_output
    output_lf_path = args.lf_output
    max_records: int | None = args.max_instances
    if max_records is not None and max_records <= 0:
        max_records = None

    output_lf_path.parent.mkdir(parents=True, exist_ok=True)

    resolved_folders = get_instances_from_job_dir(job_dir, args.instance_status)
    print(f"Total {args.instance_status} instances: {len(resolved_folders)}")

    exclusion_patterns = load_exclusion_patterns(args.exclude_repos_file)
    if exclusion_patterns:
        instance_ids = [extract_instance_id_from_config(job_dir, f) for f in resolved_folders]
        kept_ids = set(filter_instance_ids_by_repo(
            instance_ids, exclusion_patterns, label="t2",
        ))
        resolved_folders = [
            f for f in resolved_folders
            if extract_instance_id_from_config(job_dir, f) in kept_ids
        ]

    im_records: list[dict[str, Any]] = []
    failures: list[tuple[str, str]] = []
    missing: list[str] = []

    for folder_name in resolved_folders:
        instance_id = extract_instance_id_from_config(job_dir, folder_name)
        trajectory_path = job_dir / folder_name / "agent" / "trajectory.json"

        if not trajectory_path.exists():
            missing.append(folder_name)
            continue

        try:
            obj = load_json(trajectory_path)
            all_records = list(iter_records(obj))
            if not all_records:
                raise ValueError("No valid record found")

            for r in all_records:
                im_record = convert_one_record(r)
                im_record["_instance_id"] = instance_id
                im_records.append(im_record)
                if max_records is not None and len(im_records) >= max_records:
                    break
        except Exception as e:
            failures.append((str(trajectory_path), str(e)))

        if max_records is not None and len(im_records) >= max_records:
            break

    print(f"{args.instance_status.capitalize()} folders: {len(resolved_folders)}, missing trajectory: {len(missing)}")
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
