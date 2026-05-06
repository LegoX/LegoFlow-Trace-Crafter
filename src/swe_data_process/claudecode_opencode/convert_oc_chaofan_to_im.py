"""
现在对于 opencode 的 chaofan 数据，都存放在
/home/ywxzml3j/ywxzml3juser23/harbor/jobs-glm5-oraclesolved-oc-1k-success 文件夹中。
每个子文件夹的 /agent/trajectory_api.jsonl 扮演了 trajs_via_logger_file 的作用。
"""

import argparse
from pathlib import Path
from typing import Any

from tqdm import tqdm

from swe_data_process.claudecode_opencode.extract_and_deduplicate_jsonl import deduplicate_trajectories
from swe_data_process.claudecode_opencode.convert_jsonl_to_openai import convert_record
from swe_data_process.rule_score import score_dataset
from swe_data_process.utils import (
    EXCLUDED_REPOS_FILE,
    ProcessSummary,
    check_roles,
    check_reasoning_content,
    filter_paths_by_repo,
    get_resolved_instances,
    load_exclusion_patterns,
    replace_system_model_name,
    save_jsonl,
    save_lf_json,
    should_keep_instance,
    tag_instance_records,
)


DEFAULT_SOURCE_DIR = Path(
    "/home/ywxzml3j/ywxzml3juser23/harbor/jobs-GLM-5-ready/glm5-oraclesolved/jobs-glm5-oraclesolved-oc-1k-success"
)
DEFAULT_IM_OUTPUT = Path(
    "/home/ywxzml3j/ywxzml3juser57/LLaMA-Factory/data/chaofan_glm5_swerebench_oraclesolved_oc_1k.jsonl"
)
DEFAULT_LF_OUTPUT = Path(
    "/home/ywxzml3j/ywxzml3juser57/LLaMA-Factory/data/chaofan_glm5_swerebench_oraclesolved_oc_1k.json"
)

TRAJECTORY_API_REL = Path("agent") / "trajectory_api.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将 opencode chaofan 目录下的 trajectory_api.jsonl 转为 IM 数据"
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help="含多个实例子目录的根目录（每实例 agent/trajectory_api.jsonl）",
    )
    parser.add_argument("--im-output", type=Path, default=DEFAULT_IM_OUTPUT, help="输出 IM JSONL 文件")
    parser.add_argument("--lf-output", type=Path, default=DEFAULT_LF_OUTPUT, help="输出 LF JSON 文件")
    parser.add_argument(
        "--max-instances",
        type=int,
        default=None,
        help="最多保留多少个成功转换的实例，默认不限制",
    )
    parser.add_argument("--quiet", action="store_true", help="关闭详细日志")
    parser.add_argument(
        "--exclude-repos-file", type=lambda s: Path(s) if s else None,
        default=EXCLUDED_REPOS_FILE,
        help="排除 repo 列表文件路径（由 generate_excluded_repos.py 生成）",
    )
    return parser.parse_args()


def validate_source_dir(source_dir: Path) -> None:
    if not source_dir.exists():
        raise FileNotFoundError(f"source-dir 不存在: {source_dir}")
    if not source_dir.is_dir():
        raise NotADirectoryError(f"source-dir 不是目录: {source_dir}")


def collect_instance_dirs(source_dir: Path) -> list[Path]:
    validate_source_dir(source_dir)
    return sorted(p for p in source_dir.iterdir() if p.is_dir())


def process_one_instance(trajectory_api_path: Path) -> tuple[list[dict], int, int]:
    records = deduplicate_trajectories(trajectory_api_path)

    converted_records: list[dict] = []
    role_filtered = 0
    reasoning_filtered = 0

    for record in records:
        converted_record = convert_record(record)
        replace_system_model_name(converted_record["messages"])

        if not check_roles(converted_record["messages"]):
            role_filtered += 1
            continue

        if not check_reasoning_content(
            converted_record["messages"],
            think_mode=converted_record["think_mode"],
            pseudo_turns=converted_record["pseudo_turns"],
        ):
            reasoning_filtered += 1
            continue

        converted_records.append(converted_record)

    tag_instance_records(converted_records, trajectory_api_path.parent.parent.name)
    return converted_records, role_filtered, reasoning_filtered


def collect_im_data(
    instance_dirs: list[Path],
    max_instances: int | None,
    quiet: bool,
) -> tuple[list[dict[str, Any]], ProcessSummary]:
    im_data: list[dict[str, Any]] = []
    summary = ProcessSummary()

    kept_instance_count = 0
    for instance_dir in tqdm(instance_dirs, desc="Processing instances"):
        if max_instances is not None and kept_instance_count >= max_instances:
            break

        traj_path = instance_dir / TRAJECTORY_API_REL
        if not traj_path.is_file():
            summary.failed_instances += 1
            if not quiet:
                print(f"[WARN] {instance_dir.name}: 缺少 {TRAJECTORY_API_REL}")
            continue

        try:
            converted_records, role_filtered, reasoning_filtered = process_one_instance(traj_path)
            summary.role_filtered += role_filtered
            summary.reasoning_filtered += reasoning_filtered

            if converted_records and should_keep_instance(role_filtered, reasoning_filtered):
                im_data.extend(converted_records)
                kept_instance_count += 1
        except Exception as e:
            summary.failed_instances += 1
            if not quiet:
                print(f"[WARN] instance {instance_dir.name} failed: {e}")

    return im_data, summary


def main() -> None:
    args = parse_args()
    instance_dirs = collect_instance_dirs(args.source_dir)
    print(f"Total instance dirs: {len(instance_dirs)}")

    result_json_path = args.source_dir / "result.json"
    if result_json_path.exists():
        resolved_ids = set(get_resolved_instances(result_json_path))
        before = len(instance_dirs)
        instance_dirs = [p for p in instance_dirs if p.name in resolved_ids]
        print(f"  [result.json] kept {len(instance_dirs)}/{before} resolved instances")
    else:
        print(f"  [result.json] not found, treating all as resolved")

    exclusion_patterns = load_exclusion_patterns(args.exclude_repos_file)
    if exclusion_patterns:
        instance_dirs = filter_paths_by_repo(
            instance_dirs, exclusion_patterns,
            name_func=lambda p: p.name, label="oc-chaofan",
        )

    im_data, summary = collect_im_data(
        instance_dirs=instance_dirs,
        max_instances=args.max_instances,
        quiet=args.quiet,
    )

    im_data = score_dataset(im_data, quiet=True)

    save_jsonl(args.im_output, im_data)

    print(f"Total converted records for IM: {len(im_data)}")
    print(f"Filtered by roles: {summary.role_filtered}")
    print(f"Filtered by reasoning: {summary.reasoning_filtered}")
    print(f"Failed / skipped instances: {summary.failed_instances}")
    print(f"Saved to: {args.im_output}")

    save_lf_json(args.lf_output, im_data)
    print(f"Saved LF data to: {args.lf_output}")


if __name__ == "__main__":
    main()
