import argparse
import json
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
    filter_instance_ids_by_repo,
    get_resolved_instances,
    load_exclusion_patterns,
    save_jsonl,
    save_lf_json,
    should_keep_instance,
    tag_instance_records,
)


DEFAULT_JOB_DIR = Path(
    "/home/ywxzml3j/ywxzml3juser30/code/harbor/jobs/"
    "swerebench-filtered-oraclesolved-claude-code-2.1.62-GLM-5-FP8-2-20260321233207"
)
DEFAULT_TRAJ_DIR = Path(f"{DEFAULT_JOB_DIR}_trajs_via_logger")
DEFAULT_IM_OUTPUT = Path(
    "/home/ywxzml3j/ywxzml3juser57/LLaMA-Factory/data/"
    "jierun_glm5_swerebench_oraclesolved_cc_1k.jsonl"
)
DEFAULT_LF_OUTPUT = Path(
    "/home/ywxzml3j/ywxzml3juser57/LLaMA-Factory/data/"
    "jierun_glm5_swerebench_oraclesolved_cc_1k.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Claude Code trajectories to IM data")
    parser.add_argument("--job-dir", type=Path, default=DEFAULT_JOB_DIR, help="评测 job 目录")
    parser.add_argument("--trajs-dir", type=Path, default=DEFAULT_TRAJ_DIR, help="logger 轨迹目录")
    parser.add_argument("--im-output", type=Path, default=DEFAULT_IM_OUTPUT, help="输出 IM JSONL 文件")
    parser.add_argument("--lf-output", type=Path, default=DEFAULT_LF_OUTPUT, help="输出 LF JSON 文件")
    parser.add_argument("--max-instances", type=int, default=None, help="最多处理多少个实例，默认不限制")
    parser.add_argument(
        "--exclude-repos-file", type=lambda s: Path(s) if s else None,
        default=EXCLUDED_REPOS_FILE,
        help="排除 repo 列表文件路径（由 generate_excluded_repos.py 生成）",
    )
    parser.add_argument("--quiet", action="store_true", help="关闭详细日志")
    return parser.parse_args()


def validate_paths(job_dir: Path, trajs_dir: Path) -> None:
    result_json = job_dir / "result.json"
    if not result_json.exists():
        raise FileNotFoundError(f"result.json 不存在: {result_json}")
    if not trajs_dir.exists():
        raise FileNotFoundError(f"轨迹目录不存在: {trajs_dir}")


def process_one_instance(instance_id: str, traj_path: Path, trajs_via_logger_path: Path) -> tuple[list[dict], int, int]:
    config_file = traj_path / instance_id / "config.json"
    with config_file.open("r", encoding="utf-8") as f:
        config = json.load(f)

    extracted_instance_id = Path(config["task"]["path"]).name
    trajs_via_logger_file = trajs_via_logger_path / f"{extracted_instance_id}.jsonl"
    records = deduplicate_trajectories(trajs_via_logger_file)

    converted_records: list[dict] = []
    role_filtered = 0
    reasoning_filtered = 0

    for record in records:
        converted_record = convert_record(record)

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

    return converted_records, role_filtered, reasoning_filtered


def collect_im_data(
    resolved_instances: list[str],
    job_dir: Path,
    trajs_dir: Path,
    max_instances: int | None,
    quiet: bool,
) -> tuple[list[dict[str, Any]], ProcessSummary]:
    im_data: list[dict[str, Any]] = []
    summary = ProcessSummary()

    kept_instance_count = 0
    for instance_id in tqdm(resolved_instances, desc="Processing instances"):
        if max_instances is not None and kept_instance_count >= max_instances:
            break

        try:
            converted_records, role_filtered, reasoning_filtered = process_one_instance(
                instance_id, job_dir, trajs_dir
            )
            summary.role_filtered += role_filtered
            summary.reasoning_filtered += reasoning_filtered

            if should_keep_instance(role_filtered, reasoning_filtered):
                tag_instance_records(converted_records, instance_id)
                im_data.extend(converted_records)
                kept_instance_count += 1
        except Exception as e:
            summary.failed_instances += 1
            if not quiet:
                print(f"[WARN] instance {instance_id} failed: {e}")

    return im_data, summary


def main() -> None:
    args = parse_args()
    validate_paths(args.job_dir, args.trajs_dir)

    result_json = args.job_dir / "result.json"
    resolved_instances = get_resolved_instances(result_json)
    print(f"Total resolved instances: {len(resolved_instances)}")

    exclusion_patterns = load_exclusion_patterns(args.exclude_repos_file)
    if exclusion_patterns:
        resolved_instances = filter_instance_ids_by_repo(
            resolved_instances, exclusion_patterns, label="cc-jierun",
        )

    im_data, summary = collect_im_data(
        resolved_instances=resolved_instances,
        job_dir=args.job_dir,
        trajs_dir=args.trajs_dir,
        max_instances=args.max_instances,
        quiet=args.quiet,
    )

    im_data = score_dataset(im_data, quiet=True)

    save_jsonl(args.im_output, im_data)

    print(f"Total converted records for IM: {len(im_data)}")
    print(f"Filtered by roles: {summary.role_filtered}")
    print(f"Filtered by reasoning: {summary.reasoning_filtered}")
    print(f"Failed instances: {summary.failed_instances}")
    print(f"Saved to: {args.im_output}")

    save_lf_json(args.lf_output, im_data)
    print(f"Saved LF data to: {args.lf_output}")


if __name__ == "__main__":
    main()
