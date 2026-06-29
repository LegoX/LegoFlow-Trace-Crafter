import argparse
from pathlib import Path
from typing import Any, Literal

from tqdm import tqdm

from swe_data_process.claudecode_opencode.extract_and_deduplicate_jsonl import deduplicate_trajectories
from swe_data_process.claudecode_opencode.convert_jsonl_to_openai import convert_record
from swe_data_process.rule_score import score_dataset
from swe_data_process.utils import (
    EXCLUDED_REPOS_FILE,
    ProcessSummary,
    check_roles,
    check_tool_calls,
    check_reasoning_content,
    extract_instance_id_from_config,
    filter_instance_ids_by_repo,
    get_instances_from_job_dir,
    load_exclusion_patterns,
    replace_system_model_name,
    save_jsonl,
    save_lf_json,
    should_keep_instance,
    tag_instance_records,
)


DEFAULT_JOB_DIR = Path(
    "/home/ywxzml3j/ywxzml3juser57/code/harbor-dev/jobs/"
    "swebench-verified-100-custom-opencode-1.2.10-Qwen3-Coder-30B-A3B-Instruct-20260507094527"
)
DEFAULT_IM_OUTPUT = Path(
    "/home/ywxzml3j/ywxzml3juser57/LLaMA-Factory/data/"
    "glm5_swerebench_oraclesolved_oc_1k.jsonl"
)
DEFAULT_LF_OUTPUT = Path(
    "/home/ywxzml3j/ywxzml3juser57/LLaMA-Factory/data/"
    "glm5_swerebench_oraclesolved_oc_1k.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert OpenCode trajectories to IM data")
    parser.add_argument("--job-dir", type=Path, default=DEFAULT_JOB_DIR, help="评测 job 目录")
    parser.add_argument("--im-output", type=Path, default=DEFAULT_IM_OUTPUT, help="输出 IM JSONL 文件")
    parser.add_argument("--lf-output", type=Path, default=DEFAULT_LF_OUTPUT, help="输出 LF JSON 文件")
    parser.add_argument("--max-instances", type=int, default=None, help="最多处理多少个实例，默认不限制")
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
    parser.add_argument(
        "--reasoning-check-mode",
        choices=("strict", "adaptive"),
        default="strict",
        help="slow 轨迹 reasoning_content 过滤模式",
    )
    parser.add_argument(
        "--reasoning-content-ratio-threshold",
        type=float,
        default=0.5,
        help="adaptive 模式下 assistant 轮次包含 reasoning_content 的最低比例",
    )
    parser.add_argument("--quiet", action="store_true", help="关闭详细日志")
    return parser.parse_args()


def _is_subagent_record(record: dict) -> bool:
    """检测非主对话记录（如 OpenCode 的 title generator 子调用）。"""
    messages = record.get("request_body", {}).get("messages", [])
    if not messages:
        return False
    first = messages[0]
    if first.get("role") == "system":
        content = first.get("content", "")
        if "title generator" in content.lower():
            return True
    return False


def process_one_instance(
    folder_name: str,
    job_dir: Path,
    reasoning_check_mode: Literal["strict", "adaptive"] = "strict",
    reasoning_content_ratio_threshold: float = 0.5,
) -> tuple[list[dict], int, int]:
    traj_file = job_dir / folder_name / "agent" / "litellm-trajectory.jsonl"
    records = deduplicate_trajectories(traj_file)
    records = [r for r in records if not _is_subagent_record(r)]

    converted_records: list[dict] = []
    role_filtered = 0
    reasoning_filtered = 0

    for record in records:
        converted_record = convert_record(record)
        replace_system_model_name(converted_record["messages"])

        if not check_roles(converted_record["messages"]):
            role_filtered += 1
            continue

        if not check_tool_calls(converted_record["messages"]):
            role_filtered += 1
            continue

        if not check_reasoning_content(
            converted_record["messages"],
            think_mode=converted_record["think_mode"],
            pseudo_turns=converted_record["pseudo_turns"],
            reasoning_check_mode=reasoning_check_mode,
            reasoning_content_ratio_threshold=reasoning_content_ratio_threshold,
        ):
            reasoning_filtered += 1
            continue

        converted_records.append(converted_record)

    return converted_records, role_filtered, reasoning_filtered


def collect_im_data(
    resolved_folders: list[str],
    job_dir: Path,
    max_instances: int | None,
    quiet: bool,
    reasoning_check_mode: Literal["strict", "adaptive"] = "strict",
    reasoning_content_ratio_threshold: float = 0.5,
) -> tuple[list[dict[str, Any]], ProcessSummary]:
    im_data: list[dict[str, Any]] = []
    summary = ProcessSummary()

    kept_folder_count = 0
    for folder_name in tqdm(resolved_folders, desc="Processing instances"):
        if max_instances is not None and kept_folder_count >= max_instances:
            break

        instance_id = extract_instance_id_from_config(job_dir, folder_name)
        try:
            converted_records, role_filtered, reasoning_filtered = process_one_instance(
                folder_name,
                job_dir,
                reasoning_check_mode=reasoning_check_mode,
                reasoning_content_ratio_threshold=reasoning_content_ratio_threshold,
            )
            summary.role_filtered += role_filtered
            summary.reasoning_filtered += reasoning_filtered

            if converted_records and should_keep_instance(role_filtered, reasoning_filtered):
                tag_instance_records(converted_records, instance_id)
                im_data.extend(converted_records)
                kept_folder_count += 1
        except Exception as e:
            summary.failed_instances += 1
            if not quiet:
                print(f"[WARN] instance {instance_id} failed: {e}")

    return im_data, summary


def main() -> None:
    args = parse_args()
    job_dir = args.job_dir
    if not job_dir.exists():
        raise FileNotFoundError(f"Job 目录不存在: {job_dir}")

    resolved_folders = get_instances_from_job_dir(job_dir, args.instance_status)
    print(f"Total {args.instance_status} instances: {len(resolved_folders)}")

    exclusion_patterns = load_exclusion_patterns(args.exclude_repos_file)
    if exclusion_patterns:
        instance_ids = [extract_instance_id_from_config(job_dir, f) for f in resolved_folders]
        kept_ids = set(filter_instance_ids_by_repo(
            instance_ids, exclusion_patterns, label="oc",
        ))
        resolved_folders = [
            f for f in resolved_folders
            if extract_instance_id_from_config(job_dir, f) in kept_ids
        ]

    im_data, summary = collect_im_data(
        resolved_folders=resolved_folders,
        job_dir=job_dir,
        max_instances=args.max_instances,
        quiet=args.quiet,
        reasoning_check_mode=args.reasoning_check_mode,
        reasoning_content_ratio_threshold=args.reasoning_content_ratio_threshold,
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
