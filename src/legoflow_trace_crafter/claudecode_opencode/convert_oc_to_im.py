import argparse
from pathlib import Path
from typing import Any, Literal

from tqdm import tqdm

from legoflow_trace_crafter.claudecode_opencode.extract_and_deduplicate_jsonl import deduplicate_trajectories
from legoflow_trace_crafter.claudecode_opencode.convert_jsonl_to_openai import convert_record
from legoflow_trace_crafter.rule_score import score_dataset
from legoflow_trace_crafter.utils import (
    DEFAULT_TOKENIZER_NAME,
    EXCLUDED_REPOS_FILE,
    ProcessSummary,
    check_roles,
    check_tool_calls,
    check_reasoning_content,
    extract_instance_id_from_config,
    filter_instance_ids_by_repo,
    get_instances_from_job_dir,
    load_exclusion_patterns,
    load_task_metadata_from_trial,
    replace_system_model_name,
    save_jsonl,
    save_lf_json,
    should_keep_instance,
    tag_instance_records,
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert OpenCode trajectories to IM and LF data")
    parser.add_argument("--job-dir", type=Path, required=True, help="Harbor job directory")
    parser.add_argument("--im-output", type=Path, required=True, help="Output IM JSONL file")
    parser.add_argument("--lf-output", type=Path, required=True, help="Output LF JSON file")
    parser.add_argument(
        "--tokenizer-name",
        default=DEFAULT_TOKENIZER_NAME,
        help="Tokenizer name for conversion to LLaMA-Factory ShareGPT JSON",
    )
    parser.add_argument(
        "--max-instances",
        type=int,
        default=None,
        help="Maximum number of instances to process (default: unlimited)",
    )
    parser.add_argument(
        "--instance-status",
        choices=("resolved", "unresolved", "all"),
        default="resolved",
        help="Select resolved, unresolved, or all instances (default: resolved)",
    )
    parser.add_argument(
        "--exclude-repos-file", type=lambda s: Path(s) if s else None,
        default=EXCLUDED_REPOS_FILE,
        help="Path to the curated repository exclusion list bundled with the package",
    )
    parser.add_argument(
        "--reasoning-check-mode",
        choices=("strict", "adaptive"),
        default="adaptive",
        help="reasoning_content validation mode for slow trajectories",
    )
    parser.add_argument(
        "--reasoning-content-ratio-threshold",
        type=float,
        default=0.2,
        help="Minimum fraction of assistant turns with reasoning_content in adaptive mode",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress detailed logs")
    return parser.parse_args()


def _is_subagent_record(record: dict) -> bool:
    """Detect non-primary records such as OpenCode title-generator subcalls."""
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
    reasoning_check_mode: Literal["strict", "adaptive"] = "adaptive",
    reasoning_content_ratio_threshold: float = 0.2,
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
    reasoning_check_mode: Literal["strict", "adaptive"] = "adaptive",
    reasoning_content_ratio_threshold: float = 0.2,
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
                metadata = load_task_metadata_from_trial(job_dir, folder_name)
                tag_instance_records(converted_records, instance_id, metadata)
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
        raise FileNotFoundError(f"Job directory does not exist: {job_dir}")

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

    save_lf_json(args.lf_output, im_data, tokenizer_name=args.tokenizer_name)
    print(f"Saved LF data to: {args.lf_output}")


if __name__ == "__main__":
    main()
