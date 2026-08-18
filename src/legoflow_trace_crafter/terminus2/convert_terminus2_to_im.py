import argparse
import json
from pathlib import Path
from typing import Any

from legoflow_trace_crafter.terminus2.common import convert_one_record, iter_records, to_lf_record
from legoflow_trace_crafter.rule_score import score_dataset
from legoflow_trace_crafter.utils import (
    DEFAULT_TOKENIZER_NAME,
    EXCLUDED_REPOS_FILE,
    extract_instance_id_from_config,
    filter_instance_ids_by_repo,
    get_instances_from_job_dir,
    load_exclusion_patterns,
    load_json,
    load_task_metadata_from_trial,
    print_lf_token_stats,
    save_jsonl,
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert agent/trajectory.json files in a Terminus2 Harbor job to IM and LF data"
    )
    parser.add_argument(
        "--job-dir",
        type=Path,
        required=True,
        help="Harbor job directory",
    )
    parser.add_argument(
        "--im-output",
        type=Path,
        required=True,
        help="Output IM JSONL file",
    )
    parser.add_argument(
        "--lf-output",
        type=Path,
        required=True,
        help="Output LF JSON file",
    )
    parser.add_argument(
        "--tokenizer-name",
        default=DEFAULT_TOKENIZER_NAME,
        help="Tokenizer name for conversion to LLaMA-Factory ShareGPT JSON",
    )
    parser.add_argument(
        "--max-instances",
        type=int,
        default=None,
        help="Maximum number of successfully converted records (default: unlimited)",
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    job_dir = args.job_dir
    if not job_dir.exists():
        raise FileNotFoundError(f"Job directory does not exist: {job_dir}")

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
            metadata = load_task_metadata_from_trial(job_dir, folder_name)
            obj = load_json(trajectory_path)
            all_records = list(iter_records(obj))
            if not all_records:
                raise ValueError("No valid record found")

            for r in all_records:
                im_record = convert_one_record(r)
                im_record["_instance_id"] = instance_id
                im_record["_instance_metadata"] = metadata
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

    print_lf_token_stats(
        lf_records,
        tokenizer_name=args.tokenizer_name,
        stats_output_path=output_lf_path.with_suffix(".stats.json"),
    )

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
