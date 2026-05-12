import argparse
import json
from pathlib import Path
from typing import Any

from tqdm import tqdm

from swe_data_process.openhands.common import extract_text, process_tool_call
from swe_data_process.rule_score import score_dataset
from swe_data_process.utils import (
    EXCLUDED_REPOS_FILE,
    check_roles,
    check_reasoning_content,
    filter_instance_ids_by_repo,
    get_resolved_instances,
    load_exclusion_patterns,
    save_jsonl,
    save_lf_json,
)


DEFAULT_JOB_DIR = Path(
    "/home/ywxzml3j/ywxzml3juser30/code/harbor/jobs/"
    "swerebench-filtered-oraclesolved-openhands-sdk-1.14.0-GLM-5-FP8-30-20260322001450"
)
DEFAULT_TRAJ_DIR = Path(f"{DEFAULT_JOB_DIR}_trajs_via_logger")
DEFAULT_IM_OUTPUT = Path(
    "/home/ywxzml3j/ywxzml3juser57/LLaMA-Factory/data/"
    "jierun_glm5_swerebench_oraclesolved_oh_sdk_1k.jsonl"
)
DEFAULT_LF_OUTPUT = Path(
    "/home/ywxzml3j/ywxzml3juser57/LLaMA-Factory/data/"
    "jierun_glm5_swerebench_oraclesolved_oh_sdk_1k.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将 OpenHands SDK jierun Harbor job + logger 轨迹转为 IM（JSONL）与 LF JSON"
    )
    parser.add_argument(
        "--job-dir",
        type=Path,
        default=DEFAULT_JOB_DIR,
        help="Harbor job 目录（含 result.json）",
    )
    parser.add_argument(
        "--trajs-dir",
        type=Path,
        default=DEFAULT_TRAJ_DIR,
        help="logger 导出的 jsonl 目录",
    )
    parser.add_argument(
        "--im-output",
        type=Path,
        default=DEFAULT_IM_OUTPUT,
        help="IM 数据 JSONL",
    )
    parser.add_argument(
        "--lf-output",
        type=Path,
        default=DEFAULT_LF_OUTPUT,
        help="LF sharegpt 格式 JSON",
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


def _normalize_message(msg: dict[str, Any]) -> dict[str, Any]:
    role = msg.get('role')
    if role in ('system', 'user'):
        return {
            'role': role,
            'content': extract_text(msg.get('content')),
        }
    if role == 'tool':
        normalized = {
            'role': 'tool',
            'content': extract_text(msg.get('content')),
        }
        tool_call_id = msg.get('tool_call_id')
        if isinstance(tool_call_id, str) and tool_call_id:
            normalized['tool_call_id'] = tool_call_id
        return normalized
    if role == 'assistant':
        out: dict[str, Any] = {
            'role': 'assistant',
            'content': extract_text(msg.get('content')),
            'tool_calls': process_tool_call(msg.get('tool_calls', [])),
        }
        if msg.get('reasoning_content') is not None:
            out['reasoning_content'] = msg['reasoning_content']
        return out
    return {
        'role': role,
        'content': extract_text(msg.get('content')),
    }


def build_messages_from_logger_record(record: dict[str, Any]) -> tuple[list[dict[str, Any]] | None, str | None]:
    """request_body.messages + final choice message, normalized like chaofan pipeline."""
    raw_messages = record['request_body']['messages']
    messages = [_normalize_message(m) for m in raw_messages]

    response_body = record.get('response_body') or {}
    choices = response_body.get('choices') or []
    if not choices:
        return None, 'no_choices'

    messages.append(_normalize_message(choices[0]['message']))
    return messages, None


def read_last_jsonl_record(jsonl_path: Path) -> dict[str, Any] | None:
    last = None
    with jsonl_path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                last = json.loads(line)
    return last


def convert_dataset(
    traj_path: Path,
    trajs_via_logger_path: Path,
    max_samples: int | None = None,
    exclusion_patterns: list | None = None,
) -> list[dict[str, Any]]:
    result_json_path = traj_path / "result.json"
    resolved_instances = get_resolved_instances(result_json_path)
    if exclusion_patterns:
        resolved_instances = filter_instance_ids_by_repo(
            resolved_instances, exclusion_patterns, label="oh-sdk-jierun",
        )
    im_data: list[dict[str, Any]] = []
    skipped_no_logger = 0
    skipped_no_choices = 0
    skipped_invalid = 0

    for instance_id in tqdm(resolved_instances):
        config_path = traj_path / instance_id / 'config.json'
        try:
            with config_path.open('r', encoding='utf-8') as f:
                config = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            skipped_no_logger += 1
            print(f"跳过 {instance_id}: 无法读取 config.json — {e}")
            continue

        extracted_instance_id = Path(config['task']['path']).name
        jsonl_path = trajs_via_logger_path / f'{extracted_instance_id}.jsonl'

        try:
            trajs_via_logger = read_last_jsonl_record(jsonl_path)
        except FileNotFoundError:
            skipped_no_logger += 1
            print(f"File not found for instance {instance_id}: {jsonl_path}")
            continue
        except (json.JSONDecodeError, OSError) as e:
            skipped_invalid += 1
            print(f"读取 logger 失败 {instance_id}: {e}")
            continue

        if trajs_via_logger is None:
            skipped_no_logger += 1
            print(f"Empty jsonl for instance {instance_id}: {jsonl_path}")
            continue

        try:
            messages, err = build_messages_from_logger_record(trajs_via_logger)
        except (KeyError, TypeError, IndexError) as e:
            skipped_invalid += 1
            print(f"解析轨迹失败 {instance_id}: {e}")
            continue

        if err == 'no_choices':
            skipped_no_choices += 1
            print(f"No choices found in response for instance {instance_id}")
            continue

        tools = trajs_via_logger['request_body'].get('tools', [])

        if not check_roles(messages):
            skipped_invalid += 1
            continue

        if not check_reasoning_content(messages, think_mode='slow', pseudo_turns=None):
            skipped_invalid += 1
            print(f"Instance {instance_id} failed reasoning content check.")
            continue

        im_data.append({
            'messages': messages,
            'tools': tools,
            'pseudo_turns': None,
            'think_mode': 'slow',
            '_instance_id': instance_id,
        })

        if max_samples is not None and len(im_data) >= max_samples:
            break

    print(f"Total sampled instances: {len(im_data)}")
    print(f"Skipped (no config / logger file / empty jsonl): {skipped_no_logger}")
    print(f"Skipped (no choices in response): {skipped_no_choices}")
    print(f"Skipped (invalid / roles / reasoning): {skipped_invalid}")
    return im_data


def main() -> None:
    args = parse_args()
    job_dir = args.job_dir
    trajs_dir = args.trajs_dir

    max_samples: int | None = args.max_instances
    if max_samples is not None and max_samples <= 0:
        max_samples = None

    im_output = args.im_output
    lf_output = args.lf_output

    im_data = convert_dataset(
        job_dir,
        trajs_dir,
        max_samples=max_samples,
        exclusion_patterns=load_exclusion_patterns(args.exclude_repos_file),
    )

    im_data = score_dataset(im_data, quiet=True)

    save_jsonl(im_output, im_data)
    save_lf_json(lf_output, im_data)

    print(f"IM output: {im_output}")
    print(f"LF output: {lf_output}")


if __name__ == '__main__':
    main()
