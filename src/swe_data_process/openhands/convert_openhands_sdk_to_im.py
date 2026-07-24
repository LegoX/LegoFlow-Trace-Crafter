import argparse
import json
from pathlib import Path
from typing import Any, Literal

from tqdm import tqdm

from swe_data_process.openhands.common import extract_text, process_tool_call
from swe_data_process.rule_score import score_dataset
from swe_data_process.utils import (
    DEFAULT_TOKENIZER_NAME,
    EXCLUDED_REPOS_FILE,
    check_roles,
    check_tool_calls,
    check_reasoning_content,
    extract_instance_id_from_config,
    filter_instance_ids_by_repo,
    get_instances_from_job_dir,
    InstanceStatus,
    load_exclusion_patterns,
    load_task_metadata_from_trial,
    save_jsonl,
    save_lf_json,
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将 OpenHands SDK Harbor job 轨迹转为 IM（JSONL）与 LF JSON"
    )
    parser.add_argument(
        "--job-dir",
        type=Path,
        required=True,
        help="Harbor job 目录",
    )
    parser.add_argument(
        "--im-output",
        type=Path,
        required=True,
        help="输出 IM JSONL 文件",
    )
    parser.add_argument(
        "--lf-output",
        type=Path,
        required=True,
        help="输出 LF JSON 文件",
    )
    parser.add_argument(
        "--tokenizer-name",
        default=DEFAULT_TOKENIZER_NAME,
        help="转换为 LLaMA-Factory sharegpt 格式 JSON 时使用的 tokenizer 名称",
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
    parser.add_argument(
        "--reasoning-check-mode",
        choices=("strict", "adaptive"),
        default="adaptive",
        help="slow 轨迹 reasoning_content 过滤模式",
    )
    parser.add_argument(
        "--reasoning-content-ratio-threshold",
        type=float,
        default=0.2,
        help="adaptive 模式下 assistant 轮次包含 reasoning_content 的最低比例",
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
            rc = msg['reasoning_content']
            out['reasoning_content'] = rc.strip() if isinstance(rc, str) else rc
        return out
    return {
        'role': role,
        'content': extract_text(msg.get('content')),
    }


def build_messages_from_logger_record(record: dict[str, Any]) -> tuple[list[dict[str, Any]] | None, str | None]:
    """request_body.messages + final choice message, normalized."""
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


def read_last_successful_jsonl_record(jsonl_path: Path) -> dict[str, Any] | None:
    """Read the last LiteLLM logger record with ``success != False``.

    LiteLLM writes ``success: false`` on HTTP failure / upstream rejection / 5xx
    / timeout. The OpenHands SDK harness keeps retrying after such failures, so
    a trailing run of failed lines can shadow the real final assistant turn.
    We treat a missing ``success`` field as ``True`` (older logger versions).
    Returns ``None`` if the file is empty or every line failed.
    """
    last = None
    with jsonl_path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get('success') is False:
                continue
            last = record
    return last


def _infer_think_mode(messages: list[dict[str, Any]]) -> str:
    """根据 assistant 消息是否包含 reasoning_content 推断 think_mode。"""
    for msg in messages:
        if msg.get("role") == "assistant" and msg.get("reasoning_content"):
            return "slow"
    return "fast"


def convert_dataset(
    job_dir: Path,
    max_samples: int | None = None,
    exclusion_patterns: list | None = None,
    instance_status: InstanceStatus = "resolved",
    *,
    reasoning_check_mode: Literal["strict", "adaptive"] = "adaptive",
    reasoning_content_ratio_threshold: float = 0.2,
) -> list[dict[str, Any]]:
    resolved_folders = get_instances_from_job_dir(job_dir, instance_status)
    print(f"Total {instance_status} instances: {len(resolved_folders)}")
    if exclusion_patterns:
        instance_ids = [extract_instance_id_from_config(job_dir, f) for f in resolved_folders]
        kept_ids = set(filter_instance_ids_by_repo(
            instance_ids, exclusion_patterns, label="oh-sdk",
        ))
        resolved_folders = [
            f for f in resolved_folders
            if extract_instance_id_from_config(job_dir, f) in kept_ids
        ]

    im_data: list[dict[str, Any]] = []
    skipped_no_traj = 0
    skipped_no_choices = 0
    skipped_invalid = 0
    skipped_all_failed = 0

    for folder_name in tqdm(resolved_folders):
        instance_id = extract_instance_id_from_config(job_dir, folder_name)
        traj_file = job_dir / folder_name / "agent" / "litellm-trajectory.jsonl"

        try:
            last_record = read_last_successful_jsonl_record(traj_file)
        except FileNotFoundError:
            skipped_no_traj += 1
            print(f"File not found for instance {instance_id}: {traj_file}")
            continue
        except (json.JSONDecodeError, OSError) as e:
            skipped_invalid += 1
            print(f"读取轨迹失败 {instance_id}: {e}")
            continue

        if last_record is None:
            skipped_all_failed += 1
            print(f"No successful records (all success=false / empty) for instance {instance_id}: {traj_file}")
            continue

        try:
            messages, err = build_messages_from_logger_record(last_record)
        except (KeyError, TypeError, IndexError) as e:
            skipped_invalid += 1
            print(f"解析轨迹失败 {instance_id}: {e}")
            continue

        if err == 'no_choices':
            skipped_no_choices += 1
            print(f"No choices found in response for instance {instance_id}")
            continue

        request_body = last_record.get('request_body') or {}
        tools = request_body.get('tools', [])
        think_mode = _infer_think_mode(messages)

        if not check_roles(messages):
            skipped_invalid += 1
            continue

        if not check_tool_calls(messages):
            skipped_invalid += 1
            continue

        if not check_reasoning_content(
            messages,
            think_mode=think_mode,
            pseudo_turns=None,
            reasoning_check_mode=reasoning_check_mode,
            reasoning_content_ratio_threshold=reasoning_content_ratio_threshold,
        ):
            skipped_invalid += 1
            print(f"Instance {instance_id} failed reasoning content check.")
            continue

        gen_params = {
            key: request_body.get(key)
            for key in ('model', 'max_tokens', 'top_p', 'temperature')
            if key in request_body
        }
        usage = last_record.get('usage')

        try:
            metadata = load_task_metadata_from_trial(job_dir, folder_name)
        except (OSError, ValueError, KeyError, TypeError) as e:
            skipped_invalid += 1
            print(f"读取 task metadata 失败 {instance_id}: {e}")
            continue

        im_data.append({
            'messages': messages,
            'tools': tools,
            'pseudo_turns': None,
            'think_mode': think_mode,
            '_instance_id': instance_id,
            '_gen_params': gen_params,
            '_usage': usage,
            '_instance_metadata': metadata,
        })

        if max_samples is not None and len(im_data) >= max_samples:
            break

    print(f"Total sampled instances: {len(im_data)}")
    print(f"Skipped (no traj file / empty jsonl): {skipped_no_traj}")
    print(f"Skipped (all lines success=false): {skipped_all_failed}")
    print(f"Skipped (no choices in response): {skipped_no_choices}")
    print(f"Skipped (invalid / roles / reasoning): {skipped_invalid}")
    return im_data


def main() -> None:
    args = parse_args()
    job_dir = args.job_dir
    if not job_dir.exists():
        raise FileNotFoundError(f"Job 目录不存在: {job_dir}")

    max_samples: int | None = args.max_instances
    if max_samples is not None and max_samples <= 0:
        max_samples = None

    im_data = convert_dataset(
        job_dir,
        max_samples=max_samples,
        exclusion_patterns=load_exclusion_patterns(args.exclude_repos_file),
        instance_status=args.instance_status,
        reasoning_check_mode=args.reasoning_check_mode,
        reasoning_content_ratio_threshold=args.reasoning_content_ratio_threshold,
    )

    im_data = score_dataset(im_data, quiet=True)

    save_jsonl(args.im_output, im_data)
    save_lf_json(args.lf_output, im_data, tokenizer_name=args.tokenizer_name)

    print(f"IM output: {args.im_output}")
    print(f"LF output: {args.lf_output}")


if __name__ == '__main__':
    main()
