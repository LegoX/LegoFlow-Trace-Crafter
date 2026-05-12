import argparse
from pathlib import Path

from swe_data_process.openhands.common import convert_chaofan_dataset
from swe_data_process.rule_score import score_dataset
from swe_data_process.utils import (
    EXCLUDED_REPOS_FILE,
    load_exclusion_patterns,
    save_jsonl,
    save_lf_json,
)


DEFAULT_SOURCE_DIR = Path(
    "/home/ywxzml3j/ywxzml3juser23/harbor/jobs-GLM-5-ready/glm5-oraclesolved/jobs-glm5-oraclesolved-oh-1k-success"
)
DEFAULT_IM_OUTPUT = Path(
    "/home/ywxzml3j/ywxzml3juser57/LLaMA-Factory/data/"
    "chaofan_glm5_swerebench_oraclesolved_oh_1k.jsonl"
)
DEFAULT_LF_OUTPUT = Path(
    "/home/ywxzml3j/ywxzml3juser57/LLaMA-Factory/data/"
    "chaofan_glm5_swerebench_oraclesolved_oh_1k.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将 OpenHands chaofan Harbor 目录转为 IM（JSONL）与 LLaMA-Factory sharegpt JSON"
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help="含多实例子目录的根目录（每实例 agent/completions/*.json）",
    )
    parser.add_argument(
        "--im-output",
        type=Path,
        default=DEFAULT_IM_OUTPUT,
        help="IM 数据 JSONL（PangUML v2: version/meta_info/tools/messages）",
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


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir
    im_output = args.im_output
    lf_output = args.lf_output

    im_data = convert_chaofan_dataset(
        source_dir,
        max_instances=args.max_instances,
        exclusion_patterns=load_exclusion_patterns(args.exclude_repos_file),
        label="oh-chaofan",
    )

    im_data = score_dataset(im_data, quiet=True)

    save_jsonl(im_output, im_data)
    save_lf_json(lf_output, im_data)

    print(f"IM output: {im_output}")
    print(f"LF output: {lf_output}")


if __name__ == '__main__':
    main()
