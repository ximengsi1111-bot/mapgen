#!/usr/bin/env python3
"""Count lane and intersection type distributions in JSONL training data."""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterator


CounterKey = tuple[str, str]


def iter_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    """Yield non-empty JSONL records together with their source line numbers."""
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            yield line_number, json.loads(line)


def get_gpt_payloads(sample: dict[str, Any]) -> Iterator[str | dict[str, Any]]:
    """Yield JSON payloads from all GPT/assistant messages in a sample."""
    for message in sample.get("conversations", []):
        if message.get("from") in {"gpt", "assistant"}:
            value = message.get("value")
            if isinstance(value, (str, dict)):
                yield value


def display_value(value: Any) -> str:
    """Format a field value without conflating values of different types."""
    return f"{json.dumps(value, ensure_ascii=False)} ({type(value).__name__})"


def add_value(
    counts: Counter[CounterKey],
    display_names: dict[CounterKey, str],
    value: Any,
) -> None:
    """Add a possibly non-hashable JSON value to a typed counter."""
    key = (type(value).__name__, json.dumps(value, ensure_ascii=False, sort_keys=True))
    counts[key] += 1
    display_names[key] = display_value(value)


def print_distribution(
    field_name: str,
    counts: Counter[CounterKey],
    display_names: dict[CounterKey, str],
) -> None:
    """Print counts and percentages for one field."""
    counted = sum(counts.values())
    print(f"\n{field_name} 分布:")
    if not counts:
        print(f"  （未找到 {field_name}）")
        return

    for key, count in counts.most_common():
        ratio = count / counted * 100
        print(f"  {display_names[key]:<20} {count:>8}  {ratio:>7.2f}%")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "统计 JSONL 训练数据中 GPT 标注里的 lane_type、intersection_type "
            "和 intersection_subtype 值分布。"
        )
    )
    parser.add_argument("data_path", type=Path, help="JSONL 数据文件路径")
    args = parser.parse_args()

    field_names = ("lane_type", "intersection_type", "intersection_subtype")
    counts: dict[str, Counter[CounterKey]] = {
        name: Counter() for name in field_names
    }
    display_names: dict[str, dict[CounterKey, str]] = {
        name: {} for name in field_names
    }
    missing_counts = {name: 0 for name in field_names}
    sample_count = 0
    centerline_count = 0
    intersection_count = 0
    invalid_payloads: list[str] = []

    try:
        for line_number, sample in iter_jsonl(args.data_path):
            sample_count += 1
            for payload_index, payload in enumerate(get_gpt_payloads(sample), start=1):
                if isinstance(payload, str):
                    try:
                        payload = json.loads(payload)
                    except json.JSONDecodeError as error:
                        invalid_payloads.append(
                            f"line {line_number}, GPT payload {payload_index}: {error}"
                        )
                        continue

                lines = payload.get("lines", [])
                if not isinstance(lines, list):
                    invalid_payloads.append(
                        f"line {line_number}, GPT payload {payload_index}: "
                        "'lines' is not a list"
                    )
                    continue

                for line in lines:
                    if not isinstance(line, dict):
                        continue
                    category = line.get("category")
                    if category == "centerline":
                        centerline_count += 1
                        if "lane_type" in line:
                            add_value(
                                counts["lane_type"],
                                display_names["lane_type"],
                                line["lane_type"],
                            )
                        else:
                            missing_counts["lane_type"] += 1
                    elif category == "intersection":
                        intersection_count += 1
                        for field_name in ("intersection_type", "intersection_subtype"):
                            if field_name in line:
                                add_value(
                                    counts[field_name],
                                    display_names[field_name],
                                    line[field_name],
                                )
                            else:
                                missing_counts[field_name] += 1
    except FileNotFoundError:
        parser.error(f"文件不存在: {args.data_path}")
    except json.JSONDecodeError as error:
        parser.error(f"JSONL 第 {error.lineno} 行不是合法 JSON: {error.msg}")

    print(f"文件: {args.data_path}")
    print(f"样本数: {sample_count}")
    print(f"centerline 总数: {centerline_count}")
    print(f"intersection 总数: {intersection_count}")
    for field_name in field_names:
        print(
            f"{field_name}: 已统计 {sum(counts[field_name].values())}, "
            f"缺失 {missing_counts[field_name]}"
        )
        print_distribution(field_name, counts[field_name], display_names[field_name])

    if invalid_payloads:
        print(f"\n无法解析的 GPT 标注数: {len(invalid_payloads)}")
        for error in invalid_payloads[:10]:
            print(f"  - {error}")
        if len(invalid_payloads) > 10:
            print(f"  ... 另有 {len(invalid_payloads) - 10} 条")


if __name__ == "__main__":
    main()
