"""Validate and summarize the four-suite LIBERO evaluation logs."""

import argparse
import csv
import json
from collections import OrderedDict
from pathlib import Path
import re


SUITES = ("libero_spatial", "libero_goal", "libero_object", "libero_10")
TASK_RE = re.compile(r"^Task: (.+)$")
SUCCESS_RE = re.compile(r"^Success: (True|False)$")


def parse_suite(path: Path, expected_rollouts: int) -> dict:
    tasks: OrderedDict[str, list[bool]] = OrderedDict()
    current_task = None
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        task_match = TASK_RE.match(line)
        if task_match:
            current_task = task_match.group(1)
            tasks.setdefault(current_task, [])
            continue
        success_match = SUCCESS_RE.match(line)
        if success_match:
            if current_task is None:
                raise RuntimeError(f"Success line precedes any task in {path}")
            tasks[current_task].append(success_match.group(1) == "True")

    if len(tasks) != 10:
        raise RuntimeError(f"Expected 10 tasks in {path}, found {len(tasks)}")
    bad_counts = {
        task: len(outcomes)
        for task, outcomes in tasks.items()
        if len(outcomes) != expected_rollouts
    }
    if bad_counts:
        raise RuntimeError(
            f"Incomplete protocol in {path}; expected {expected_rollouts} rollouts/task, "
            f"got {bad_counts}"
        )

    task_rows = []
    for task, outcomes in tasks.items():
        successes = sum(outcomes)
        task_rows.append(
            {
                "task": task,
                "episodes": len(outcomes),
                "successes": successes,
                "success_rate": successes / len(outcomes),
                "success_percent": 100.0 * successes / len(outcomes),
            }
        )
    episodes = sum(row["episodes"] for row in task_rows)
    successes = sum(row["successes"] for row in task_rows)
    return {
        "tasks": task_rows,
        "episodes": episodes,
        "successes": successes,
        "success_rate": successes / episodes,
        "success_percent": 100.0 * successes / episodes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--expected-rollouts-per-task", type=int, default=50)
    args = parser.parse_args()

    if args.expected_rollouts_per_task <= 0:
        raise ValueError("expected rollouts must be positive")
    suites = {}
    for suite in SUITES:
        path = args.log_dir / f"{suite}.log"
        if not path.is_file():
            raise FileNotFoundError(path)
        suites[suite] = parse_suite(path, args.expected_rollouts_per_task)

    total_episodes = sum(row["episodes"] for row in suites.values())
    total_successes = sum(row["successes"] for row in suites.values())
    macro_rate = sum(row["success_rate"] for row in suites.values()) / len(suites)
    expected_total = len(SUITES) * 10 * args.expected_rollouts_per_task
    if total_episodes != expected_total:
        raise RuntimeError(
            f"Expected {expected_total} total episodes, found {total_episodes}"
        )

    result = {
        "protocol": {
            "suites": list(SUITES),
            "tasks_per_suite": 10,
            "rollouts_per_task": args.expected_rollouts_per_task,
            "expected_total_rollouts": expected_total,
        },
        "suites": suites,
        "macro_average_success_rate": macro_rate,
        "macro_average_success_percent": 100.0 * macro_rate,
        "total_episodes": total_episodes,
        "total_successes": total_successes,
        "micro_success_rate": total_successes / total_episodes,
        "micro_success_percent": 100.0 * total_successes / total_episodes,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")

    with args.output_csv.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("suite", "episodes", "successes", "success_percent"),
        )
        writer.writeheader()
        for suite in SUITES:
            row = suites[suite]
            writer.writerow(
                {
                    "suite": suite,
                    "episodes": row["episodes"],
                    "successes": row["successes"],
                    "success_percent": f'{row["success_percent"]:.4f}',
                }
            )
        writer.writerow(
            {
                "suite": "macro_average",
                "episodes": total_episodes,
                "successes": total_successes,
                "success_percent": f"{100.0 * macro_rate:.4f}",
            }
        )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
