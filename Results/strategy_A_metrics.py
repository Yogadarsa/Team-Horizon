
import json
import sys
from collections import defaultdict
from pathlib import Path

def main():
    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: python strategy_A_metrics.py EVALUATION_RESULTS.jsonl"
        )

    input_path = Path(sys.argv[1])

    if not input_path.is_file():
        raise SystemExit(f"Results file not found: {input_path}")

    records = [
        json.loads(line)
        for line in input_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    grouped = defaultdict(list)

    for record in records:
        grouped[record["model"]].append(record)

    for model, tasks in sorted(grouped.items()):
        solved = sum(
            r["tests_total"] > 0
            and r["tests_passed"] == r["tests_total"]
            and r["evaluation_status"] == "completed"
            for r in tasks
        )

        total = len(tasks)
        pass_at_1 = solved / total if total else 0

        print(f"\nMODEL: {model}")
        print("Tasks evaluated:", total)
        print("Fully solved:", solved)
        print(f"Pass@1: {pass_at_1:.2%}")

if __name__ == "__main__":
    main()
