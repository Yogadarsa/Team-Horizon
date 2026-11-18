
import json
import subprocess
import sys
from pathlib import Path

TIMEOUT_SECONDS = 10

def evaluate_one(record, worker_path):
    try:
        completed = subprocess.run(
            [sys.executable, "-I", "-S", str(worker_path)],
            input=json.dumps(record),
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS
        )

        if completed.returncode != 0:
            return {
                "model": record["model"],
                "task_id": record["task_id"],
                "tests_total": len(record["test_list"]),
                "tests_passed": 0,
                "evaluation_status": "worker_error",
                "error": completed.stderr[:1000]
            }

        return json.loads(completed.stdout)

    except subprocess.TimeoutExpired:
        return {
            "model": record["model"],
            "task_id": record["task_id"],
            "tests_total": len(record["test_list"]),
            "tests_passed": 0,
            "evaluation_status": "timeout",
            "error": f"Exceeded {TIMEOUT_SECONDS} seconds"
        }

def main():
    if len(sys.argv) != 3:
        raise SystemExit(
            "Usage: python controller.py INPUT.jsonl OUTPUT.jsonl"
        )

    manifest_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    worker_path = Path(__file__).with_name(
        "mbpp_evaluation_worker.py"
    )

    if not worker_path.exists():
        raise SystemExit("Evaluation worker not found")

    if output_path.exists():
        raise SystemExit(
            f"Refusing to overwrite: {output_path}"
        )

    with manifest_path.open(encoding="utf-8") as f:
        records = [
            json.loads(line)
            for line in f
            if line.strip()
        ]

    with output_path.open("x", encoding="utf-8") as f:
        for record in records:
            result = evaluate_one(record, worker_path)

            f.write(json.dumps(result) + "\n")
            f.flush()

            print(
                result["model"],
                result["task_id"],
                f'{result["tests_passed"]}/{result["tests_total"]}',
                result["evaluation_status"],
                flush=True
            )

    print("Evaluation completed:", output_path)

if __name__ == "__main__":
    main()
