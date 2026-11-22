#!/usr/bin/env python3
"""
Runs the final evaluation of all saved solutions again inside the sandbox and compares
with the CSV results. If everything matches, the old runs are still valid.

    python src/reevaluate_sandbox.py
"""
import csv
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline_oneshot as base  # noqa: E402

print("Sandbox:", base.sandbox_info(), "\n")

tasks = {}
for f in glob.glob("Dataset/mbpp_*.jsonl"):
    for line in open(f, encoding="utf-8"):
        if line.strip():
            t = base.normalize(json.loads(line), "mbpp")
            tasks[t["task_id"]] = t

total = same = 0
for code_file in sorted(glob.glob("results/oneshot_*_code.jsonl") + glob.glob("results/feedback_*_code.jsonl")):
    csv_file = code_file.replace("_code.jsonl", ".csv")
    stored = {r["task_id"]: r["status"] for r in csv.DictReader(open(csv_file, encoding="utf-8"))}
    diffs = []
    for line in open(code_file, encoding="utf-8"):
        g = json.loads(line)
        code = g["attempts"][-1]["code"] if "attempts" in g else g["code"]  # B: last attempt, A: the answer
        status = base.run_code(base.add_tests_to_code(tasks[g["task_id"]], code), 10)[0] if code.strip() else "no_code"
        total += 1
        if status == stored[g["task_id"]]:
            same += 1
        else:
            diffs.append(f"{g['task_id']}: stored={stored[g['task_id']]} sandbox={status}")
    name = Path(csv_file).name
    print(f"{name[:75]:<77} {'all the same' if not diffs else 'DIFFERENT: ' + '; '.join(diffs)}")

print(f"\n{same}/{total} solutions give the same result inside the sandbox.")
print("Sandbox:", base.sandbox_info())
