#!/usr/bin/env python3
"""
Does the write_code example in C's system message change how the model names its function?

Compares the normal C run (example: def add(a, b)) with the run that uses a placeholder example
(no real function, --example placeholder, saved in results/ablation_example/). For each model:
solved tasks, first code correct, how often the saved code does not define the function from the
test (wrong function name), format errors, and the tasks that changed.

    python src/check_example_effect.py
"""
import csv
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline_oneshot as base  # noqa: E402

DATA = "mbpp_final_30"
names = {}
for line in open(f"Dataset/{DATA}.jsonl", encoding="utf-8"):
    if line.strip():
        row = json.loads(line)
        names[f"MBPP/{row['task_id']}"] = base.called_function(row["test_list"][0])


def load(stem):
    rows = {r["task_id"]: r for r in csv.DictReader(open(stem + ".csv", encoding="utf-8"))}
    logs = {json.loads(l)["task_id"]: json.loads(l) for l in open(stem + "_code.jsonl", encoding="utf-8")}
    return rows, logs


def wrong_name(log, fn):
    """Number of saved codes that do not define the function the test calls."""
    codes = [s["code"] for s in log["steps"] if s["action"] == "write_code" and s["code"]]
    return sum(not base.defines_function(c, fn) for c in codes), len(codes)


print(f"{'Model':<10}{'Example':<13}{'Solved':>8}{'First code ok':>15}{'Tasks w. wrong name':>21}"
      f"{'Codes w. wrong name':>21}{'Format errors':>15}")
for normal in sorted(glob.glob(f"results/agent_s24_*_mbpp_{DATA}.csv")):
    label = Path(normal).stem[len("agent_s24_"):]
    other = f"results/ablation_example/agent_s24_ex-placeholder_{label}.csv"
    if not Path(other).exists():
        continue
    model = "Qwen" if "Qwen" in label else "DeepSeek"
    runs = {"def add": load(normal[:-4]), "placeholder": load(other[:-4])}
    for ex, (rows, logs) in runs.items():
        bad_tasks = bad_codes = all_codes = 0
        for t, g in logs.items():
            b, n = wrong_name(g, names[t])
            bad_tasks += b > 0
            bad_codes += b
            all_codes += n
        print(f"{model:<10}{ex:<13}{sum(int(r['solved']) for r in rows.values()):>5}/30"
              f"{sum(r['first_code_status'] == 'passed' for r in rows.values()):>12}/30{bad_tasks:>18}/30"
              f"{bad_codes:>15}/{all_codes:<5}{sum(int(r['format_errors']) for r in rows.values()):>12}")
    (r1, _), (r2, _) = runs["def add"], runs["placeholder"]
    changed = [f"{t} ({'S' if r1[t]['solved'] == '1' else 'x'} -> {'S' if r2[t]['solved'] == '1' else 'x'})"
               for t in r1 if r1[t]["solved"] != r2[t]["solved"]]
    print(f"{'':<10}tasks with a different result: {', '.join(changed) if changed else 'none'}\n")
