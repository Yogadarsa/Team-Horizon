#!/usr/bin/env python3
"""
Checks the code extraction on all saved answers (no model needed).

For every answer it applies the extraction rule again and checks that the run used the
same code. For C it also checks every agent reply: the action and the saved code must be
exactly what the rules (first_action, parse_action) give. Answers with two different versions of the function (ambiguous) are listed,
and every version is run on test 1 and on all tests, so we can see if picking another
block would have changed anything.

    python src/verify_extraction.py
    python src/verify_extraction.py --quiet     # only the counts
"""
import argparse
import glob
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline_oneshot as base  # noqa: E402
import strategy_b as sb  # noqa: E402
import strategy_c as sc  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--results", default="results")
ap.add_argument("--quiet", action="store_true")
a = ap.parse_args()

tasks = {}
for f in glob.glob("Dataset/mbpp_*.jsonl"):
    for line in open(f, encoding="utf-8"):
        if line.strip():
            row = json.loads(line)
            t = base.normalize(row, "mbpp")
            t["setup"] = "\n".join(row.get("test_imports") or [])
            t["dev_test"] = row["test_list"][0]
            tasks[t["task_id"]] = t

files = sorted(glob.glob(f"{a.results}/oneshot_*_code.jsonl") + glob.glob(f"{a.results}/feedback_*_code.jsonl"))
total, mismatches, changed_result = Counter(), [], []
print("Sandbox:", base.sandbox_info(), "\n")
for f in files:
    run = Path(f).name.replace("_code.jsonl", "")
    cases = Counter()
    for line in open(f, encoding="utf-8"):
        g = json.loads(line)
        task = tasks[g["task_id"]]
        attempts = g.get("attempts", [g])
        for k, att in enumerate(attempts):
            code, info = base.extract_code_info(att["raw_output"], task["func_name"])
            cases[info["case"]] += 1
            if code != att["extracted_code"]:
                mismatches.append(f"{run} {g['task_id']} attempt {k}")
            if not info["ambiguous"]:
                continue
            # every different candidate: dev test (feedback) and all tests (final result)
            blocks = [b for t, b in base.FENCE.findall(att["raw_output"]) if t.lower() in base.PY_TAGS]
            cands = list(dict.fromkeys(b.strip() + "\n" for b in blocks if base.defines_function(b, task["func_name"])))
            is_last = k == len(attempts) - 1
            results = []
            for c in cands:
                clean = base.remove_model_tests(c)
                dev = sb.run_tester(task, clean, 10)[0]
                final = base.run_code(base.add_tests_to_code(task, clean), 10)[0]
                results.append((dev, final))
            same_dev = len({r[0] for r in results}) == 1
            same_final = len({r[1] for r in results}) == 1
            if is_last and not same_final:
                changed_result.append(f"{run} {g['task_id']}")
            print(f"AMBIGUOUS  {run[:60]}  {g['task_id']}  attempt {k} ({'final code' if is_last else 'repaired later'}): "
                  f"{len(cands)} different versions")
            for i, (c, (dev, final)) in enumerate(zip(cands, results), 1):
                tag = "  <- taken (last)" if i == len(cands) else ""
                print(f"    version {i}: visible test {'passed' if dev else 'failed'}, all tests: {final}{tag}")
            print(f"    choice changes the feedback: {'no' if same_dev else 'YES'};"
                  f" changes the final result: {'no' if same_final or not is_last else 'YES'}")
            if not a.quiet:
                txt = att["raw_output"]
                for marker in ("corrected", "Corrected", "fixed", "Fixed", "Here is", "Here's"):
                    p = txt.find(marker)
                    if p >= 0:
                        print(f"    model's text before the last version: ...{txt[max(0, p - 60):p + 60]!r}...")
                        break
            print()
    total.update(cases)
    print(f"{run[:75]:<77} " + ", ".join(f"{k}={v}" for k, v in sorted(cases.items())))

n = sum(total.values())
print(f"\n{n} answers in {len(files)} runs: " + ", ".join(f"{k}={v}" for k, v in sorted(total.items())))
print(f"Code used in the runs = code picked by the rule: {n - len(mismatches)}/{n}"
      + ("" if not mismatches else "   DIFFERENT: " + "; ".join(mismatches[:10])))
print(f"Ambiguous answers: {total['several_last']}; final results that would change with another choice: "
      f"{len(changed_result)}" + ("" if not changed_result else " -> " + "; ".join(changed_result)))

# Strategy C: the agent's replies go through first_action + parse_action. Check that the action and the
# saved code in the log are exactly what the rules give for the raw reply.
c_files = sorted(glob.glob(f"{a.results}/agent_*_code.jsonl"))
if c_files:
    print()
    c_total, c_cases, c_bad = 0, Counter(), []
    for f in c_files:
        run = Path(f).name.replace("_code.jsonl", "")
        for line in open(f, encoding="utf-8"):
            g = json.loads(line)
            fn = tasks[g["task_id"]]["func_name"]
            for st in g["steps"]:
                reply, _ = sc.first_action(st["raw_output"])
                action, code, fmt, ext = sc.parse_action(reply, fn)
                c_total += 1
                if action != st["action"]:
                    c_bad.append(f"{run} {g['task_id']} step {st['step']} (action)")
                if action == "write_code":
                    c_cases[ext["case"]] += 1
                    if base.remove_model_tests(code) != st["code"]:
                        c_bad.append(f"{run} {g['task_id']} step {st['step']} (code)")
        print(f"{run[:75]:<77} checked")
    print(f"\nC: {c_total} agent replies, saved code cases: " + ", ".join(f"{k}={v}" for k, v in sorted(c_cases.items())))
    print(f"C: action and saved code match the rules: {c_total - len(c_bad)}/{c_total}"
          + ("" if not c_bad else "   DIFFERENT: " + "; ".join(c_bad[:10])))
