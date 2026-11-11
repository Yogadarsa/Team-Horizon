#!/usr/bin/env python3
"""
Strategy B - LLM + real compiler/test feedback (same LLM, no extra model).

Professor's architecture:
  Task -> LLM -> Solution -> Compile / Test -> Feedback -> same LLM -> corrected solution
  A plain Python loop controls the repairs. No second LLM, no agent.

Per task:
  1. LLM writes code with exactly the same prompt as Strategy A (so attempt 1 == Strategy A).
  2. Compile check: Python's own compiler (same as py_compile). Syntax error -> feedback.
  3. Dev test: run the code against the ONE test that is already in the prompt (test_list[0]),
     in the same isolated process as Strategy A (time + memory limits).
     If it fails, the feedback says what was expected and what the code returned.
  4. If 2 or 3 failed: send task + previous code + feedback to the same LLM -> new code -> back to 2.
     Stop when the dev test passes, or after --max-repairs repairs (default 3), or when the model
     writes the same code again: with greedy decoding the next prompt would then be identical,
     so the next answer would be identical too. Stopping there cannot change the result, it only saves cost.
  5. Final evaluation: the last code is run against ALL 3 tests (same check as Strategy A),
     so Pass@1 is directly comparable with A. Tests 2 and 3 are never shown to the model.

Output (results/):  feedback_n<k>_<model>_<dataset>_<data file>.csv / .json / _code.jsonl
Extra CSV columns: first_status (attempt 1 on all tests = Strategy A result), dev_passed,
stop_reason (dev_test_passed / repeated_code / max_repairs_reached).

Needs src/baseline_oneshot.py next to this file.

    python src/strategy_b.py --selftest
    python -u src/strategy_b.py --backend hf --model Qwen/Qwen2.5-Coder-7B-Instruct \
        --dataset mbpp --data-file Dataset/mbpp_initial_10.jsonl --max-new-tokens 512
    python -u src/strategy_b.py --backend hf --model deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct \
        --quant 8bit --no-remote-code --dataset mbpp --data-file Dataset/mbpp_initial_10.jsonl --max-new-tokens 512
"""
import argparse
import ast
import csv
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline_oneshot as base  # noqa: E402

STRATEGY = "B_feedback"
CSV_FIELDS = base.CSV_FIELDS + ["first_status", "dev_passed", "stop_reason"]


# --------------------------------------------------------------------------- #
# Tasks: same as Strategy A, plus the dev test (the one test shown in the prompt)
# --------------------------------------------------------------------------- #
def load_tasks(dataset, data_file):
    if dataset != "mbpp" or not data_file:
        sys.exit("Strategy B needs --dataset mbpp and --data-file (e.g. Dataset/mbpp_initial_10.jsonl)")
    tasks = []
    for line in open(data_file, encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        t = base.normalize(row, "mbpp")
        t["setup"] = row.get("test_setup_code") or "\n".join(row.get("test_imports") or [])
        t["dev_test"] = row["test_list"][0]
        tasks.append(t)
    return tasks


def dev_check_code(test):
    """Turn `assert f(x) == y` into a check that also reports what the code returned."""
    try:
        node = ast.parse(test).body[0]
        cmp = node.test
        if (isinstance(node, ast.Assert) and isinstance(cmp, ast.Compare)
                and len(cmp.ops) == 1 and isinstance(cmp.ops[0], ast.Eq)):
            got, exp = ast.unparse(cmp.left), ast.unparse(cmp.comparators[0])
            return (f"__got = {got}\n__exp = {exp}\n"
                    f"if not (__got == __exp):\n"
                    f"    raise AssertionError('expected ' + repr(__exp)[:200] + ', got ' + repr(__got)[:200])\n")
    except (SyntaxError, IndexError, AttributeError):
        pass
    return test + "\n"  # other test shapes (e.g. math.isclose): plain assert


def compile_check(code):
    """Python's own compiler, the same step py_compile does. Nothing is executed."""
    try:
        compile(code, "solution.py", "exec")
        return None
    except SyntaxError as e:
        return f"{type(e).__name__}: {e.msg} (line {e.lineno})"


def check(task, code, timeout):
    """Compile, then run the dev test in the isolated process. Returns (ok, feedback, exec_time)."""
    if not code.strip():
        return False, "Your reply contained no Python code.", 0.0
    err = compile_check(code)
    if err:
        return False, err, 0.0
    program = code + "\n\n" + task["setup"] + "\n" + dev_check_code(task["dev_test"])
    status, err, dt = base.run_program(program, timeout)
    if status == "passed":
        return True, "", dt
    if status == "timeout":
        err = f"Timeout: the code did not finish in {timeout} seconds ({err})."
    return False, err, dt


def build_repair_prompt(task, code, feedback):
    return (
        base.build_user_prompt(task)
        + "\n\nYour previous solution:\n```python\n" + code + "\n```\n"
        + "We ran it and got this error:\n" + feedback + "\n\n"
        + "Fix the code. Reply with the complete corrected Python code in a single ```python code block."
    )


# --------------------------------------------------------------------------- #
# One task
# --------------------------------------------------------------------------- #
def run_task(task, backend, model_label, timeout, max_repairs):
    t_gen = t_exec = 0.0
    n_in = n_out = 0
    attempts = []
    prompt = base.build_user_prompt(task)
    code = feedback = first = ""
    dev_ok, stop = False, "max_repairs_reached"
    for gen_no in range(max_repairs + 1):
        if gen_no > 0:
            prompt = build_repair_prompt(task, code, feedback)
        t0 = time.perf_counter()
        text, i, o = backend.generate(base.SYSTEM_PROMPT, prompt, dict(task, repair=gen_no))
        g = time.perf_counter() - t0
        t_gen += g
        n_in, n_out = n_in + i, n_out + o
        code = base.extract_code(text)
        dev_ok, feedback, dt = check(task, code, timeout)
        t_exec += dt
        attempts.append({"code": code, "dev_passed": dev_ok, "feedback": feedback, "raw_output": text,
                         "input_tokens": i, "output_tokens": o, "gen_time_s": round(g, 3)})
        if gen_no == 0:  # attempt 1 on all tests = what Strategy A would score (not counted in time)
            first = base.run_program(base.build_program(task, code), timeout)[0] if code.strip() else "no_code"
        if dev_ok:
            stop = "dev_test_passed"
            break
        if gen_no > 0 and (code, feedback) == (attempts[-2]["code"], attempts[-2]["feedback"]):
            stop = "repeated_code"  # same code + same error -> next prompt and answer would be identical
            break

    # final evaluation on all 3 tests (same as Strategy A)
    if code.strip():
        status, err, dt = base.run_program(base.build_program(task, code), timeout)
        t_exec += dt
    else:
        status, err = "no_code", "empty model output"
    row = {
        "task_id": task["task_id"], "difficulty": task.get("difficulty", ""),
        "model": model_label, "strategy": STRATEGY,
        "solved": int(status == "passed"), "status": status, "error": err,
        "iterations": len(attempts),  # number of generations (1 = no repair needed)
        "input_tokens": n_in, "output_tokens": n_out,
        "gen_time_s": round(t_gen, 3), "exec_time_s": round(t_exec, 3),
        "total_time_s": round(t_gen + t_exec, 3),
        "first_status": first, "dev_passed": int(dev_ok),
        "stop_reason": stop,
    }
    return row, {"task_id": task["task_id"], "status": status, "attempts": attempts}


def summarize(rows, settings=None):
    s = base.summarize(rows, settings)
    s["strategy"] = STRATEGY
    n = len(rows)
    first_pass = sum(r["first_status"] == "passed" for r in rows)
    s["first_attempt_pass@1"] = round(first_pass / n, 4) if n else None  # should equal Strategy A
    s["fixed_by_repair"] = sum(r["solved"] for r in rows) - first_pass
    s["avg_iterations"] = round(sum(r["iterations"] for r in rows) / n, 3) if n else None
    s["stop_reason_counts"] = dict(Counter(r["stop_reason"] for r in rows))
    s["dev_passed_but_hidden_failed"] = sum(1 for r in rows if int(r["dev_passed"]) and not r["solved"])
    s["first_status_counts"] = dict(Counter(r["first_status"] for r in rows))
    return s


# --------------------------------------------------------------------------- #
# Self-test (no model)
# --------------------------------------------------------------------------- #
class ScriptedMock:
    def __init__(self, answers):
        self.answers, self.calls, self.prompts = answers, Counter(), []

    def generate(self, system, user, task=None):
        tid = task["task_id"]
        seq = self.answers[tid]
        out = seq[min(self.calls[tid], len(seq) - 1)]
        self.calls[tid] += 1
        self.prompts.append(user)
        return "```python\n" + out + "\n```", len(user.split()), len(out.split())


def selftest():
    row = {"task_id": 0, "prompt": "Write a function to add two numbers.", "test_imports": [],
           "test_list": ["assert add(1, 2) == 3", "assert add(10, 5) == 15", "assert add(-1, 1) == 0"]}
    ok_code = "def add(a, b):\n    return a + b"
    bad_syntax = "def add(a, b)\n    return a + b"
    bad_logic = "def add(a, b):\n    return a - b"
    only_dev = "def add(a, b):\n    return 3"  # passes test 1, fails tests 2-3
    crash = "def add(a, b):\n    return a + c"
    cases = {  # answers, expected (solved, iterations, first_status, stop_reason)
        "right_first": ([ok_code], (1, 1, "passed", "dev_test_passed")),
        "fixed_syntax": ([bad_syntax, ok_code], (1, 2, "syntax_error", "dev_test_passed")),
        "fixed_logic": ([bad_logic, ok_code], (1, 2, "test_failure", "dev_test_passed")),
        "fixed_crash": ([crash, ok_code], (1, 2, "runtime_error", "dev_test_passed")),
        "repeats_code": ([bad_logic], (0, 2, "test_failure", "repeated_code")),
        "never_fixed": ([bad_logic, crash, bad_syntax, "def add(a, b):\n    return a * b"],
                        (0, 4, "test_failure", "max_repairs_reached")),
        "dev_only": ([only_dev], (0, 1, "test_failure", "dev_test_passed")),
    }
    tasks = []
    for k in cases:
        t = dict(base.normalize(row, "mbpp"), setup="", dev_test=row["test_list"][0])
        t["task_id"] = k
        tasks.append(t)
    mock = ScriptedMock({k: v[0] for k, v in cases.items()})
    rows = [run_task(t, mock, "mock", timeout=3, max_repairs=3)[0] for t in tasks]
    ok = True
    for r in rows:
        want = cases[r["task_id"]][1]
        got = (r["solved"], r["iterations"], r["first_status"], r["stop_reason"])
        ok &= got == want
        print(f"  {'OK ' if got == want else 'FAIL'} {r['task_id']:<13} solved={r['solved']} "
              f"iter={r['iterations']} first={r['first_status']:<13} stop={r['stop_reason']}")
    repairs = [p for p in mock.prompts if "Your previous solution" in p]
    no_leak = all("add(10, 5)" not in p and "add(-1, 1)" not in p for p in mock.prompts)
    detail = any("expected 3, got -1" in p for p in repairs)
    syntax = any("SyntaxError" in p and "line 1" in p for p in repairs)
    for name, good in [("hidden tests 2-3 never shown to the model", no_leak),
                       ("feedback says expected vs got", detail),
                       ("compiler feedback has error + line", syntax)]:
        ok &= good
        print(f"  {'OK ' if good else 'FAIL'} {name}")
    print("summary:", json.dumps(summarize(rows)))
    print("SELFTEST PASSED" if ok else "SELFTEST FAILED")
    return 0 if ok else 1


# --------------------------------------------------------------------------- #
def read_rows(csv_path):
    rows = []
    for r in base._read_rows(csv_path):
        if r.get("first_status") and r.get("stop_reason"):
            r["dev_passed"] = int(r["dev_passed"])
            rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser(description="Strategy B: LLM + compiler/test feedback")
    ap.add_argument("--backend", choices=["hf", "ollama", "mock"], default="hf")
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    ap.add_argument("--dataset", choices=["mbpp"], default="mbpp")
    ap.add_argument("--data-file", help="team dataset, e.g. Dataset/mbpp_initial_10.jsonl")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--max-repairs", type=int, default=3, help="max number of repair rounds (default 3)")
    ap.add_argument("--timeout", type=int, default=10)
    ap.add_argument("--mem-mb", type=int, default=1024)
    ap.add_argument("--max-new-tokens", type=int, default=512, help="keep identical to Strategy A")
    ap.add_argument("--quant", choices=["8bit"])
    ap.add_argument("--no-remote-code", action="store_true")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    base.LIMITS["mem_mb"] = a.mem_mb

    if a.selftest:
        sys.exit(selftest())

    tasks = load_tasks(a.dataset, a.data_file)
    if a.limit:
        tasks = tasks[:a.limit]
    if a.backend == "hf":
        backend = base.HFBackend(a.model, a.max_new_tokens, a.quant, not a.no_remote_code)
    elif a.backend == "ollama":
        backend = base.OllamaBackend(a.model, a.max_new_tokens)
    else:
        backend = base.MockBackend()

    label = a.model + (f"@{a.quant}" if a.quant else "")
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"feedback_n{a.max_repairs}_{re.sub(r'[^\w.-]', '_', label)}_{a.dataset}_{Path(a.data_file).stem}"
    csv_path, code_path = out / f"{stem}.csv", out / f"{stem}_code.jsonl"
    settings = {"max_new_tokens": a.max_new_tokens, "decoding": "greedy", "timeout_s": a.timeout,
                "mem_limit_mb": a.mem_mb, "quant": a.quant or "none", "max_repairs": a.max_repairs,
                "feedback": "compiler + dev test (test 1, shown in prompt)",
                "final_tests": "all 3 tests", "data": a.data_file}

    rows = read_rows(csv_path) if a.resume and csv_path.exists() else []
    if rows:
        print(f"Resuming: {len(rows)} tasks already finished")
    done = {r["task_id"] for r in rows}
    with open(csv_path, "w", newline="", encoding="utf-8") as fh, \
            open(code_path, "a" if rows else "w", encoding="utf-8") as cf:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
        fh.flush()
        for i, task in enumerate(tasks, 1):
            if task["task_id"] in done:
                continue
            row, gen = run_task(task, backend, label, a.timeout, a.max_repairs)
            rows.append(row)
            w.writerow(row)
            fh.flush()
            cf.write(json.dumps(gen) + "\n")
            cf.flush()
            print(f"[{i}/{len(tasks)}] {row['task_id']} ({row['difficulty']}): {row['first_status']} -> "
                  f"{row['status']}  iter={row['iterations']}  {row['total_time_s']}s  "
                  f"{row['input_tokens']}+{row['output_tokens']} tok")
    summary = summarize(rows, settings)
    (out / f"{stem}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
