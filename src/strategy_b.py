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
     Stop when the dev test passes, or after --max-iterations LLM calls (default 5 = 1 first try + 4 repairs),
     or when the model
     writes the same code again: with greedy decoding the next prompt would then be identical,
     so the next answer would be identical too. Stopping there cannot change the result, it only saves cost.
  5. Final evaluation: the last code is run against ALL 3 tests (same check as Strategy A).
     Tests 2 and 3 are never shown to the model. The final result is reported as
     final_success_rate (= solve rate), not pass@1, because B may call the LLM several times.

Code extraction and cleaning are the same as in Strategy A (baseline_oneshot.extract_code / remove_model_tests).

Measurements per task (same columns as A, plus B-only columns):
  llm_calls = 1 initial + repair_calls;  tokens and generation time are split into initial and repair parts.
  eval_exec_time_s     = final evaluation run on all tests (identical procedure in A and B)
  feedback_exec_time_s = the extra dev-test runs that B needs to create feedback
  The diagnostic run of attempt 1 on all tests (first_status, to compare with A) is NOT counted in any time.
_code.jsonl keeps, for every attempt: full rendered prompt, raw answer, extracted + cleaned code,
compile result, dev-test result, feedback, tokens, times, token-limit flag.

Output (results/):  feedback_it<k>_<model>_<dataset>_<data file>.csv / .json / _code.jsonl
Extra CSV columns: first_status (attempt 1 on all tests = Strategy A result), dev_passed,
stop_reason (dev_test_passed / repeated_code / max_iterations_reached).

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


def make_dev_test(test):
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


def run_compiler(code):
    """COMPILER: Python's own compiler, the same step py_compile does. Nothing is executed.
    Returns None if the code compiles, else the error message with the line number."""
    try:
        compile(code, "solution.py", "exec")
        return None
    except SyntaxError as e:
        return f"{type(e).__name__}: {e.msg} (line {e.lineno})"


def run_tester(task, code, timeout):
    """TESTER: first the compiler, then the dev test (test 1) in the isolated process.
    Returns (ok, feedback, exec_time, stage) with stage = no_code / compile_error / dev_<status>."""
    if not code.strip():
        return False, "Your reply contained no Python code.", 0.0, "no_code"
    err = run_compiler(code)
    if err:
        return False, err, 0.0, "compile_error"
    program = code + "\n\n" + task["setup"] + "\n" + make_dev_test(task["dev_test"])
    status, err, dt = base.run_code(program, timeout)
    if status == "passed":
        return True, "", dt, "dev_passed"
    if status == "timeout":
        err = f"Timeout: the code did not finish in {timeout} seconds ({err})."
    return False, err, dt, "dev_" + status


def make_fix_prompt(task, code, feedback):
    return (
        base.make_prompt(task)
        + "\n\nYour previous solution:\n```python\n" + code + "\n```\n"
        + "We ran it and got this error:\n" + feedback + "\n\n"
        + "Fix the code. Reply with the complete corrected Python code in a single ```python code block."
    )


# --------------------------------------------------------------------------- #
# One task
# --------------------------------------------------------------------------- #
def run_task(task, backend, model_label, timeout, max_iterations):
    gen_t = [0.0, 0.0]          # [initial, repair] generation time
    tok_in, tok_out = [0, 0], [0, 0]
    t_feedback = 0.0
    attempts = []
    prompt = base.make_prompt(task)
    code = feedback = first = ""
    dev_ok, stop = False, "max_iterations_reached"
    limit = getattr(backend, "max_new_tokens", 10**9)
    for gen_no in range(max_iterations):  # one iteration = one LLM call (0 = first try, 1.. = repairs)
        if gen_no > 0:
            prompt = make_fix_prompt(task, code, feedback)
        k = 0 if gen_no == 0 else 1
        t0 = time.perf_counter()
        text, i, o = backend.generate(base.SYSTEM_PROMPT, prompt, dict(task, repair=gen_no))
        g = time.perf_counter() - t0
        gen_t[k] += g
        tok_in[k] += i
        tok_out[k] += o
        extracted = base.extract_code(text)
        code = base.remove_model_tests(extracted)
        dev_ok, feedback, dt, stage = run_tester(task, code, timeout)
        t_feedback += dt
        attempts.append({"attempt": gen_no, "kind": "initial" if gen_no == 0 else "repair",
                         "prompt_rendered": getattr(backend, "last_prompt", "") or prompt,
                         "raw_output": text, "extracted_code": extracted, "code": code,
                         "check_stage": stage, "dev_passed": dev_ok, "feedback": feedback,
                         "input_tokens": i, "output_tokens": o, "hit_token_limit": o >= limit,
                         "gen_time_s": round(g, 3), "feedback_exec_time_s": round(dt, 3)})
        if gen_no == 0:  # attempt 1 on all tests = what Strategy A would score (diagnostic, not timed)
            first = base.run_code(base.add_tests_to_code(task, code), timeout)[0] if code.strip() else "no_code"
        if dev_ok:
            stop = "dev_test_passed"
            break
        if gen_no > 0 and (code, feedback) == (attempts[-2]["code"], attempts[-2]["feedback"]):
            stop = "repeated_code"  # same code + same error -> next prompt and answer would be identical
            break

    # final evaluation on all 3 tests (same procedure as Strategy A)
    if code.strip():
        status, err, t_eval = base.run_code(base.add_tests_to_code(task, code), timeout)
    else:
        status, err, t_eval = "no_code", "empty model output", 0.0
    t_gen = sum(gen_t)
    t_exec = t_eval + t_feedback
    row = {
        "task_id": task["task_id"], "difficulty": task.get("difficulty", ""),
        "model": model_label, "strategy": STRATEGY,
        "solved": int(status == "passed"), "status": status, "error": err,
        "llm_calls": len(attempts), "repair_calls": len(attempts) - 1,
        "input_tokens": sum(tok_in), "output_tokens": sum(tok_out),
        "initial_input_tokens": tok_in[0], "initial_output_tokens": tok_out[0],
        "repair_input_tokens": tok_in[1], "repair_output_tokens": tok_out[1],
        "gen_time_s": round(t_gen, 3), "initial_gen_time_s": round(gen_t[0], 3),
        "repair_gen_time_s": round(gen_t[1], 3),
        "eval_exec_time_s": round(t_eval, 3), "feedback_exec_time_s": round(t_feedback, 3),
        "exec_time_s": round(t_exec, 3), "total_time_s": round(t_gen + t_exec, 3),
        "first_status": first, "dev_passed": int(dev_ok),
        "stop_reason": stop,
    }
    return row, {"task_id": task["task_id"], "status": status, "error": err, "stop_reason": stop,
                 "attempts": attempts}


def summarize(rows, settings=None):
    s = base.summarize(rows, settings)
    s["strategy"] = STRATEGY
    s.pop("pass@1", None)  # iterative strategy: report the final success rate instead
    s["final_success_rate"] = s["solve_rate"]
    n = len(rows)
    first_pass = sum(r["first_status"] == "passed" for r in rows)
    s["first_attempt_pass@1"] = round(first_pass / n, 4) if n else None  # should equal Strategy A
    s["fixed_by_repair"] = sum(r["solved"] for r in rows) - first_pass
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
                        (0, 4, "test_failure", "max_iterations_reached")),
        "dev_only": ([only_dev], (0, 1, "test_failure", "dev_test_passed")),
    }
    tasks = []
    for k in cases:
        t = dict(base.normalize(row, "mbpp"), setup="", dev_test=row["test_list"][0])
        t["task_id"] = k
        tasks.append(t)
    mock = ScriptedMock({k: v[0] for k, v in cases.items()})
    rows = [run_task(t, mock, "mock", timeout=3, max_iterations=4)[0] for t in tasks]
    ok = True
    for r in rows:
        want = cases[r["task_id"]][1]
        got = (r["solved"], r["llm_calls"], r["first_status"], r["stop_reason"])
        ok &= got == want
        print(f"  {'OK ' if got == want else 'FAIL'} {r['task_id']:<13} solved={r['solved']} "
              f"calls={r['llm_calls']} first={r['first_status']:<13} stop={r['stop_reason']}")
    repairs = [p for p in mock.prompts if "Your previous solution" in p]
    no_leak = all("add(10, 5)" not in p and "add(-1, 1)" not in p for p in mock.prompts)
    detail = any("expected 3, got -1" in p for p in repairs)
    syntax = any("SyntaxError" in p and "line 1" in p for p in repairs)
    for name, good in [("hidden tests 2-3 never shown to the model", no_leak),
                       ("feedback says expected vs got", detail),
                       ("compiler feedback has error + line", syntax)]:
        ok &= good
        print(f"  {'OK ' if good else 'FAIL'} {name}")
    # model repeats old code first, corrected code second; and adds its own (wrong) assert
    t = dict(tasks[0], task_id="echo_then_fix")
    m2 = ScriptedMock({"echo_then_fix": [bad_logic + "\nassert add(1, 2) == 99",
                                         bad_logic + "\n```\nFixed:\n```python\n" + ok_code]})
    r2, g2 = run_task(t, m2, "mock", timeout=3, max_iterations=4)
    good = (r2["solved"], r2["llm_calls"]) == (1, 2) and "expected 3, got -1" in g2["attempts"][0]["feedback"]
    ok &= good
    print(f"  {'OK ' if good else 'FAIL'} corrected (last) block used; model's own assert ignored -> real feedback")
    times = all(abs(r["exec_time_s"] - r["eval_exec_time_s"] - r["feedback_exec_time_s"]) < 0.002 and
                r["repair_calls"] == r["llm_calls"] - 1 for r in rows)
    ok &= times
    print(f"  {'OK ' if times else 'FAIL'} exec = eval + feedback, repair_calls = llm_calls - 1")
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
    ap.add_argument("--max-iterations", type=int, default=5,
                    help="max LLM calls per task: 1 first try + repairs (default 5)")
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
    stem = f"feedback_it{a.max_iterations}_{re.sub(r'[^\w.-]', '_', label)}_{a.dataset}_{Path(a.data_file).stem}"
    csv_path, code_path = out / f"{stem}.csv", out / f"{stem}_code.jsonl"
    settings = {"model": a.model, "max_new_tokens": a.max_new_tokens, "decoding": "greedy", "timeout_s": a.timeout,
                "mem_limit_mb": a.mem_mb, "quant": a.quant or "none", "remote_code": not a.no_remote_code,
                "max_iterations": a.max_iterations,
                "feedback": "compiler + dev test (test 1, shown in prompt)",
                "final_tests": "all 3 tests", "data": a.data_file,
                "code_extraction": "last python block with a function; model's own top-level test code removed",
                "environment": base.env_info(backend)}

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
            row, gen = run_task(task, backend, label, a.timeout, a.max_iterations)
            rows.append(row)
            w.writerow(row)
            fh.flush()
            cf.write(json.dumps(gen, ensure_ascii=False) + "\n")
            cf.flush()
            hit = any(x["hit_token_limit"] for x in gen["attempts"])
            print(f"[{i}/{len(tasks)}] {row['task_id']} ({row['difficulty']}): {row['first_status']} -> "
                  f"{row['status']}  calls={row['llm_calls']} ({row['stop_reason']})  {row['total_time_s']}s  "
                  f"{row['input_tokens']}+{row['output_tokens']} tok{'  HIT TOKEN LIMIT' if hit else ''}")
    summary = summarize(rows, settings)
    (out / f"{stem}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
