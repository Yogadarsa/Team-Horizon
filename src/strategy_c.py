#!/usr/bin/env python3
"""
Strategy C: the model works as an agent and decides itself what to do next.

There is no fixed "write, test, fix" loop like in B. In every step the model picks one action, we run it
and show the result, and the model sees the whole history so far before picking the next action. It
stops when it submits or when it runs out of steps. This is a single ReAct-style agent (like
mini-SWE-agent), with no framework and no second model.

Actions (plain text format, no JSON tool calls):
    write_code    save a full solution (in a ```python block in the same reply)
    read_code     show the saved code
    check_syntax  compile the saved code (same compiler as B)
    run_tests     run the saved code on test 1, the one shown in the task (same tester as B)
    submit        finish, the saved code is the answer

Same as A and B: model, greedy decoding, max_new_tokens per call, the task prompt, code extraction
and cleaning, sandbox, and the final evaluation on all tests. Tests 2 and 3 are never shown.

Budget: --max-steps LLM calls (default 24). B can make up to 6 code versions (1 + 5 repairs). In the
pilot the agent used 4 steps per round (write_code, check_syntax, run_tests, read_code), so 24 steps
give it room for the same 6 versions. If the steps run out, the saved code is the answer.

Stop rules: the agent submits; the steps run out (max_steps_reached); or the agent saves exactly the
same code for the third time in a row (repeated_code, the same idea as in B: it is stuck).

Only the first action of a reply is run. Some models keep writing the next THOUGHT/ACTION themselves.
Generation stops as soon as that starts (like the stop sequences in ReAct), the extra part is cut off
(counted in multi_action_replies) and it is not kept in the history.

Format problems are counted separately from wrong answers:
  format_fixes   no valid ACTION line, but the reply had code -> taken as write_code
  format_errors  no usable reply: no valid action, or write_code without code -> the model is told
                 exactly what was wrong
  code_ignored   code was written under another action (e.g. run_tests); it is NOT saved and the model is told

    python src/strategy_c.py --selftest
    python -u src/strategy_c.py --backend hf --model Qwen/Qwen2.5-Coder-7B-Instruct \
        --dataset mbpp --data-file Dataset/mbpp_initial_10.jsonl --max-new-tokens 512
"""
import argparse
import csv
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline_oneshot as base  # noqa: E402
import strategy_b as sb  # noqa: E402

STRATEGY = "C_agent"
ACTIONS = ("write_code", "read_code", "check_syntax", "run_tests", "submit")
CSV_FIELDS = base.CSV_FIELDS + [
    "steps", "stop_reason", "n_write_code", "n_read_code", "n_check_syntax", "n_run_tests",
    "format_errors", "format_fixes", "code_ignored", "multi_action_replies", "tested_before_submit",
    "first_code_status", "final_dev_passed",
]


EXAMPLES = {
    # default: a small made-up function, not related to any task
    "add": "THOUGHT: I add the two numbers.\nACTION: write_code\n```python\ndef add(a, b):\n    return a + b\n```",
    # for the check whether the example's function name changes the model's naming: no real function in it
    "placeholder": "THOUGHT: <why this code>\nACTION: write_code\n```python\n<your complete code>\n```",
}
EXAMPLE = "add"  # set by --example


def agent_system_prompt(max_steps, example=None):
    return (
        "You are an expert Python programmer working as an agent. You solve one programming task "
        "step by step. In every reply, choose exactly ONE action.\n\n"
        "Actions:\n"
        "  write_code   - save your complete solution. Put the full code in one ```python block.\n"
        "  read_code    - show the code you saved.\n"
        "  check_syntax - check the saved code for syntax errors.\n"
        "  run_tests    - run the saved code on the example test from the task.\n"
        "  submit       - finish. The saved code is your final answer.\n\n"
        "Reply format:\n"
        "THOUGHT: <one or two short sentences>\n"
        "ACTION: <one action name>\n\n"
        "For write_code, the code must be in the SAME reply, right after the ACTION line. Example:\n"
        + EXAMPLES[example or EXAMPLE] + "\n\n"
        f"You can take at most {max_steps} actions. If you run out of actions, the saved code is submitted."
    )


def first_action(text):
    """Cut the reply after its first action. Models sometimes go on and write the next THOUGHT/ACTION
    themselves (pretending to see the result). Only the first action is run, like in mini-SWE-agent."""
    m = re.search(r"ACTION\s*:", text)
    if not m:
        return text, False
    nxt = re.search(r"(?m)^\s*\**\s*(THOUGHT|ACTION|OBSERVATION)\s*:", text[m.end():])
    if not nxt:
        return text, False
    return text[:m.end() + nxt.start()].rstrip(), True


def parse_action(text, func_name=None):
    """Returns (action, code or None, format status, extraction info).
    Status: ok, fixed (code but no ACTION line), missing_code (write_code without code), error (nothing usable)."""
    m = re.search(r"ACTION\s*:\s*\**\s*([a-zA-Z_]+)", text)
    code, ext = base.extract_code_info(text, func_name) if "```" in text else (None, None)
    if m and m.group(1).lower() in ACTIONS:
        action = m.group(1).lower()
        if action == "write_code" and not (code and code.strip()):
            return None, None, "missing_code", None  # chose write_code but sent no code
        return action, code, "ok", ext
    if code and code.strip():
        return "write_code", code, "fixed", ext  # no ACTION line but there is code: take it as write_code
    return None, None, "error", None


def run_task(task, backend, model_label, timeout, max_steps):
    messages = [{"role": "system", "content": agent_system_prompt(max_steps)},
                {"role": "user", "content": base.make_prompt(task)}]
    gen_t, tok_in, tok_out = [0.0, 0.0], [0, 0], [0, 0]  # [first call, later calls]
    t_feedback = 0.0
    counts = Counter()
    code, first_code_status, stop = "", None, "max_steps_reached"
    tested_since_write = tested_before_submit = False
    same_in_row = 0  # how many times in a row the agent saved the code it already had
    steps, first_prompt = [], ""
    limit = getattr(backend, "max_new_tokens", 10**9)

    for step in range(max_steps):
        k = 0 if step == 0 else 1
        t0 = time.perf_counter()
        text, i, o = backend.generate_chat(messages, stop_when=lambda t: first_action(t)[1])
        g = time.perf_counter() - t0
        gen_t[k] += g
        tok_in[k] += i
        tok_out[k] += o
        if step == 0:
            first_prompt = getattr(backend, "last_prompt", "")
        reply, cut = first_action(text)
        counts["multi_action_replies"] += cut
        messages.append({"role": "assistant", "content": reply})  # history keeps only what was run

        action, new_code, fmt, ext = parse_action(reply, task.get("func_name"))
        counts["format_errors"] += fmt in ("error", "missing_code")
        counts["format_fixes"] += fmt == "fixed"
        ignored = action not in (None, "write_code") and new_code is not None and new_code.strip() \
            and base.remove_model_tests(new_code).strip() != code.strip()
        counts["code_ignored"] += bool(ignored)
        dt = 0.0
        if fmt == "missing_code":
            obs = ("You chose write_code but your reply has no code, so nothing was saved. Send the full "
                   "solution in a ```python block in your next reply.")
        elif action is None:
            obs = ("No valid action found. Use the format:\nTHOUGHT: ...\nACTION: <write_code | read_code | "
                   "check_syntax | run_tests | submit>\nFor write_code, put the full code in a ```python block "
                   "in the same reply.")
        elif action == "write_code":
            new = base.remove_model_tests(new_code)
            same = bool(code) and new.strip() == code.strip()
            same_in_row = same_in_row + 1 if same else 0
            code = new
            counts["n_write_code"] += 1
            counts["ambiguous_extractions"] += ext["ambiguous"]
            tested_since_write = False
            if first_code_status is None:  # diagnostic: first written code on all tests (not timed)
                first_code_status = base.run_code(base.add_tests_to_code(task, code), timeout)[0]
            obs = f"Code saved ({len(code.splitlines())} lines)."
            if same:
                obs += " It is exactly the same code as before."
        elif action == "read_code":
            counts["n_read_code"] += 1
            obs = "Current code:\n```python\n" + code + "\n```" if code else "No code saved yet. Use write_code."
        elif action == "check_syntax":
            counts["n_check_syntax"] += 1
            err = sb.run_compiler(code) if code else "No code saved yet. Use write_code."
            obs = err or "No syntax errors."
        elif action == "run_tests":
            counts["n_run_tests"] += 1
            if code:
                ok, fb, dt, _ = sb.run_tester(task, code, timeout)
                obs = "PASSED: the example test passed." if ok else "FAILED: " + fb
                tested_since_write = True
            else:
                obs = "No code saved yet. Use write_code."
        else:  # submit
            obs = "Submitted."
        if ignored and action != "submit":
            obs += "\nNote: the code in your reply was NOT saved. Use write_code to save code."
        t_feedback += dt
        steps.append({"step": step, "raw_output": text, "cut_after_first_action": cut,
                      "action": action, "format": fmt, "extraction": ext,
                      "code_ignored": bool(ignored),
                      "code": code if action == "write_code" else None, "observation": obs,
                      "input_tokens": i, "output_tokens": o, "hit_token_limit": o >= limit,
                      "gen_time_s": round(g, 3), "feedback_exec_time_s": round(dt, 3)})
        if action == "submit":
            stop, tested_before_submit = "submitted", tested_since_write
            break
        if same_in_row >= 2:  # third time the same code: the agent is stuck, like B's repeated_code
            stop = "repeated_code"
            break
        messages.append({"role": "user", "content": "OBSERVATION: " + obs})

    # final evaluation on all 3 tests (same procedure as A and B)
    if code.strip():
        status, err, t_eval = base.run_code(base.add_tests_to_code(task, code), timeout)
        final_dev = int(sb.run_tester(task, code, timeout)[0])  # diagnostic, not timed
    else:
        status, err, t_eval, final_dev = "no_code", "no code was written", 0.0, 0
    t_gen = sum(gen_t)
    row = {
        "task_id": task["task_id"], "difficulty": task.get("difficulty", ""),
        "model": model_label, "strategy": STRATEGY,
        "solved": int(status == "passed"), "status": status, "error": err,
        "llm_calls": len(steps), "repair_calls": len(steps) - 1,  # in C: all calls after the first one
        "input_tokens": sum(tok_in), "output_tokens": sum(tok_out),
        "initial_input_tokens": tok_in[0], "initial_output_tokens": tok_out[0],
        "repair_input_tokens": tok_in[1], "repair_output_tokens": tok_out[1],
        "gen_time_s": round(t_gen, 3), "initial_gen_time_s": round(gen_t[0], 3), "repair_gen_time_s": round(gen_t[1], 3),
        "eval_exec_time_s": round(t_eval, 3), "feedback_exec_time_s": round(t_feedback, 3),
        "exec_time_s": round(t_eval + t_feedback, 3), "total_time_s": round(t_gen + t_eval + t_feedback, 3),
        "ambiguous_extractions": counts["ambiguous_extractions"],
        "steps": len(steps), "stop_reason": stop,
        "n_write_code": counts["n_write_code"], "n_read_code": counts["n_read_code"],
        "n_check_syntax": counts["n_check_syntax"], "n_run_tests": counts["n_run_tests"],
        "format_errors": counts["format_errors"], "format_fixes": counts["format_fixes"],
        "code_ignored": counts["code_ignored"], "multi_action_replies": counts["multi_action_replies"],
        "tested_before_submit": int(tested_before_submit), "first_code_status": first_code_status or "no_code",
        "final_dev_passed": final_dev,
    }
    log = {"task_id": task["task_id"], "status": status, "error": err, "stop_reason": stop,
           "first_prompt_rendered": first_prompt, "final_code": code, "steps": steps}
    return row, log


INT_C = ["steps", "n_write_code", "n_read_code", "n_check_syntax", "n_run_tests", "format_errors",
         "format_fixes", "code_ignored", "multi_action_replies", "tested_before_submit", "final_dev_passed"]


def summarize(rows, settings=None):
    s = base.summarize(rows, settings)
    s["strategy"] = STRATEGY
    s["final_success_rate"] = s["solve_rate"]
    n = len(rows) or 1
    s["avg_steps"] = round(sum(r["steps"] for r in rows) / n, 3)
    for k in INT_C[1:]:
        s["total_" + k] = sum(int(r[k]) for r in rows)
    s["first_code_solve_rate"] = round(sum(r["first_code_status"] == "passed" for r in rows) / n, 4)
    s["stop_reason_counts"] = dict(Counter(r["stop_reason"] for r in rows))
    s["dev_passed_but_hidden_failed"] = sum(1 for r in rows if int(r["final_dev_passed"]) and not r["solved"])
    # same idea as B's repair_summary: the first code the agent wrote, and what happened after
    first_bad = [r for r in rows if r["first_code_status"] != "passed"]
    fixed = sum(r["solved"] for r in first_bad)
    s["repair_summary"] = {"tasks": len(rows), "initially_correct": len(rows) - len(first_bad),
                           "initially_failed": len(first_bad), "repaired": fixed,
                           "still_failed": len(first_bad) - fixed,
                           "broken_by_repair": sum(1 for r in rows if r["first_code_status"] == "passed" and not r["solved"]),
                           "repair_success_rate": round(fixed / len(first_bad), 4) if first_bad else None}
    return s


# --------------------------------------------------------------------------- #
class ScriptedAgent:
    """Selftest helper: returns a fixed list of replies per task, records what it was shown."""

    def __init__(self, replies):
        self.replies, self.calls, self.seen = replies, Counter(), []
        self.max_new_tokens = 512

    def generate_chat(self, messages, stop_when=None):
        tid = messages[1]["content"]  # task text identifies the script
        seq = self.replies[tid]
        out = seq[min(self.calls[tid], len(seq) - 1)]
        self.calls[tid] += 1
        self.seen.append("\n".join(m["content"] for m in messages))
        return out, sum(len(m["content"].split()) for m in messages), len(out.split())


def selftest():
    row = {"task_id": 0, "prompt": "Write a function to add two numbers.", "test_imports": [],
           "test_list": ["assert add(1, 2) == 3", "assert add(10, 5) == 15", "assert add(-1, 1) == 0"]}
    good = "```python\ndef add(a, b):\n    return a + b\n```"
    bad = "```python\ndef add(a, b):\n    return a - b\n```"
    W = lambda c: "THOUGHT: write it.\nACTION: write_code\n" + c
    T = "THOUGHT: test it.\nACTION: run_tests"
    S = "THOUGHT: done.\nACTION: submit"
    cases = {  # replies, expected (solved, steps, stop_reason, n_run_tests, format_errors, format_fixes)
        "write_test_submit": ([W(good), T, S], (1, 3, "submitted", 1, 0, 0)),
        "fix_after_test": ([W(bad), T, W(good), T, S], (1, 5, "submitted", 2, 0, 0)),
        "submit_untested": ([W(bad), S], (0, 2, "submitted", 0, 0, 0)),
        "never_submits": ([W(bad), T], (0, 6, "max_steps_reached", 5, 0, 0)),
        "no_action_line": ([good, S], (1, 2, "submitted", 0, 0, 1)),
        "garbage_then_ok": (["I think so.", W(good), S], (1, 3, "submitted", 0, 1, 0)),
        "action_then_code": (["THOUGHT: write it.\nACTION: write_code", good, S], (1, 3, "submitted", 0, 1, 1)),
        "same_code_3_times": ([W(bad), T, W(bad), T, W(bad)], (0, 5, "repeated_code", 2, 0, 0)),
        "two_actions_in_one_reply": ([W(bad) + "\nTHOUGHT: now test.\nACTION: run_tests", T, W(good), T, S],
                                     (1, 5, "submitted", 2, 0, 0)),
        "code_under_run_tests": ([W(bad), "THOUGHT: fix and test.\nACTION: run_tests\n" + good, W(good), S],
                                 (1, 4, "submitted", 1, 0, 0)),
    }
    agent_replies, tasks = {}, []
    for k, (replies, _) in cases.items():
        t = dict(base.normalize(row, "mbpp"), setup="", dev_test=row["test_list"][0])
        t["task_id"], t["prompt"] = k, f"[{k}] " + t["prompt"]
        tasks.append(t)
        agent_replies[base.make_prompt(t)] = replies
    agent = ScriptedAgent(agent_replies)
    rows = [run_task(t, agent, "mock", timeout=3, max_steps=6)[0] for t in tasks]
    ok = True
    for r in rows:
        want = cases[r["task_id"]][1]
        got = (r["solved"], r["steps"], r["stop_reason"], r["n_run_tests"], r["format_errors"], r["format_fixes"])
        ok &= got == want
        print(f"  {'OK ' if got == want else 'FAIL'} {r['task_id']:<18} solved={r['solved']} steps={r['steps']} "
              f"stop={r['stop_reason']:<17} tests={r['n_run_tests']} fmt_err={r['format_errors']} fixes={r['format_fixes']}")
    no_leak = all("add(10, 5)" not in s and "add(-1, 1)" not in s for s in agent.seen)
    fb = any("FAILED: AssertionError: expected 3, got -1" in s for s in agent.seen)
    multi = next(r for r in rows if r["task_id"] == "two_actions_in_one_reply")["multi_action_replies"] == 1
    told = any("has no code, so nothing was saved" in s for s in agent.seen)
    ign = next(r for r in rows if r["task_id"] == "code_under_run_tests")["code_ignored"] == 1 and \
        any("was NOT saved" in s for s in agent.seen)
    timing = all(abs(r["exec_time_s"] - r["eval_exec_time_s"] - r["feedback_exec_time_s"]) < 0.002 for r in rows)
    for name, g in [("hidden tests 2-3 never shown to the agent", no_leak),
                    ("run_tests gives B's feedback (expected vs got)", fb),
                    ("exec = eval + feedback", timing),
                    ("code under run_tests is not saved, and the agent is told", ign),
                    ("write_code without code: the agent is told exactly what is missing", told),
                    ("only the first action of a reply is run", multi)]:
        ok &= g
        print(f"  {'OK ' if g else 'FAIL'} {name}")
    print("summary:", json.dumps(summarize(rows)))
    print("SELFTEST PASSED" if ok else "SELFTEST FAILED")
    return 0 if ok else 1


def read_rows(csv_path):
    rows = []
    for r in base._read_rows(csv_path):
        try:
            for k in INT_C:
                r[k] = int(r[k])
        except (KeyError, TypeError, ValueError):
            continue
        rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser(description="Strategy C: LLM agent")
    ap.add_argument("--backend", choices=["hf"], default="hf")
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    ap.add_argument("--dataset", choices=["mbpp"], default="mbpp")
    ap.add_argument("--data-file", help="team dataset, e.g. Dataset/mbpp_initial_10.jsonl")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--max-steps", type=int, default=24,
                    help="max agent actions = LLM calls per task (default 24: room for 6 code versions like B)")
    ap.add_argument("--timeout", type=int, default=10)
    ap.add_argument("--mem-mb", type=int, default=1024)
    ap.add_argument("--max-new-tokens", type=int, default=512, help="per LLM call, same as A and B")
    ap.add_argument("--quant", choices=["8bit"])
    ap.add_argument("--no-remote-code", action="store_true")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--example", choices=list(EXAMPLES), default="add",
                    help="write_code example in the system message (placeholder = no real function, for the check)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    base.LIMITS["mem_mb"] = a.mem_mb
    global EXAMPLE
    EXAMPLE = a.example
    if a.selftest:
        sys.exit(selftest())

    tasks = sb.load_tasks(a.dataset, a.data_file)
    if a.limit:
        tasks = tasks[:a.limit]
    backend = base.HFBackend(a.model, a.max_new_tokens, a.quant, not a.no_remote_code)
    label = a.model + (f"@{a.quant}" if a.quant else "")
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ex = "" if a.example == "add" else f"ex-{a.example}_"
    stem = f"agent_s{a.max_steps}_{ex}{re.sub(r'[^\w.-]', '_', label)}_{a.dataset}_{Path(a.data_file).stem}"
    csv_path, log_path = out / f"{stem}.csv", out / f"{stem}_code.jsonl"
    settings = {"model": a.model, "max_new_tokens": a.max_new_tokens, "decoding": "greedy", "timeout_s": a.timeout,
                "mem_limit_mb": a.mem_mb, "quant": a.quant or "none", "remote_code": not a.no_remote_code,
                "max_steps": a.max_steps, "max_llm_calls": a.max_steps, "actions": list(ACTIONS),
                "stop_rules": "submit; max_steps reached; repeated_code (same code saved 3 times in a row)",
                "one_action_per_reply": "generation stops when a second action starts; the rest is cut off",
                "code_extraction": base.EXTRACTION_RULE,
                "dev_tests": "test 1 (shown in prompt), same as B", "final_tests": "all 3 tests",
                "data": a.data_file, "system_prompt": agent_system_prompt(a.max_steps),
                "example": a.example,
                "environment": base.env_info(backend)}

    rows = read_rows(csv_path) if a.resume and csv_path.exists() else []
    if rows:
        print(f"Resuming: {len(rows)} tasks already finished")
    done = {r["task_id"] for r in rows}
    with open(csv_path, "w", newline="", encoding="utf-8") as fh, \
            open(log_path, "a" if rows else "w", encoding="utf-8") as lf:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
        fh.flush()
        for i, task in enumerate(tasks, 1):
            if task["task_id"] in done:
                continue
            row, log = run_task(task, backend, label, a.timeout, a.max_steps)
            rows.append(row)
            w.writerow(row)
            fh.flush()
            lf.write(json.dumps(log, ensure_ascii=False) + "\n")
            lf.flush()
            acts = " ".join(s["action"] or "?" for s in log["steps"])
            print(f"[{i}/{len(tasks)}] {row['task_id']} ({row['difficulty']}): {row['status']}  steps={row['steps']} "
                  f"({row['stop_reason']})  fmt_err={row['format_errors']}  {row['total_time_s']}s  | {acts}")
    summary = summarize(rows, settings)
    (out / f"{stem}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("settings", "by_difficulty")}, indent=2))


if __name__ == "__main__":
    main()
