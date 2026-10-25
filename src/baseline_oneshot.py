#!/usr/bin/env python3
"""
Strategy A - one-shot baseline (n = 0 corrections).

Flow per task:  task -> LLM generates code ONCE -> run unit tests -> log result.
No compiler feedback, no retries. This is the reference point for strategies B and C.

Fixed on purpose (team decision 12 Oct: "everything fixed, no tuning"):
  * greedy decoding (temperature 0 / do_sample=False), so runs are reproducible
  * max_new_tokens identical for every model and every strategy

Quick check that your environment works (no model, no dataset needed):
    python baseline_oneshot.py --selftest

Real runs (start with --limit 10 as a pilot):
    python baseline_oneshot.py --backend hf     --model Qwen/Qwen2.5-Coder-7B-Instruct --dataset humaneval --limit 10
    python baseline_oneshot.py --backend ollama --model qwen2.5-coder:7b               --dataset mbpp      --limit 10
    python baseline_oneshot.py --backend hf --model <id> --data-file sorted_dataset.jsonl --dataset humaneval

Outputs (in --out-dir, default ./results):
    oneshot_<model>_<dataset>.csv   one row per task (same columns B and C should reuse)
    oneshot_<model>_<dataset>.json  summary: pass@1, time, tokens, tokens per solved task
"""
import argparse
import csv
import json
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
from collections import Counter
from pathlib import Path

STRATEGY = "A_oneshot"
SYSTEM_PROMPT = (
    "You are an expert Python programmer. "
    "Reply with the complete Python code in a single ```python code block and nothing else."
)
CSV_FIELDS = [
    "task_id", "model", "strategy", "solved", "status", "error", "iterations",
    "input_tokens", "output_tokens", "gen_time_s", "exec_time_s", "total_time_s",
]


# --------------------------------------------------------------------------- #
# Dataset  (replace load_tasks() if the team's "sorted dataset" has another format)
# --------------------------------------------------------------------------- #
def normalize(row, dataset):
    """Turn a HumanEval or MBPP row into one common task dict."""
    if dataset == "humaneval":
        ep = row["entry_point"]
        return {
            "task_id": row["task_id"],
            "prompt": row["prompt"],
            "entry_point": ep,
            "tests": row["test"] + f"\ncheck({ep})\n",
            "mock_output": row["prompt"] + row.get("canonical_solution", ""),
        }
    tests = "\n".join(row["test_list"])
    prompt = f"{row['text']}\nYour code must pass this test:\n{row['test_list'][0]}"
    return {
        "task_id": f"MBPP/{row['task_id']}",
        "prompt": prompt,
        "entry_point": None,
        "tests": (row.get("test_setup_code") or "") + "\n" + tests + "\n",
        "mock_output": row.get("code", ""),
    }


def load_tasks(dataset, data_file=None, limit=None):
    if data_file:
        rows = [json.loads(line) for line in open(data_file, encoding="utf-8") if line.strip()]
    else:
        from datasets import load_dataset  # pip install datasets
        if dataset == "humaneval":
            rows = list(load_dataset("openai/openai_humaneval", split="test"))
        else:
            rows = list(load_dataset("google-research-datasets/mbpp", "full", split="test"))
    tasks = [normalize(r, dataset) for r in rows]
    return tasks[:limit] if limit else tasks


# --------------------------------------------------------------------------- #
# LLM backends  (Nihar's part can replace these; only generate() must keep its shape)
# --------------------------------------------------------------------------- #
class HFBackend:
    """Hugging Face transformers - works on a laptop GPU or the KTU JupyterLab server."""

    def __init__(self, model_id, max_new_tokens):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.max_new_tokens = max_new_tokens
        self.tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype="auto", device_map="auto", trust_remote_code=True
        )

    def generate(self, system, user, task=None):
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        text = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = self.tok(text, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        out = self.model.generate(
            **enc,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,  # greedy -> reproducible
            pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id,
        )
        n_in = enc["input_ids"].shape[-1]
        gen = out[0][n_in:]
        return self.tok.decode(gen, skip_special_tokens=True), n_in, len(gen)


class OllamaBackend:
    """Local Ollama server (default http://localhost:11434)."""

    def __init__(self, model, max_new_tokens, host="http://localhost:11434"):
        self.model, self.max_new_tokens, self.host = model, max_new_tokens, host

    def generate(self, system, user, task=None):
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "stream": False,
            "options": {"temperature": 0, "seed": 0, "num_predict": self.max_new_tokens},
        }
        req = urllib.request.Request(
            self.host + "/api/chat", json.dumps(body).encode(), {"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=900) as resp:
            r = json.load(resp)
        return r["message"]["content"], r.get("prompt_eval_count", 0), r.get("eval_count", 0)


class MockBackend:
    """Returns the reference solution. Only for testing the pipeline itself."""

    def generate(self, system, user, task=None):
        out = "```python\n" + task["mock_output"] + "\n```"
        return out, len(user.split()), len(out.split())


# --------------------------------------------------------------------------- #
# Code extraction + sandboxed execution
# --------------------------------------------------------------------------- #
def extract_code(text):
    blocks = re.findall(r"```(?:python|py)?[ \t]*\n(.*?)(?:```|\Z)", text, re.S)
    if not blocks:
        return text.strip("\n")
    for b in blocks:
        if "def " in b:
            return b
    return blocks[0]


def build_user_prompt(task):
    if task["entry_point"]:
        return (
            "Complete the following Python function. Return the complete function "
            "(imports and signature included).\n\n```python\n" + task["prompt"] + "\n```"
        )
    return task["prompt"]


def build_program(task, code):
    if task["entry_point"] and f"def {task['entry_point']}" not in code:
        program = task["prompt"] + code  # model returned only a function body
    else:
        program = task["prompt"] + "\n\n" + code if task["entry_point"] else code
    return program + "\n\n" + task["tests"]


def classify(last_line):
    if re.search(r"(SyntaxError|IndentationError|TabError)", last_line):
        return "syntax_error"
    if "AssertionError" in last_line:
        return "test_failure"
    return "runtime_error"


def run_program(program, timeout):
    """Run generated code + tests in a separate process with a time limit.
    NOTE: this is a subprocess, not a hardened sandbox. Run on your own machine/server only."""
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "prog.py"
        path.write_text(program, encoding="utf-8")
        t0 = time.perf_counter()
        try:
            p = subprocess.run([sys.executable, str(path)], capture_output=True,
                               text=True, timeout=timeout, cwd=d)
        except subprocess.TimeoutExpired:
            return "timeout", f"exceeded {timeout}s", time.perf_counter() - t0
        dt = time.perf_counter() - t0
    if p.returncode == 0:
        return "passed", "", dt
    lines = p.stderr.strip().splitlines()
    last = lines[-1] if lines else "unknown error"
    return classify(last), last[:300], dt


# --------------------------------------------------------------------------- #
# One task, summary
# --------------------------------------------------------------------------- #
def run_task(task, backend, model_name, timeout):
    t0 = time.perf_counter()
    text, n_in, n_out = backend.generate(SYSTEM_PROMPT, build_user_prompt(task), task)
    gen_time = time.perf_counter() - t0
    code = extract_code(text)
    if not code.strip():
        status, err, exec_time = "no_code", "empty model output", 0.0
    else:
        status, err, exec_time = run_program(build_program(task, code), timeout)
    return {
        "task_id": task["task_id"], "model": model_name, "strategy": STRATEGY,
        "solved": int(status == "passed"), "status": status, "error": err,
        "iterations": 1,  # one generation, zero corrections (n = 0)
        "input_tokens": n_in, "output_tokens": n_out,
        "gen_time_s": round(gen_time, 3), "exec_time_s": round(exec_time, 3),
        "total_time_s": round(gen_time + exec_time, 3),
    }


def summarize(rows):
    n = len(rows)
    solved = sum(r["solved"] for r in rows)
    tok_in = sum(r["input_tokens"] for r in rows)
    tok_out = sum(r["output_tokens"] for r in rows)
    return {
        "strategy": STRATEGY, "model": rows[0]["model"] if rows else None, "tasks": n,
        "solved": solved, "pass@1": round(solved / n, 4) if n else None,
        "avg_total_time_s": round(sum(r["total_time_s"] for r in rows) / n, 3) if n else None,
        "total_input_tokens": tok_in, "total_output_tokens": tok_out,
        "tokens_per_solved_task": round((tok_in + tok_out) / solved, 1) if solved else None,
        "status_counts": dict(Counter(r["status"] for r in rows)),
    }


# --------------------------------------------------------------------------- #
# Self-test: exercises every outcome category without any model or dataset
# --------------------------------------------------------------------------- #
def selftest():
    stub = 'def f(a, b):\n    """add two numbers"""\n'
    tests = "def check(candidate):\n    assert candidate(1, 2) == 3\n\ncheck(f)\n"
    cases = {
        "ok": ("def f(a, b):\n    return a + b", "passed"),
        "wrong_logic": ("def f(a, b):\n    return a - b", "test_failure"),
        "bad_syntax": ("def f(a, b)\n    return a + b", "syntax_error"),
        "crash": ("def f(a, b):\n    return undefined_name + a", "runtime_error"),
        "infinite_loop": ("def f(a, b):\n    while True:\n        pass", "timeout"),
    }
    tasks = [{"task_id": k, "prompt": stub, "entry_point": "f", "tests": tests, "mock_output": v[0]}
             for k, v in cases.items()]
    rows = [run_task(t, MockBackend(), "mock", timeout=2) for t in tasks]
    ok = True
    for r in rows:
        expected = cases[r["task_id"]][1]
        flag = "OK " if r["status"] == expected else "FAIL"
        ok &= r["status"] == expected
        print(f"  {flag} {r['task_id']:<14} -> {r['status']:<14} (expected {expected})")
    print("summary:", json.dumps(summarize(rows)))
    print("SELFTEST PASSED" if ok else "SELFTEST FAILED")
    return 0 if ok else 1


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Strategy A one-shot baseline")
    ap.add_argument("--backend", choices=["hf", "ollama", "mock"], default="hf")
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    ap.add_argument("--dataset", choices=["humaneval", "mbpp"], default="humaneval")
    ap.add_argument("--data-file", help="local JSONL in HumanEval/MBPP format (else downloads)")
    ap.add_argument("--limit", type=int, help="only the first N tasks (use 10-20 for a pilot)")
    ap.add_argument("--timeout", type=int, default=10, help="seconds allowed to run each solution")
    ap.add_argument("--max-new-tokens", type=int, default=512, help="keep identical for all runs")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        sys.exit(selftest())

    tasks = load_tasks(a.dataset, a.data_file, a.limit)
    if a.backend == "hf":
        backend = HFBackend(a.model, a.max_new_tokens)
    elif a.backend == "ollama":
        backend = OllamaBackend(a.model, a.max_new_tokens)
    else:
        backend = MockBackend()

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"oneshot_{re.sub(r'[^\w.-]', '_', a.model)}_{a.dataset}"
    rows = []
    with open(out / f"{stem}.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        w.writeheader()
        for i, task in enumerate(tasks, 1):
            row = run_task(task, backend, a.model, a.timeout)
            rows.append(row)
            w.writerow(row)
            fh.flush()  # keep partial results if the run crashes
            print(f"[{i}/{len(tasks)}] {row['task_id']}: {row['status']} "
                  f"({row['total_time_s']}s, {row['input_tokens']}+{row['output_tokens']} tok)")
    summary = summarize(rows)
    (out / f"{stem}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
