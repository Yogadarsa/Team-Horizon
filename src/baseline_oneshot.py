#!/usr/bin/env python3
"""
Strategy A - one-shot baseline (n = 0 corrections).

Flow per task:  task -> LLM generates code ONCE -> run unit tests -> log result.
No compiler feedback, no retries. This is the reference point for strategies B and C.

Fixed on purpose :
  * greedy decoding (do_sample=False), so runs are reproducible
  * max_new_tokens identical for every model and every strategy

Execution of generated code :
  * every solution runs in its own temporary folder, as a separate Python process in isolated mode (-I)
  * limits per run: wall-clock timeout, CPU time, memory, file size
  * on timeout the whole process group is killed (also child processes the code started)
  This is still not a container sandbox: network and the user's home folder are not blocked.

Quick check that your environment works (no model, no dataset needed):
    python baseline_oneshot.py --selftest

Team dataset:
    python src/baseline_oneshot.py --backend hf --model Qwen/Qwen2.5-Coder-7B-Instruct --dataset mbpp --data-file Dataset/mbpp_initial_10.jsonl
    python src/baseline_oneshot.py --backend hf --model deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct --quant 8bit --no-remote-code --dataset mbpp --data-file Dataset/mbpp_final_30.jsonl

Full official datasets (downloaded from Hugging Face):
    python src/baseline_oneshot.py --backend hf --model Qwen/Qwen2.5-Coder-7B-Instruct --dataset humaneval

Outputs (in --out-dir, default ./results); <name> = model + dataset (+ data file name):
    oneshot_<name>.csv          one row per task (same columns B and C should reuse)
    oneshot_<name>.json         summary: pass@1 (also per difficulty), time, tokens, tokens per solved task
    oneshot_<name>_code.jsonl   the model's raw answer and the extracted code for every task
"""
import argparse
import csv
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

STRATEGY = "A_oneshot"
SYSTEM_PROMPT = (
    "You are an expert Python programmer. "
    "Reply with the complete Python code in a single ```python code block and nothing else."
)
CSV_FIELDS = [
    "task_id", "difficulty", "model", "strategy", "solved", "status", "error", "iterations",
    "input_tokens", "output_tokens", "gen_time_s", "exec_time_s", "total_time_s",
]
LIMITS = {"mem_mb": 1024, "fsize_mb": 10}  # set from the command line in main()


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
def normalize(row, dataset):
    """Turn a HumanEval row, an official MBPP row, or a row of the team dataset into one task dict."""
    if dataset == "humaneval":
        ep = row["entry_point"]
        return {
            "task_id": row["task_id"],
            "difficulty": row.get("difficulty", ""),
            "prompt": row["prompt"],
            "entry_point": ep,
            "tests": row["test"] + f"\ncheck({ep})\n",
            "mock_output": row["prompt"] + row.get("canonical_solution", ""),
        }
    # official MBPP "full" uses text + test_setup_code; the dataset uses prompt + test_imports
    text = row["text"] if "text" in row else row["prompt"]
    setup = row.get("test_setup_code") or "\n".join(row.get("test_imports") or [])
    prompt = f"{text}\nYour code must pass this test:\n{row['test_list'][0]}"
    return {
        "task_id": f"MBPP/{row['task_id']}",
        "difficulty": row.get("difficulty", ""),
        "prompt": prompt,
        "entry_point": None,
        "tests": setup + "\n" + "\n".join(row["test_list"]) + "\n",
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
# LLM backends  (only generate() must keep its shape)
# --------------------------------------------------------------------------- #
class HFBackend:
    """Hugging Face transformers - KTU JupyterLab server or any GPU machine."""

    def __init__(self, model_id, max_new_tokens, quant=None, remote_code=True):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.max_new_tokens = max_new_tokens
        self.tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=remote_code)
        kwargs = dict(device_map="auto", trust_remote_code=remote_code)
        if quant == "8bit":  # same setting as 02_Models_setup.ipynb (DeepSeek)
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        else:
            kwargs["torch_dtype"] = "auto"
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)

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
# Code extraction + isolated execution
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


# Runs inside the child process: sets its own limits, then executes the program.
# (Setting limits in the child itself is safe even when the parent has many threads, e.g. torch.)
_LAUNCHER = r"""
import resource, runpy, sys
mem, cpu, fsize, path = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
for lim, val in ((resource.RLIMIT_AS, mem), (resource.RLIMIT_CPU, cpu), (resource.RLIMIT_FSIZE, fsize)):
    try:
        resource.setrlimit(lim, (val, val))
    except (ValueError, OSError):
        pass
sys.argv = [path]
runpy.run_path(path, run_name="__main__")
"""


def run_program(program, timeout):
    """Run generated code + tests in a separate, isolated Python process with time and resource limits."""
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "prog.py"
        path.write_text(program, encoding="utf-8")
        cmd = [sys.executable, "-I", "-c", _LAUNCHER,
               str(LIMITS["mem_mb"] * 2**20), str(timeout + 1), str(LIMITS["fsize_mb"] * 2**20), str(path)]
        t0 = time.perf_counter()
        p = subprocess.Popen(cmd, cwd=d, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             stdin=subprocess.DEVNULL, text=True, start_new_session=True)
        try:
            _, stderr = p.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(p.pid, signal.SIGKILL)  # kill the program and anything it started
            except ProcessLookupError:
                pass
            p.communicate()
            return "timeout", f"exceeded {timeout}s", time.perf_counter() - t0
        dt = time.perf_counter() - t0
    if p.returncode == 0:
        return "passed", "", dt
    if p.returncode == -signal.SIGXCPU:
        return "timeout", "CPU time limit exceeded", dt
    lines = stderr.strip().splitlines()
    last = lines[-1] if lines else f"killed (exit code {p.returncode})"
    if "MemoryError" in last:
        return "runtime_error", f"MemoryError (limit {LIMITS['mem_mb']} MB)", dt
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
    row = {
        "task_id": task["task_id"], "difficulty": task.get("difficulty", ""),
        "model": model_name, "strategy": STRATEGY,
        "solved": int(status == "passed"), "status": status, "error": err,
        "iterations": 1,  # one generation, zero corrections (n = 0)
        "input_tokens": n_in, "output_tokens": n_out,
        "gen_time_s": round(gen_time, 3), "exec_time_s": round(exec_time, 3),
        "total_time_s": round(gen_time + exec_time, 3),
    }
    return row, {"task_id": task["task_id"], "status": status, "raw_output": text, "code": code}


def _stats(rows):
    n = len(rows)
    solved = sum(r["solved"] for r in rows)
    tok_in = sum(r["input_tokens"] for r in rows)
    tok_out = sum(r["output_tokens"] for r in rows)
    return {
        "tasks": n, "solved": solved, "pass@1": round(solved / n, 4) if n else None,
        "avg_total_time_s": round(sum(r["total_time_s"] for r in rows) / n, 3) if n else None,
        "total_input_tokens": tok_in, "total_output_tokens": tok_out,
        "tokens_per_solved_task": round((tok_in + tok_out) / solved, 1) if solved else None,
        "status_counts": dict(Counter(r["status"] for r in rows)),
    }


def summarize(rows, settings=None):
    s = {"strategy": STRATEGY, "model": rows[0]["model"] if rows else None, **_stats(rows)}
    levels = defaultdict(list)
    for r in rows:
        if r.get("difficulty"):
            levels[r["difficulty"]].append(r)
    if levels:
        order = {"Easy": 0, "Medium": 1, "Hard": 2}
        s["by_difficulty"] = {k: _stats(v) for k, v in sorted(levels.items(), key=lambda kv: order.get(kv[0], 9))}
    if settings:
        s["settings"] = settings
    return s


# --------------------------------------------------------------------------- #
# Self-test: every outcome category, the limits, and the team-dataset format, without any model
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
        "memory_hog": ("def f(a, b):\n    x = bytearray(4 * 2**30)\n    return a + b", "runtime_error"),
        "child_process": ("import subprocess, sys\n"
                          "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
                          "def f(a, b):\n    while True:\n        pass", "timeout"),
    }
    tasks = [{"task_id": k, "prompt": stub, "entry_point": "f", "tests": tests, "mock_output": v[0]}
             for k, v in cases.items()]
    results = [run_task(t, MockBackend(), "mock", timeout=2) for t in tasks]
    rows = [r for r, _ in results]
    ok = True
    for r in rows:
        expected = cases[r["task_id"]][1]
        flag = "OK " if r["status"] == expected else "FAIL"
        ok &= r["status"] == expected
        print(f"  {flag} {r['task_id']:<14} -> {r['status']:<14} (expected {expected})  {r['error'][:50]}")

    # team dataset format (sanitized MBPP: prompt + test_imports + difficulty)
    team_row = {"task_id": 82, "prompt": "Write a function to find the volume of a sphere.",
                "test_imports": ["import math"],
                "test_list": ["assert math.isclose(volume_sphere(10), 4188.790204786391, rel_tol=0.001)"],
                "difficulty": "Easy", "code": "import math\ndef volume_sphere(r):\n    return 4/3*math.pi*r**3"}
    t = normalize(team_row, "mbpp")
    r, gen = run_task(t, MockBackend(), "mock", timeout=5)
    good = (t["task_id"], r["difficulty"], r["status"], "volume_sphere" in gen["code"]) == ("MBPP/82", "Easy", "passed", True)
    ok &= good
    print(f"  {'OK ' if good else 'FAIL'} team dataset row  -> {r['status']} ({r['difficulty']}, code saved)")

    print("summary:", json.dumps(summarize(rows + [r])))
    print("SELFTEST PASSED" if ok else "SELFTEST FAILED")
    return 0 if ok else 1


# --------------------------------------------------------------------------- #
def _read_rows(csv_path):
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as old:
        for r in csv.DictReader(old):
            try:
                r["solved"] = int(r["solved"])
                for k in ("iterations", "input_tokens", "output_tokens"):
                    r[k] = int(r[k])
                for k in ("gen_time_s", "exec_time_s", "total_time_s"):
                    r[k] = float(r[k])
            except (TypeError, ValueError, KeyError):
                continue  # half-written last line from a killed run
            r.setdefault("difficulty", "")
            rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser(description="Strategy A one-shot baseline")
    ap.add_argument("--backend", choices=["hf", "ollama", "mock"], default="hf")
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    ap.add_argument("--dataset", choices=["humaneval", "mbpp"], default="humaneval")
    ap.add_argument("--data-file", help="local JSONL (e.g. Dataset/mbpp_final_30.jsonl); else the official dataset is downloaded")
    ap.add_argument("--limit", type=int, help="only the first N tasks (pilot)")
    ap.add_argument("--timeout", type=int, default=10, help="seconds allowed to run each solution")
    ap.add_argument("--mem-mb", type=int, default=1024, help="memory limit for each solution run")
    ap.add_argument("--max-new-tokens", type=int, default=256, help="keep identical for all models and strategies")
    ap.add_argument("--quant", choices=["8bit"], help="8-bit bitsandbytes, HF backend only (as in 02_Models_setup.ipynb)")
    ap.add_argument("--no-remote-code", action="store_true",
                    help="use the transformers built-in model class instead of the repo's custom code")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--resume", action="store_true",
                    help="continue an interrupted run: keep finished tasks in the CSV and skip them")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    LIMITS["mem_mb"] = a.mem_mb

    if a.selftest:
        sys.exit(selftest())

    tasks = load_tasks(a.dataset, a.data_file, a.limit)
    if a.backend == "hf":
        backend = HFBackend(a.model, a.max_new_tokens, a.quant, not a.no_remote_code)
    elif a.backend == "ollama":
        backend = OllamaBackend(a.model, a.max_new_tokens)
    else:
        backend = MockBackend()

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    label = a.model + (f"@{a.quant}" if a.quant else "")
    name = f"{label}_{a.dataset}" + (f"_{Path(a.data_file).stem}" if a.data_file else "")
    stem = "oneshot_" + re.sub(r"[^\w.-]", "_", name)
    csv_path, code_path = out / f"{stem}.csv", out / f"{stem}_code.jsonl"
    settings = {"max_new_tokens": a.max_new_tokens, "decoding": "greedy", "timeout_s": a.timeout,
                "mem_limit_mb": a.mem_mb, "quant": a.quant or "none",
                "data": a.data_file or f"official {a.dataset}", "limit": a.limit}

    rows = _read_rows(csv_path) if a.resume and csv_path.exists() else []
    if a.resume:
        print(f"Resuming: {len(rows)} tasks already finished")
    done = {r["task_id"] for r in rows}
    if not a.resume:
        code_path.write_text("", encoding="utf-8")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh, open(code_path, "a", encoding="utf-8") as cf:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
        fh.flush()
        for i, task in enumerate(tasks, 1):
            if task["task_id"] in done:
                continue
            row, gen = run_task(task, backend, label, a.timeout)
            rows.append(row)
            w.writerow(row)
            cf.write(json.dumps(gen, ensure_ascii=False) + "\n")
            fh.flush()
            cf.flush()  # keep partial results if the run crashes
            print(f"[{i}/{len(tasks)}] {row['task_id']} {row['difficulty']}: {row['status']} "
                  f"({row['total_time_s']}s, {row['input_tokens']}+{row['output_tokens']} tok)")
    summary = summarize(rows, settings)
    (out / f"{stem}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
