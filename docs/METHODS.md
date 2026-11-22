# How Strategy A and B work

This file explains how we run Strategy A (one try) and Strategy B (the model fixes its code using feedback),
in enough detail that someone else could repeat it and get the same results. The settings of every run are also
saved in its JSON file under `settings`.

## 1. Tasks

We use MBPP (the sanitized version). The tasks we picked are in `Dataset/`:

* `mbpp_final_30.jsonl`: 30 tasks, 10 Easy, 10 Medium, 10 Hard
* `mbpp_initial_10.jsonl`: 10 tasks we used for the first tests. All 10 are also in `final_30`, so the two files
  are never added together; the results we report are the `final_30` ones.

How we chose the tasks and their difficulty is in `Dataset/mbpp_difficulty_log.csv` and `Dataset/README.md`.

Every task has a short description and 3 tests (only MBPP/58 has 4). The first test goes into the prompt, so the
model knows the function name and what the input and output look like. We call it the visible test. The other
tests are hidden: the model never sees them, they are never used for feedback, and the generated code can't read
them from disk (section 5). A task counts as solved only if the code passes all of its tests.

## 2. Models and settings

| | Qwen | DeepSeek |
|---|---|---|
| Model | `Qwen/Qwen2.5-Coder-7B-Instruct` | `deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct` |
| Precision | bfloat16, no quantization | bfloat16, no quantization |
| Model code | from the model repository | built into transformers (`--no-remote-code`) |
| Python | base environment (transformers 5.x) | `~/venv-glm` (transformers 4.57.6) |

We run on the KTU JupyterHub server, one A100 40 GB GPU per run. Both models fit on one GPU.

Generation is greedy (`do_sample=False`) with `max_new_tokens = 512`, the same for both models and both strategies.
With greedy decoding the same prompt always gives the same answer, so the runs can be repeated.
In the final runs none of the 179 answers reached the 512-token limit, so no answer was cut off.

We use each model's own chat template. The exact text the model got is saved for every call (`prompt_rendered` in
`_code.jsonl`). Library versions, GPU, CUDA, dtype and model revision are saved in the JSON (`settings.environment`).

## 3. Prompts

The prompts are the same for both models.

System message (A and B):

    You are an expert Python programmer. Reply with the complete Python code in a single ```python code block and nothing else.

First try (A, and attempt 1 of B, so B's first attempt is exactly Strategy A):

    <task description>
    Your code must pass this test:
    <visible test>

Repair (only B). The first-try message, followed by:

    Your previous solution:
    ```python
    <code of the previous attempt>
    ```
    We ran it and got this error:
    <feedback>

    Fix the code. Reply with the complete corrected Python code in a single ```python code block.

Every repair starts a new conversation. We don't keep the whole chat history, only the last code and its error.

## 4. Getting the code out of the answer

The model answers with text and code blocks, sometimes more than one block. `extract_code_info`
(`src/baseline_oneshot.py`) picks the solution the same way in A and B:

1. We only look at code blocks marked `python` or `py`, or not marked at all. Other blocks, like `bash`, are
   skipped. If the answer was cut at the token limit, the open block runs to the end.
2. From those, we keep the blocks that define the function used in the visible test (for example `max_product`).
   If none of them does, we take blocks with any `def`, then any block, and if there is no block at all, the whole
   answer.
3. If one block is left, or the same block repeated, we use it. If there are two different versions, we use the
   last one, because the model usually writes the old code first and then "here is the corrected code". These
   answers are marked as ambiguous and counted (`ambiguous_extractions`).

For every answer we save which of these cases happened (`extraction` in `_code.jsonl`).

`src/verify_extraction.py` checks this on all saved answers. In the final runs there were 179 answers: 173 with one
block, 2 with the same block repeated, and 4 ambiguous ones (all DeepSeek, MBPP/468). In all 4 the model wrote
"Here is the corrected code" before the last block. We ran every version of those answers, and the choice changes
neither the feedback nor the final result. The code used in the runs is the code the rule picks in 179 of 179
answers.

After that, `remove_model_tests` cleans the code. It keeps imports, functions, classes and constants, and removes the
model's own test code (asserts, prints, example calls, `if __name__ == "__main__"`). Otherwise the model's own tests,
which are sometimes wrong, could fail the run or hide the real error. If the code doesn't parse we leave it as it is,
so the syntax error still shows up.

## 5. Running the generated code

Generated code never runs in the notebook or in our main process. `run_code` starts a new Python process for every
run, with three layers of protection:

* `unshare` (Linux namespaces, the same feature Docker uses, works without root): no network and a separate process
  tree. The script uses the strongest level the machine allows and saves it in the results. On the KTU server this
  is `net+pid`.
* A Python audit hook (PEP 578) that the generated code can't remove. The code can only write, delete or rename
  inside its own temp folder. It can't read or list the project folder (dataset with the hidden tests, results,
  source code) or private config (`~/.ssh`, `~/.cache`, `~/.config`, saved credentials). It can't start processes,
  open sockets or load `ctypes`.
* Limits: 10 seconds (then the whole process group is killed), CPU time, 1024 MB memory, 10 MB file size.

The program that runs is the cleaned code plus the tests. It is written to the temp folder, which is deleted
afterwards.

The professor said in the supervisor meeting that this is enough for the project, since there is no internet and no
access to the hidden tests. We also ran the final evaluation of all saved solutions again inside the sandbox
(`src/reevaluate_sandbox.py`) and got the same result for 160 of 160. The self-tests (`--selftest`) try things that
must be blocked: network access, writing and deleting files, shell commands, ctypes and reading the dataset.

A run ends as one of: `passed`, `test_failure` (AssertionError), `syntax_error` (SyntaxError, IndentationError,
TabError), `runtime_error` (any other error, MemoryError included), `timeout`, or `no_code` (empty answer).

## 6. Strategy A

One LLM call, then extraction, cleaning and the final evaluation on all tests. No feedback.

## 7. Strategy B

A normal Python loop does the repairs. There is no second model and no agent.

1. Attempt 1 uses the same prompt as A.
2. Compile check (`run_compiler`): Python's `compile()`, the same as `py_compile`. Nothing runs yet. On a syntax
   error, the feedback is the error with its line number, for example `SyntaxError: expected ':' (line 1)`.
3. Test check (`run_tester`): the code runs with the visible test only, in the sandbox. A test like
   `assert f(x) == y` is rewritten so the feedback shows what was expected and what came back, for example
   `AssertionError: expected 3000, got 45000`. Long values are cut after 200 characters. Other errors give the
   error message, and a timeout gives `Timeout: the code did not finish in 10 seconds`.
4. If step 2 or 3 failed, the repair prompt from section 3 goes to the model, and the new code goes back to step 2.

Budget: one first try and up to 5 repairs, so at most 6 LLM calls per task (`--max-repairs 5`). We don't use the
word "iteration" because it was confusing. We report LLM calls and repair calls instead.

The loop stops in one of three cases, checked in this order after every LLM call:

| stop_reason | When |
|---|---|
| `dev_test_passed` | the code passes the visible test |
| `max_repairs_reached` | the 5th repair was used |
| `repeated_code` | the repair is exactly the same code, with the same error, as the attempt before. With greedy decoding the next answer would be the same too, so stopping changes nothing and only saves time. |

At the end, the last code is run on all tests, the same way as in A. Passing the visible test doesn't mean passing
the hidden ones. We count those cases as `dev_passed_but_hidden_failed`.

To compare with A, B also runs attempt 1 on all tests (`first_status`). This run is only for checking: it isn't
timed and the model never sees it.

## 8. What we measure

Per task (CSV). A and B have the same columns, and B has three more:

| Column | What it is |
|---|---|
| `solved`, `status`, `error` | result of the final evaluation |
| `llm_calls`, `repair_calls` | number of LLM calls (A: 1 and 0) |
| `input_tokens`, `output_tokens` | over all calls; input is the full prompt after the chat template, output is what the model generated |
| `initial_*`, `repair_*` tokens | the same, split into the first call and the repairs |
| `gen_time_s`, `initial_gen_time_s`, `repair_gen_time_s` | time of the LLM calls (`time.perf_counter`, includes tokenizing and decoding) |
| `eval_exec_time_s` | time of the final evaluation run (same in A and B) |
| `feedback_exec_time_s` | B's compile and test runs for the feedback (0 in A) |
| `exec_time_s`, `total_time_s` | eval + feedback, and generation + execution |
| `ambiguous_extractions` | answers where two different versions were found |
| B only: `first_status`, `dev_passed`, `stop_reason` | attempt 1 on all tests, whether the visible test passed at the end, why the loop stopped |

Execution times include starting Python and the sandbox. The server is shared, so times can only be compared within
the same setup. Tokens and LLM calls don't depend on the hardware, so they are the main cost numbers.

The JSON summary of each run has the solve rate (also per difficulty), totals and averages of the columns above and
the status counts. B also has:

* `first_attempt_solve_rate`, which should be the same as A's solve rate
* `repair_summary`: `initially_failed`, `repaired`, `still_failed`, `broken_by_repair` and `repair_success_rate`
  (repaired divided by initially failed), also split by difficulty and by the first error
* `stop_reason_counts` and `dev_passed_but_hidden_failed`

We report the solve rate (share of tasks solved) and not pass@k, because with greedy decoding there is only one
answer per task.

## 9. Output files

Each run writes three files to `results/`:

* `<name>.csv`: one row per task (section 8)
* `<name>.json`: the summary, all settings and the environment
* `<name>_code.jsonl`: for every LLM call, the prompt, the raw answer, the extraction case, the extracted and cleaned
  code and whether it hit the token limit; for B also the compile and test result, the feedback and the times

`<name>` is `oneshot_<model>_mbpp_<data>` for A and `feedback_r5_<model>_mbpp_<data>` for B (r5 means 5 repairs).

`src/analyze_failures.py` writes the failure report and a CSV of the failed tasks to `results/analysis/`. It puts
each task still wrong after B into one of three groups: `feedback_ignored` (the model kept returning the same
program; comments and formatting don't count as a change), `changed_but_still_wrong`, and
`passes_visible_fails_hidden`. We also read every failure by hand and wrote down the reason in
`manual_notes_<data>.csv` (`task_spec_mismatch`, `wrong_algorithm` or `wrong_edge_case`). The script adds these
notes to the report.

## 10. How to run everything again

    cd ~/Team-Horizon
    python src/baseline_oneshot.py --selftest          # A, sandbox and extraction
    python src/strategy_b.py --selftest                # B loop, stop rules, no hidden tests leaked
    nohup bash src/run_all.sh > run_all.log 2>&1 &     # A and B, both models, both data files
    python src/verify_extraction.py                    # check the extraction on all answers
    python src/reevaluate_sandbox.py                   # final evaluation again in the sandbox
    python src/analyze_failures.py                     # failure report

On 22 Nov we ran all A and B experiments again with the final code. Every task in all 8 runs got the same result as
before. Going from 1 + 4 to 1 + 5 repairs changed nothing: only one task (DeepSeek, MBPP/103) used the extra call,
and the model returned the same code again.

## 11. Limitations

* 30 tasks is only a pilot. A difference of a few tasks doesn't mean much statistically. For a journal paper the
  professor wants at least 30 tasks per difficulty level and a statistical test.
* MBPP has only 3 tests per task, so a wrong solution can sometimes pass. MBPP+, which has more hidden tests, would
  be stricter.
* We set the difficulty labels ourselves (`Dataset/mbpp_difficulty_log.csv`).
* Some MBPP tasks have tests that don't match their description. In `mbpp_final_30` this happens twice. In MBPP/468
  the text asks for a subsequence, but all tests expect a contiguous run. In MBPP/407 a hidden test expects `False`
  when there is no bigger number, which the text doesn't mention. Following the text, these two can't be solved. We
  kept them so the results stay comparable with MBPP, and they are marked in the failure report.
* Execution times depend on the server load, so tokens and LLM calls are the main cost numbers.
* The two models run with different transformers versions (DeepSeek needs 4.57.6). Both use greedy decoding and the
  same prompts and settings, but the library version is not identical.
