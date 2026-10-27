# Strategy A — One-Shot Baseline

## Research project
Experimental Study of the Effectiveness of AI Programming Agents

## Objective
Evaluate code generation using a single LLM response per task,
without feedback, correction, or regeneration (n = 0).

## Dataset
- MBPP initial pilot: 10 tasks
- Difficulty: 4 easy, 3 medium, 3 hard
- Three predefined assertions per task

## Models
1. Qwen2.5-Coder-7B-Instruct
2. DeepSeek-Coder-V2-Lite-Instruct

## Generation configuration
- Maximum new tokens: 256
- Sampling: disabled
- One saved solution per task
- No correction iterations

The original prompts differed slightly:
Qwen received the task description.
DeepSeek additionally received the required function name,
extracted from the task's test assertions.

This difference limits strict prompt-controlled comparisons.

## Verified pilot results

| Metric | Qwen | DeepSeek |
|---|---:|---:|
| Generated solutions | 10 | 10 |
| Syntax-valid solutions | 10 | 9 |
| Total generation time (s) | 11.5842 | 206.2228 |
| Average generation time (s) | 1.1584 | 20.6223 |
| Total output tokens | 485 | 855 |
| Average output tokens | 48.5 | 85.5 |
| Actual functional pass@1 | Pending | Pending |

Generation times reflect the recorded experimental runs.
Output-token counts exclude input tokens.

## Files

- `04_Strategy_A_OneShot_Generation.ipynb`:
  Selected original model-loading and generation implementation.
  Not independently rerun end-to-end.
- `Results/qwen_initial_10_v2_generations.jsonl`:
  Original saved Qwen pilot outputs.
- `Results/deepseek_initial_10_generations.jsonl`:
  Original saved DeepSeek pilot outputs.
- `Results/mbpp_initial_10_evaluation_manifest.jsonl`:
  Test manifest containing 20 model-task records.
- `Results/mbpp_evaluation_controller.py`:
  Functional evaluation controller.
- `Results/mbpp_evaluation_worker.py`:
  Functional evaluation worker.
- `Results/strategy_A_metrics.py`:
  Calculates pass@1 from actual evaluation results.
- `Results/strategy_A_initial_10_summary.json`:
  Verified generation metrics and pending evaluation status.

## Functional evaluation status

The controller and worker have been prepared, and the
pass@1 calculation was validated using synthetic data.

Actual generated solutions have not yet undergone
functional test execution. No actual pass@1 score is claimed.

The evaluation scripts execute model-generated Python code.
A subprocess timeout is not a hardened security sandbox.
Execute only in an approved isolated environment.

## Reproducibility notes

The generation notebook preserves selected original code,
but depends on the original model files and KTU environment.
Its saved output paths refer to the original experiment.

The original 20 solutions must remain unchanged when
performing the first functional evaluation.

## Scope

This submission contains only Mahesh's Strategy A pilot.
It does not incorporate Ilkin's implementation or results.
