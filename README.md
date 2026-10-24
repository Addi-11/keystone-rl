# agentic-rl-mismatch

Where does rollout/trainer mismatch hurt in agentic RL? This repo measures how the
gap between the inference engine that generates rollouts (vLLM, possibly low
precision, compressed KV, stale weights) and the trainer is distributed across
**action tokens** (tool calls) versus **reasoning tokens**, and how much each kind
of divergence changes what happens next.

Research plan: see the project doc `agentic-rl-mismatch/research_plan.md`.

## Layout

```
src/agentmismatch/
  labels.py              token span labels (REASONING, TOOL_NAME, TOOL_ARGS, ANSWER, ...)
  spans.py               labels every token of a generated turn
  data.py                problem sets (DAPO-Math-17k, AIME24)
  envs/math_python.py    E1: math with a Python tool, Qwen3 native tool-call format
  tools/python_sandbox.py  subprocess with time/memory limits
scripts/
  pace_setup.sh          one-time conda env on PACE
  smoke_check.py         checks tokenizer + chat template + datasets (run before any GPU job)
tests/                   CPU-only tests with a fake Qwen-like tokenizer
```

## Status

- [x] Step 1: environment, reward, span tagger, tests
- [ ] Step 2: token-in/token-out multi-turn rollouts in vLLM + exact BF16 re-scoring
- [ ] Step 3: per-span mismatch diagnostic (source S0: vLLM-BF16 vs HF-BF16)
- [ ] Step 4: counterfactual forking (consequence of mismatch per span)
- [ ] Step 5: other mismatch sources (INT8, FP8, KV quantisation, staleness, re-tokenisation)

## Quick start (PACE)

```bash
bash scripts/pace_setup.sh
python scripts/smoke_check.py --model Qwen/Qwen3-4B
```
