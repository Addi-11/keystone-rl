"""Step-1 smoke check on PACE (CPU is enough; downloads only the tokenizer + datasets).

    python scripts/smoke_check.py --model Qwen/Qwen3-4B

Checks, in order:
 1. The special tokens we rely on are single tokens in the real tokenizer.
 2. Our hand-written ENV text (tool responses + next assistant header) matches
    Qwen3's chat template byte-for-byte, in thinking and non-thinking mode.
 3. The span tagger on a real tokenization of a tool-call turn and an answer turn.
 4. One sample row from each dataset.
 5. The Python sandbox.
Exit code is non-zero if 1 or 2 fail: those must pass before any rollouts.
"""

import argparse
import json
import sys

from transformers import AutoTokenizer

from agentmismatch.data import load_problems
from agentmismatch.envs.math_python import TOOLS, EnvConfig, initial_messages, observation_text
from agentmismatch.labels import Label
from agentmismatch.spans import SpecialIds, label_turn
from agentmismatch.tools.python_sandbox import run_python


def render(tok, messages, thinking):
    return tok.apply_chat_template(
        messages, tools=TOOLS, tokenize=False, add_generation_prompt=True, enable_thinking=thinking
    )


def check_template(tok, thinking: bool) -> bool:
    problem = "What is 17*23?"
    call = {"name": "python", "arguments": {"code": "print(17*23)"}}
    assistant = {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": call}]}
    if thinking:
        assistant["reasoning_content"] = "I will compute it with code."
    full_msgs = initial_messages(problem) + [assistant, {"role": "tool", "content": "391"}]

    init = render(tok, initial_messages(problem), thinking)
    full = render(tok, full_msgs, thinking)
    if not full.startswith(init):
        print(f"  [thinking={thinking}] FAIL: full conversation does not start with the initial prompt")
        print("  initial tail:", repr(init[-200:]))
        print("  full    head:", repr(full[: len(init) + 50][-250:]))
        return False
    rest = full[len(init):]
    end = rest.find("<|im_end|>") + len("<|im_end|>")
    assistant_turn, env_text = rest[:end], rest[end:]
    ours = observation_text(["391"], EnvConfig(enable_thinking=thinking))
    ok = env_text == ours
    print(f"  [thinking={thinking}] assistant turn as rendered: {assistant_turn!r}")
    print(f"  [thinking={thinking}] ENV text {'MATCHES' if ok else 'DIFFERS'}")
    if not ok:
        print("    template:", repr(env_text))
        print("    ours    :", repr(ours))
    return ok


def show_labels(tok, sp, text):
    ids = tok(text, add_special_tokens=False)["input_ids"]
    labels = label_turn(tok, ids, sp)
    print("   ", " ".join(f"[{tok.decode([i])!r}:{Label(l).name}]" for i, l in zip(ids, labels)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.model)
    ok = True

    print("1. special tokens")
    try:
        sp = SpecialIds.from_tokenizer(tok)
        print("  ", sp)
        for t in ["<think>", "</think>", "<tool_call>", "</tool_call>", "<|im_end|>"]:
            ids = tok(t, add_special_tokens=False)["input_ids"]
            print(f"   {t!r} -> {ids}")
            ok &= len(ids) == 1
    except ValueError as e:
        print("   FAIL:", e)
        return 1

    print("2. chat template vs our ENV text")
    ok &= check_template(tok, thinking=True)
    ok &= check_template(tok, thinking=False)

    print("3. span labels on real tokenization")
    show_labels(tok, sp, '<think>\nI should multiply.\n</think>\n\n<tool_call>\n'
                + json.dumps({"name": "python", "arguments": {"code": "print(17*23)"}})
                + "\n</tool_call><|im_end|>")
    show_labels(tok, sp, "<think>\nIt printed 391.\n</think>\n\nThe answer is \\boxed{391}.<|im_end|>")

    print("4. datasets")
    for name in ["dapo17k", "aime24"]:
        try:
            rows = load_problems(name, n=2)
            print(f"   {name}: {len(rows)} sampled; example:", {k: str(v)[:120] for k, v in rows[0].items()})
        except Exception as e:  # report, don't abort: a schema fix is a one-line change
            print(f"   {name}: FAILED to load ({type(e).__name__}: {e})")

    print("5. sandbox:", run_python("print(sum(range(10)))").to_observation())

    print("\nRESULT:", "PASS" if ok else "FAIL (fix 1-2 before rollouts)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
