"""E1: math problems solved with a Python tool, in Qwen3's native tool-call format.

Episode: system + problem -> assistant turn -> (tool calls executed, responses
appended) -> assistant turn -> ... until a turn has no tool call, or limits hit.
Reward: 1.0 if the last \\boxed{...} in the final turn matches the ground truth
(checked with math-verify), else 0.0.

All template text that we insert ourselves (tool responses, the next
"<|im_start|>assistant" header) is ENV text: the policy never generated it.
scripts/smoke_check.py verifies that our manual strings match Qwen3's chat
template exactly.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from math_verify import parse, verify

from agentmismatch.tools.python_sandbox import run_python

SYSTEM_PROMPT = (
    "You are a careful mathematician. Solve the problem step by step. "
    "You can call the `python` tool to run code; it returns what the code prints. "
    "Use it for any non-trivial calculation. "
    "When you are done, give the final answer as \\boxed{...}."
)

PYTHON_TOOL = {
    "type": "function",
    "function": {
        "name": "python",
        "description": "Execute Python 3 code and return its stdout (or the error). Use print() to show results.",
        "parameters": {
            "type": "object",
            "properties": {"code": {"type": "string", "description": "Python code to execute."}},
            "required": ["code"],
        },
    },
}
TOOLS = [PYTHON_TOOL]

TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


@dataclass
class ToolCall:
    name: str | None
    arguments: dict | None
    error: str | None = None  # set when the call could not be parsed


@dataclass
class EnvConfig:
    max_turns: int = 6               # assistant turns per episode
    max_calls_per_turn: int = 3
    tool_timeout_s: float = 10.0
    max_obs_chars: int = 2000
    enable_thinking: bool = True     # Qwen3 thinking mode


@dataclass
class EpisodeState:
    problem: str
    answer: str
    turns: int = 0
    done: bool = False
    final_text: str = ""
    n_tool_calls: int = 0
    n_tool_errors: int = 0
    extra: dict = field(default_factory=dict)


def initial_messages(problem: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem},
    ]


def parse_tool_calls(turn_text: str) -> list[ToolCall]:
    """Extract every <tool_call>{json}</tool_call> block from one assistant turn."""
    calls = []
    for block in TOOL_CALL_RE.findall(turn_text):
        try:
            obj = json.loads(block.strip())
        except json.JSONDecodeError as e:
            calls.append(ToolCall(None, None, f"Invalid JSON in tool call: {e.msg}"))
            continue
        if not isinstance(obj, dict):
            calls.append(ToolCall(None, None, "Tool call must be a JSON object."))
            continue
        args = obj.get("arguments")
        if isinstance(args, str):  # some models emit arguments as a JSON string
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = None
        if not isinstance(args, dict):
            calls.append(ToolCall(obj.get("name"), None, "Tool call 'arguments' must be an object."))
            continue
        calls.append(ToolCall(obj.get("name"), args))
    return calls


def execute(call: ToolCall, cfg: EnvConfig) -> str:
    if call.error:
        return f"Error: {call.error}"
    if call.name != "python":
        return f"Error: unknown tool '{call.name}'. Available tools: python."
    code = call.arguments.get("code")
    if not isinstance(code, str) or not code.strip():
        return "Error: 'code' must be a non-empty string."
    return run_python(code, timeout_s=cfg.tool_timeout_s).to_observation(cfg.max_obs_chars)


def observation_text(observations: list[str], cfg: EnvConfig) -> str:
    """ENV text appended after an assistant turn that ended with <|im_end|>.

    Mirrors Qwen3's template: consecutive tool messages share one user block,
    each wrapped in <tool_response>. Then the next assistant header, plus the
    empty think block that the template inserts when thinking is disabled.
    """
    body = "\n".join(f"<tool_response>\n{o}\n</tool_response>" for o in observations)
    text = f"\n<|im_start|>user\n{body}<|im_end|>\n<|im_start|>assistant\n"
    if not cfg.enable_thinking:
        text += "<think>\n\n</think>\n\n"
    return text


def step(state: EpisodeState, turn_text: str, cfg: EnvConfig, truncated: bool = False) -> str | None:
    """Advance the episode with one assistant turn.

    Returns the ENV text to append before the next turn, or None if the episode ended.
    `truncated` = the turn hit the token limit without <|im_end|>.
    """
    state.turns += 1
    calls = parse_tool_calls(turn_text)
    if truncated or not calls or state.turns >= cfg.max_turns:
        state.done = True
        state.final_text = turn_text
        return None
    calls = calls[: cfg.max_calls_per_turn]
    observations = []
    for c in calls:
        obs = execute(c, cfg)
        state.n_tool_calls += 1
        state.n_tool_errors += obs.startswith("Error")
        observations.append(obs)
    return observation_text(observations, cfg)


def last_boxed(text: str) -> str | None:
    """Content of the last \\boxed{...}, with balanced braces."""
    start = text.rfind("\\boxed{")
    if start < 0:
        return None
    i = start + len("\\boxed{")
    depth = 1
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i:j]
    return None


def reward(final_text: str, gold: str) -> float:
    pred = last_boxed(final_text)
    if pred is None:
        return 0.0
    try:
        ok = verify(parse(f"\\boxed{{{gold}}}"), parse(f"\\boxed{{{pred}}}"))
    except Exception:
        ok = False
    return 1.0 if ok else 0.0
