"""Label every token of one assistant turn as reasoning / action / answer / format.

Works directly on the generated token ids, so labels line up 1:1 with the
log-probs we compare. Two passes:

1. Character offsets for each token, via incremental decoding (byte-level BPE
   can split a UTF-8 character across tokens; those tokens get an empty span
   and the token that completes the character gets the whole character).
2. A state machine over the special tokens <think>, </think>, <tool_call>,
   </tool_call>, plus character ranges for the tool-call "name"/"arguments"
   values and the final \\boxed{...}.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol, Sequence

from agentmismatch.labels import Label


class TokenizerLike(Protocol):
    def decode(self, ids: Sequence[int], skip_special_tokens: bool = ...) -> str: ...
    def convert_tokens_to_ids(self, token: str) -> int: ...


@dataclass(frozen=True)
class SpecialIds:
    think_start: int
    think_end: int
    tool_start: int
    tool_end: int
    other: frozenset[int]  # e.g. <|im_end|>, <|endoftext|>

    @classmethod
    def from_tokenizer(cls, tok: TokenizerLike) -> "SpecialIds":
        def one(t: str) -> int:
            i = tok.convert_tokens_to_ids(t)
            if i is None or (hasattr(tok, "unk_token_id") and i == tok.unk_token_id):
                raise ValueError(f"{t!r} is not a single token in this tokenizer")
            return i

        other = {one("<|im_end|>")}
        if getattr(tok, "eos_token_id", None) is not None:
            other.add(tok.eos_token_id)
        return cls(one("<think>"), one("</think>"), one("<tool_call>"), one("</tool_call>"), frozenset(other))


# ---------------------------------------------------------------- char offsets

def char_spans(tok: TokenizerLike, ids: Sequence[int], window: int = 8) -> list[tuple[int, int]]:
    """(start, end) character offsets of each token in tok.decode(ids)."""
    spans: list[tuple[int, int]] = []
    pos = 0
    pending_from = None  # first index of tokens whose text is still incomplete
    for i in range(len(ids)):
        a = max(0, (pending_from if pending_from is not None else i) - window)
        prefix = tok.decode(ids[a:pending_from if pending_from is not None else i], skip_special_tokens=False)
        full = tok.decode(ids[a : i + 1], skip_special_tokens=False)
        piece = full[len(prefix):]
        if piece.endswith("�"):  # incomplete UTF-8 sequence, wait for more bytes
            if pending_from is None:
                pending_from = i
            spans.append((pos, pos))
            continue
        spans.append((pos, pos + len(piece)))
        pos += len(piece)
        pending_from = None
    full_text = tok.decode(ids, skip_special_tokens=False)
    if pos != len(full_text):  # windowed decode disagreed; fall back to exact O(n^2)
        return _char_spans_exact(tok, ids)
    return spans


def _char_spans_exact(tok: TokenizerLike, ids: Sequence[int]) -> list[tuple[int, int]]:
    spans, prev = [], 0
    for i in range(len(ids)):
        cur = len(tok.decode(ids[: i + 1], skip_special_tokens=False))
        cur = max(cur, prev)
        spans.append((prev, cur))
        prev = cur
    return spans


# ---------------------------------------------------------------- JSON ranges

_NAME_RE = re.compile(r'"name"\s*:\s*"((?:[^"\\]|\\.)*)"')
_ARGS_KEY_RE = re.compile(r'"arguments"\s*:\s*')


def _json_value_end(s: str, i: int) -> int:
    """Index just past the JSON value starting at s[i] (object, array, string or scalar)."""
    if i >= len(s):
        return i
    if s[i] == '"':
        j = i + 1
        while j < len(s):
            if s[j] == "\\":
                j += 2
                continue
            if s[j] == '"':
                return j + 1
            j += 1
        return len(s)
    if s[i] in "{[":
        depth, j, in_str = 0, i, False
        while j < len(s):
            c = s[j]
            if in_str:
                if c == "\\":
                    j += 2
                    continue
                if c == '"':
                    in_str = False
            elif c == '"':
                in_str = True
            elif c in "{[":
                depth += 1
            elif c in "}]":
                depth -= 1
                if depth == 0:
                    return j + 1
            j += 1
        return len(s)
    m = re.match(r"[^,}\]\s]*", s[i:])
    return i + (m.end() if m else 0)


def tool_call_ranges(text: str, offset: int = 0) -> dict[str, tuple[int, int]]:
    """Character ranges (absolute, using `offset`) of the name and arguments values."""
    out = {}
    m = _NAME_RE.search(text)
    if m:
        out["name"] = (offset + m.start(1), offset + m.end(1))
    m = _ARGS_KEY_RE.search(text)
    if m:
        out["args"] = (offset + m.end(), offset + _json_value_end(text, m.end()))
    return out


def boxed_range(text: str) -> tuple[int, int] | None:
    """Range covering the last \\boxed{...} including the command and braces."""
    start = text.rfind("\\boxed{")
    if start < 0:
        return None
    depth, j = 0, start + len("\\boxed")
    while j < len(text):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return (start, j + 1)
        j += 1
    return (start, len(text))


def _overlaps(span: tuple[int, int], rng: tuple[int, int] | None) -> bool:
    if rng is None:
        return False
    a, b = span
    if a == b:  # zero-width (partial UTF-8) token: treat as inside if it sits within the range
        return rng[0] <= a < rng[1]
    return a < rng[1] and b > rng[0]


# ---------------------------------------------------------------- tagger

def label_turn(tok: TokenizerLike, ids: Sequence[int], sp: SpecialIds) -> list[Label]:
    """Labels for the tokens of ONE generated assistant turn."""
    ids = list(ids)
    spans = char_spans(tok, ids)
    text = tok.decode(ids, skip_special_tokens=False)

    # Pass 1: region of each token from the special-token state machine.
    region = []  # "think" | "tool" | "out" | "special"
    in_think = in_tool = False
    tool_blocks: list[tuple[int, int]] = []  # char ranges of tool-call bodies
    cur_tool_start = None
    for i, t in enumerate(ids):
        if t == sp.think_start:
            in_think = True
            region.append("special")
        elif t == sp.think_end:
            in_think = False
            region.append("special")
        elif t == sp.tool_start and not in_think:
            in_tool = True
            cur_tool_start = spans[i][1]
            region.append("special")
        elif t == sp.tool_end and in_tool:
            in_tool = False
            tool_blocks.append((cur_tool_start, spans[i][0]))
            region.append("special")
        elif t in sp.other:
            region.append("special")
        else:
            region.append("think" if in_think else "tool" if in_tool else "out")
    if in_tool and cur_tool_start is not None:  # truncated inside a tool call
        tool_blocks.append((cur_tool_start, len(text)))

    # Character ranges inside tool calls, and the final answer outside them.
    name_rngs, args_rngs = [], []
    for a, b in tool_blocks:
        r = tool_call_ranges(text[a:b], offset=a)
        if "name" in r:
            name_rngs.append(r["name"])
        if "args" in r:
            args_rngs.append(r["args"])
    out_text_chars = [c if region_at else " " for c, region_at in _out_mask(text, spans, region)]
    box = boxed_range("".join(out_text_chars))

    # Pass 2: final labels.
    labels = []
    for i, reg in enumerate(region):
        s = spans[i]
        if reg == "special":
            labels.append(Label.FORMAT)
        elif reg == "think":
            labels.append(Label.REASONING)
        elif reg == "tool":
            if any(_overlaps(s, r) for r in name_rngs):
                labels.append(Label.TOOL_NAME)
            elif any(_overlaps(s, r) for r in args_rngs):
                labels.append(Label.TOOL_ARGS)
            else:
                labels.append(Label.ACTION_OTHER)
        else:
            labels.append(Label.ANSWER if _overlaps(s, box) else Label.REASONING)
    return labels


def _out_mask(text: str, spans, region):
    """Yield (char, is_outside_think_and_tool) for every character of text."""
    inside = [False] * len(text)
    for (a, b), reg in zip(spans, region):
        if reg == "out":
            for k in range(a, b):
                inside[k] = True
    return zip(text, inside)
