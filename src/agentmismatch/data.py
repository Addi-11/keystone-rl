"""Problem sets for E1. Every loader returns a list of {"id", "problem", "answer"}.

Hub schemas change, so each loader validates columns and fails loudly.
scripts/smoke_check.py prints a sample row from each on PACE.
"""

from __future__ import annotations

import random

DATASETS = {
    # name: (hub id, split, problem column, answer column)
    "dapo17k": ("open-r1/DAPO-Math-17k-Processed", "train", "prompt", "solution"),
    "aime24": ("HuggingFaceH4/aime_2024", "train", "problem", "answer"),
}


def normalize_rows(rows, problem_col: str, answer_col: str, prefix: str) -> list[dict]:
    out, seen = [], set()
    for i, r in enumerate(rows):
        if problem_col not in r or answer_col not in r:
            raise KeyError(f"expected columns {problem_col!r}, {answer_col!r}; got {sorted(r)}")
        problem = str(r[problem_col]).strip()
        answer = str(r[answer_col]).strip()
        if not problem or not answer or problem in seen:  # DAPO-17k has many duplicates
            continue
        seen.add(problem)
        out.append({"id": f"{prefix}-{i}", "problem": problem, "answer": answer})
    return out


def load_problems(name: str, n: int | None = None, seed: int = 0, config: str | None = None) -> list[dict]:
    from datasets import load_dataset

    hub_id, split, pcol, acol = DATASETS[name]
    ds = load_dataset(hub_id, config, split=split) if config else load_dataset(hub_id, split=split)
    rows = normalize_rows(ds, pcol, acol, prefix=name)
    if n is not None and n < len(rows):
        rows = random.Random(seed).sample(rows, n)
    return rows
