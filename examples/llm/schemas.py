"""Pydantic schemas for structured LLM output (via ``outlines``).

Kept separate from ``tgtalker_utils`` so the prompt helpers stay importable
without ``pydantic`` installed (it ships with the ``llm`` dependency extra).
"""

from __future__ import annotations

from typing import List

from pydantic import BaseModel


class Step(BaseModel):
    """A single chain-of-thought reasoning step."""

    explanation: str
    output: str


class TGAnswer(BaseModel):
    """Base (non-reasoning) answer schema."""

    destination_node: int


class TGReasoning(BaseModel):
    """Chain-of-thought answer schema: reasoning steps then the prediction."""

    steps: List[Step]
    destination_node: int
