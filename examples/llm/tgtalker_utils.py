"""Prompt construction and scoring helpers for the TGTalker example.

This module is intentionally free of heavy/optional dependencies (``pydantic``,
``outlines``, ``transformers``) so that the prompt-building logic can be unit
tested without an LLM runtime installed. The structured-output schemas live in
``schemas.py`` and the inference entrypoints live in ``TGTalker.py`` /
``multihop.py``.
"""

from __future__ import annotations

import collections
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch

from tgm.constants import PADDED_NODE_ID

Row = Tuple[int, int, int]


def _as_int(value: object, name: str) -> int:
    """Coerce a scalar (python ``int``, ``numpy`` integer, or 0/1-element
    tensor/array) into a python ``int``.

    The original TGTalker code used ``isinstance(src, int)`` which rejects the
    ``numpy.int64`` / ``torch`` scalars that the data pipeline actually yields,
    causing every real call to raise. We accept integer scalars from any of
    those libraries and raise ``ValueError`` for genuinely non-scalar input.
    """
    if isinstance(value, bool):
        raise ValueError(f'{name} must be an integer, got bool')
    if isinstance(value, (int, np.integer)):
        return int(value)
    if hasattr(value, 'item'):
        try:
            return int(value.item())  # type: ignore[union-attr]
        except Exception as e:  # multi-element tensor/array, etc.
            raise ValueError(
                f'{name} must be an integer scalar, got {type(value).__name__} ({e})'
            )
    raise ValueError(f'{name} must be an integer scalar, got {type(value).__name__}')


def row2text(rows: Iterable[Sequence[int]]) -> str:
    """Serialize an iterable of ``(src, dst, ts)`` rows into text.

    One tuple per line, e.g. ``(1, 42, 1200)``. The original concatenated tuples
    with no separators; we add a newline per row for readability (see the
    "Divergences" section of the README).
    """
    out = ''
    for row in rows:
        out += f'({int(row[0])}, {int(row[1])}, {int(row[2])})\n'
    return out


def make_user_prompt(
    src: object,
    ts: object,
    nbr_nids: object | None = None,
    nbr_times: object | None = None,
) -> str:
    """Build the user prompt for a single ``(src, ts)`` query.

    If recent neighbors of ``src`` are provided they are listed as
    ``(src, dst, ts)`` tuples (padding entries equal to ``PADDED_NODE_ID`` are
    skipped). Otherwise a simpler neighbor-free prompt is returned.
    """
    if src is None or ts is None:
        raise ValueError('Source node and timestamp must be provided')

    src_i = _as_int(src, 'Source node')
    ts_i = _as_int(ts, 'Timestamp')

    if nbr_nids is not None and len(nbr_nids) > 0:  # type: ignore[arg-type]
        if nbr_times is None:
            raise ValueError('nbr_times must be provided when nbr_nids is given')
        user_prompt = f'`Source Node` {src_i} has the following past interactions:\n'
        for nid, t in zip(nbr_nids, nbr_times):  # type: ignore[arg-type]
            nid_i = int(nid)
            if nid_i == PADDED_NODE_ID:
                continue
            user_prompt += f'({src_i}, {nid_i}, {int(t)})\n'
        user_prompt += (
            f'Please predict the most likely `Destination Node` for '
            f'`Source Node` {src_i} at `Timestamp` {ts_i}.'
        )
    else:
        user_prompt = (
            f'Predict the next interaction for `Source Node` {src_i} '
            f'at `Timestamp` {ts_i}.'
        )
    return user_prompt


def gather_hop_edges(
    i: int,
    hops: int,
    k: int,
    seed_nids: Sequence,
    nbr_nids: Sequence,
    nbr_times: Sequence,
) -> List[List[Row]]:
    """Reconstruct the per-hop edge lists for source ``i`` from the recency
    hook's flattened per-hop tensors.

    For hop ``h``, source ``i`` owns rows ``[i * k**h, (i + 1) * k**h)`` of the
    flattened hop-``h`` tensors (``seed_nids[h]``, ``nbr_nids[h]``,
    ``nbr_times[h]``). Edges expanded from padded seeds or padded neighbors are
    skipped. Returns ``hop_edges`` where ``hop_edges[h]`` is a list of
    ``(seed, neighbor, time)`` tuples.
    """
    hop_edges: List[List[Row]] = []
    for h in range(hops):
        block = k**h
        start, end = i * block, (i + 1) * block
        seed_slice = seed_nids[h][start:end]
        nbr_slice = nbr_nids[h][start:end]
        time_slice = nbr_times[h][start:end]
        edges: List[Row] = []
        for r in range(len(seed_slice)):
            s = int(seed_slice[r])
            if s == PADDED_NODE_ID:
                continue
            for m in range(nbr_slice.shape[1]):
                d = int(nbr_slice[r][m])
                if d == PADDED_NODE_ID:
                    continue
                edges.append((s, d, int(time_slice[r][m])))
        hop_edges.append(edges)
    return hop_edges


def make_multihop_user_prompt(
    src: object,
    ts: object,
    hop_edges: Sequence[Sequence[Row]],
) -> str:
    """Build a user prompt that includes multi-hop neighborhood context.

    Args:
        src: query source node.
        ts: query timestamp.
        hop_edges: ``hop_edges[h]`` is the list of ``(a, b, t)`` edges gathered
            at hop ``h`` (hop 0 = the source's own recent interactions, hop 1 =
            interactions of those neighbors, ...).
    """
    src_i = _as_int(src, 'Source node')
    ts_i = _as_int(ts, 'Timestamp')

    has_any = any(len(edges) > 0 for edges in hop_edges)
    if not has_any:
        return (
            f'Predict the next interaction for `Source Node` {src_i} '
            f'at `Timestamp` {ts_i}.'
        )

    user_prompt = (
        f'`Source Node` {src_i} is connected to the following multi-hop '
        f'temporal neighborhood:\n'
    )
    for hop, edges in enumerate(hop_edges):
        if not edges:
            continue
        user_prompt += f'{hop + 1}-hop interactions:\n'
        for a, b, t in edges:
            user_prompt += f'({int(a)}, {int(b)}, {int(t)})\n'
    user_prompt += (
        f'Please predict the most likely `Destination Node` for '
        f'`Source Node` {src_i} at `Timestamp` {ts_i}.'
    )
    return user_prompt


def make_system_prompt(
    background_rows: Sequence[Row] | None = None,
    demos: Sequence[Tuple[str, str]] | None = None,
    use_cot: bool = False,
) -> str:
    """Build the system prompt.

    Args:
        background_rows: Recent global ``(src, dst, ts)`` edges shown as the
            shared "TEMPORAL GRAPH" context (the ``--bg-size`` window).
        demos: In-context-learning ``(instruction, answer)`` text pairs
            (the ``--icl`` demonstrations).
        use_cot: If True, instruct the model to reason step by step.
    """
    system_prompt = (
        'You are an expert temporal graph learning agent. Your task is to '
        'predict the next interaction (i.e. `Destination Node`) given the '
        '`Source Node` and `Timestamp`.\n\n'
        'Description of the temporal graph is provided below, where each line '
        'is a tuple of (`Source Node`, `Destination Node`, `Timestamp`).\n\n'
        'TEMPORAL GRAPH:\n'
    )

    if background_rows:
        system_prompt += row2text(background_rows)

    if use_cot:
        system_prompt += "\nLet's think step by step about the problem.\n"

    if demos:
        system_prompt += '\nHere are some examples:\n'
        for instruction, answer in demos:
            system_prompt += instruction + answer

    return system_prompt


def predict_link(query_dst: torch.Tensor, llm_dst: int) -> torch.Tensor:
    """Score candidate destinations against the LLM's single prediction.

    Returns a 0/1 vector aligned with ``query_dst`` (1.0 where a candidate
    equals the predicted destination). With ``query_dst = [true_dst, *negs]``
    this yields MRR 1.0 when the LLM picks the true destination and 0.0 when it
    picks a negative (or a node outside the candidate set).
    """
    return (query_dst == llm_dst).float()


class BackgroundBuffer:
    """Sliding window of the most recent global ``(src, dst, ts)`` edges.

    Replaces the original's ``background_rows`` numpy slicing. Seed it from the
    tail of the validation edges, then ``extend`` it with each test batch *after*
    that batch's prompts are built so background context never leaks the edge
    currently being predicted.
    """

    def __init__(self, max_size: int) -> None:
        self.max_size = max_size
        self._buf: collections.deque[Row] | None = (
            collections.deque(maxlen=max_size) if max_size > 0 else None
        )

    def extend(
        self,
        src: Iterable[int],
        dst: Iterable[int],
        ts: Iterable[int],
    ) -> None:
        if self._buf is None:
            return
        for s, d, t in zip(src, dst, ts):
            self._buf.append((int(s), int(d), int(t)))

    def rows(self) -> List[Row]:
        return list(self._buf) if self._buf is not None else []


class ICLWindow:
    """Sliding window of the most recent edges, rendered as ICL demos.

    Each demonstration is an ``(instruction, answer)`` pair where the
    instruction reuses ``make_user_prompt`` (neighbor-free form) and the answer
    is the JSON the model is asked to emit. Updated the same way as
    ``BackgroundBuffer`` to avoid leakage.
    """

    def __init__(self, max_size: int) -> None:
        self.max_size = max_size
        self._buf: collections.deque[Row] | None = (
            collections.deque(maxlen=max_size) if max_size > 0 else None
        )

    def extend(
        self,
        src: Iterable[int],
        dst: Iterable[int],
        ts: Iterable[int],
    ) -> None:
        if self._buf is None:
            return
        for s, d, t in zip(src, dst, ts):
            self._buf.append((int(s), int(d), int(t)))

    def demos(self) -> List[Tuple[str, str]]:
        if self._buf is None:
            return []
        out: List[Tuple[str, str]] = []
        for s, d, t in self._buf:
            instruction = make_user_prompt(s, t)
            answer = f' Answer: {{"destination_node": {d}}}\n'
            out.append((instruction, answer))
        return out
