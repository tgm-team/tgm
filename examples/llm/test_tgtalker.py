import numpy as np
import pytest
import torch
from tgtalker_utils import (
    BackgroundBuffer,
    ICLWindow,
    gather_hop_edges,
    make_multihop_user_prompt,
    make_system_prompt,
    make_user_prompt,
    predict_link,
    row2text,
)

from tgm.constants import PADDED_NODE_ID


def test_make_system_prompt_base():
    system_prompt = make_system_prompt()
    assert system_prompt is not None and len(system_prompt) > 0
    assert 'TEMPORAL GRAPH:' in system_prompt


def test_make_system_prompt_with_background():
    rows = [(1, 2, 100), (3, 4, 101)]
    system_prompt = make_system_prompt(background_rows=rows)
    assert '(1, 2, 100)' in system_prompt
    assert '(3, 4, 101)' in system_prompt


def test_make_system_prompt_cot_and_demos():
    demos = [('instruction text ', 'answer text\n')]
    system_prompt = make_system_prompt(demos=demos, use_cot=True)
    assert 'step by step' in system_prompt
    assert 'instruction text' in system_prompt
    assert 'answer text' in system_prompt


def test_make_user_prompt_basic():
    user_prompt = make_user_prompt(1, 200)
    assert user_prompt is not None and len(user_prompt) > 0
    assert '1' in user_prompt and '200' in user_prompt


def test_make_user_prompt_accepts_numpy_and_torch_scalars():
    # The real data pipeline yields numpy / torch integer scalars, not python ints.
    assert make_user_prompt(np.int64(1), np.int64(200))
    assert make_user_prompt(torch.tensor(1), torch.tensor(200))


def test_make_user_prompt_with_neighbors_filters_padding():
    nbr_nids = np.array([5, PADDED_NODE_ID, 7])
    nbr_times = np.array([10, 11, 12])
    user_prompt = make_user_prompt(np.int64(1), np.int64(200), nbr_nids, nbr_times)
    assert '(1, 5, 10)' in user_prompt
    assert '(1, 7, 12)' in user_prompt
    # Padded neighbor must be skipped.
    assert f'{PADDED_NODE_ID}' not in user_prompt


def test_bad_arg_user_prompt():
    src = torch.IntTensor([0, 1, 2, 3])
    ts = torch.IntTensor([5, 5])
    with pytest.raises(ValueError):
        make_user_prompt(src, ts)

    with pytest.raises(ValueError):
        make_user_prompt(None, None)


def test_row2text():
    text = row2text([(1, 2, 3), (4, 5, 6)])
    assert text == '(1, 2, 3)\n(4, 5, 6)\n'


def test_predict_link():
    query_dst = torch.tensor([10, 20, 30])
    # LLM predicts the (positive) true destination at index 0.
    y_pred = predict_link(query_dst, 10)
    assert y_pred[0].item() == 1.0
    assert y_pred[1:].sum().item() == 0.0
    # LLM predicts a node outside the candidate set -> all zeros.
    y_pred = predict_link(query_dst, 999)
    assert y_pred.sum().item() == 0.0


def test_background_buffer_sliding_window():
    buf = BackgroundBuffer(max_size=2)
    buf.extend([1, 2], [10, 20], [100, 101])
    buf.extend([3], [30], [102])
    rows = buf.rows()
    assert len(rows) == 2
    # Oldest (1, 10, 100) evicted; newest two retained.
    assert rows == [(2, 20, 101), (3, 30, 102)]


def test_background_buffer_disabled():
    buf = BackgroundBuffer(max_size=0)
    buf.extend([1], [2], [3])
    assert buf.rows() == []


def test_icl_window_demos():
    window = ICLWindow(max_size=2)
    window.extend([1, 2, 3], [10, 20, 30], [100, 101, 102])
    demos = window.demos()
    assert len(demos) == 2
    for instruction, answer in demos:
        assert isinstance(instruction, str) and len(instruction) > 0
        assert 'destination_node' in answer


def test_gather_hop_edges_two_hops():
    # k=2, hops=2, single source (i=0).
    # hop0: 1 row of 2 neighbors; hop1: 2 rows (k**1) of 2 neighbors each.
    seed_nids = [np.array([0]), np.array([2, 1])]
    nbr_nids = [
        np.array([[2, 1]]),
        np.array([[PADDED_NODE_ID, PADDED_NODE_ID], [3, 5]]),
    ]
    nbr_times = [
        np.array([[3, 6]]),
        np.array([[0, 0], [5, 7]]),
    ]
    hop_edges = gather_hop_edges(0, 2, 2, seed_nids, nbr_nids, nbr_times)
    assert hop_edges[0] == [(0, 2, 3), (0, 1, 6)]
    # Node 2 expanded to only padding -> dropped; node 1 -> two edges.
    assert hop_edges[1] == [(1, 3, 5), (1, 5, 7)]


def test_make_multihop_user_prompt():
    hop_edges = [[(0, 2, 3)], [(2, 5, 1)]]
    prompt = make_multihop_user_prompt(np.int64(0), np.int64(9), hop_edges)
    assert '1-hop interactions' in prompt
    assert '2-hop interactions' in prompt
    assert '(0, 2, 3)' in prompt and '(2, 5, 1)' in prompt


def test_make_multihop_user_prompt_empty():
    prompt = make_multihop_user_prompt(np.int64(0), np.int64(9), [[], []])
    assert 'Predict the next interaction' in prompt


def test_schemas_importable_if_pydantic_present():
    pytest.importorskip('pydantic')
    from schemas import Step, TGAnswer, TGReasoning

    answer = TGAnswer(destination_node=42)
    assert answer.destination_node == 42
    reasoning = TGReasoning(
        steps=[Step(explanation='because', output='42')], destination_node=42
    )
    assert reasoning.destination_node == 42
    assert reasoning.steps[0].output == '42'
