import pytest
import torch
from tgtalker_utils import make_system_prompt, make_user_prompt


def test_make_system_prompt():
    system_prompt = make_system_prompt()
    assert system_prompt is not None and len(system_prompt) > 0


def test_make_user_prompt():
    src = 1
    ts = 200
    user_prompt = make_user_prompt(src, ts)
    assert user_prompt is not None and len(user_prompt) > 0


def test_bad_arg_user_prompt():
    src = torch.IntTensor([0, 1, 2, 3])
    ts = torch.IntTensor([5, 5])
    with pytest.raises(ValueError):
        make_user_prompt(src, ts)

    with pytest.raises(ValueError):
        make_user_prompt(None, None)
