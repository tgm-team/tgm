import pytest

from tgm.util.seed import seed_everything


@pytest.fixture(autouse=True)
def run_seed_before_tests():
    seed_everything(1337)
    yield
