import os
import subprocess
from datetime import datetime as dt
from pathlib import Path

import pytest

DATASETS = [
    pytest.param('tgbl-wiki', marks=pytest.mark.small),
    pytest.param('tgbn-trade', marks=pytest.mark.small),
    pytest.param('tgbn-genre', marks=pytest.mark.medium),
    pytest.param('tgbl-coin', marks=pytest.mark.medium),
    pytest.param('tgbl-flight', marks=pytest.mark.large),
    pytest.param('tgbn-reddit', marks=pytest.mark.large),
]


@pytest.fixture(scope='session')
def datasets():
    return DATASETS


@pytest.fixture(scope='session', autouse=True)
def ci_run_context():
    # File-io work that should be shared across all performance tests in a single run
    def get_commit_hash() -> str:
        return subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'], text=True
        ).strip()

    ci_run_dir = f'{dt.now().strftime("%Y-%m-%d-%H-%M")}_{get_commit_hash()}'
    log_base = Path(
        os.path.expanduser(
            os.environ.get('TGM_CI_PERF_LOG_BASE', str(Path.home() / 'tgm_ci_perf'))
        )
    )
    log_dir = log_base / ci_run_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    # Save the log directory path for easy parsing in the Github action
    latest_path_file = log_base / 'latest_path.txt'
    latest_path_file.write_text(f'{log_dir}\n{ci_run_dir}')
