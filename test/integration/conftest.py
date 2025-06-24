import os
import subprocess
import tempfile
import time
from datetime import datetime as dt
from pathlib import Path

import pytest


def pytest_configure(config):
    config.addinivalue_line(
        'markers', 'slurm(resources): Run test via sbatch with given Slurm resources.'
    )


@pytest.fixture(scope='session', autouse=True)
def ci_run_context():
    # File-io work that should be shared across all integration tests in a single run
    def get_commit_hash() -> str:
        return subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'], text=True
        ).strip()

    ci_run_dir = f'{dt.now().strftime("%Y-%m-%d-%H-%M")}_{get_commit_hash()}'
    log_base = Path(
        os.path.expanduser(
            os.environ.get('TGM_CI_LOG_BASE', str(Path.home() / 'tgm_ci'))
        )
    )
    log_dir = log_base / ci_run_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    # Save the log directory path for easy parsing in the Github action
    latest_path_file = log_base / 'latest_path.txt'
    latest_path_file.write_text(f'{log_dir}\n{ci_run_dir}')

    return {
        'log_dir': log_dir,
        'project_root': Path(__file__).resolve().parents[2],
    }


@pytest.fixture
def slurm_job_runner(ci_run_context, request):
    def run(cmd):
        job_script = f"""#!/bin/bash
set -euo pipefail

ROOT_DIR="{ci_run_context['project_root']}"
module load python/3.10
module load cudatoolkit/11.7

echo "===== JOB INFO ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node List: $SLURM_NODELIST"
echo "Num CPUs: $SLURM_CPUS_PER_TASK"
echo "GPU(s): ${{CUDA_VISIBLE_DEVICES:-None}}"
echo "Memory: ${{SLURM_MEM_PER_NODE:-N/A}}"
echo "Start Time: $(date)"
echo "===================="

{cmd}
"""
        with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.sh') as f:
            f.write(job_script)
            script_path = f.name

        caller = request.node
        marker = caller.get_closest_marker('slurm')
        slurm_resources = marker.kwargs.get('resources', []) if marker else []

        log_dir = ci_run_context['log_dir']
        job_name = caller.name.replace('[', '_').replace(']', '').replace(':', '_')
        slurm_out = log_dir / f'{job_name}.out'
        slurm_err = log_dir / f'{job_name}.err'

        sbatch_cmd = [
            'sbatch',
            f'--job-name={job_name}',
            f'--output={slurm_out}',
            f'--error={slurm_err}',
            *slurm_resources,
            script_path,
        ]

        output = subprocess.check_output(sbatch_cmd, text=True)
        job_id = output.strip().split()[-1]

        # Save job id as metadata for post-scheduling collecction in slurm_job_finish
        job_record_file = log_dir / f'{job_name}.job'
        job_record_file.write_text(job_id)

    return run


def slurm_job_finish(session, _):
    log_base = Path(os.environ.get('TGM_CI_LOG_BASE', str(Path.home() / 'tgm_ci')))
    latest_path_file = log_base / 'latest_path.txt'
    log_dir = Path(latest_path_file.read_text().splitlines()[0])

    job_files = list(log_dir.glob('*.job'))
    for job_file in job_files:
        job_id = job_file.read_text()

        # Poll slurm for job completion status
        while True:
            time.sleep(10)
            result = subprocess.run(
                ['sacct', '-j', job_id, '--format=State', '--noheader'],
                capture_output=True,
                text=True,
            )
            state = result.stdout.strip().split()[0]
            if state in ['COMPLETED', 'FAILED', 'CANCELLED']:
                break

        if state != 'COMPLETED':
            session.exitstatus = 1  # Mark pytest as failed
