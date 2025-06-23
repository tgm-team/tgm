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


@pytest.fixture
def slurm_job_runner(request):
    project_root = Path(__file__).resolve().parents[2]

    def run(cmd):
        job_script = f"""#!/bin/bash
set -euo pipefail

ROOT_DIR="{project_root}"
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

        def get_commit_hash() -> str:
            return subprocess.check_output(
                ['git', 'rev-parse', '--short', 'HEAD'], text=True
            ).strip()

        ci_run_dir = f'{dt.now().strftime("%Y-%m-%d-%H-%M")}_{get_commit_hash()}'
        log_base = Path(os.environ.get('TGM_CI_LOG_BASE', Path.home() / 'tgm_ci'))
        log_dir = log_base / ci_run_dir
        log_dir.mkdir(parents=True, exist_ok=True)

        # Save the log directory path for easy parsing in the Github action
        latest_path_file = log_base / 'latest_path.txt'
        if not latest_path_file.exists():
            latest_path_file.write_text(f'{log_dir}\n{ci_run_dir}')

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
        job_number = output.strip().split()[-1]

        # Poll slurm for job completion status
        while True:
            time.sleep(10)

            result = subprocess.run(
                ['sacct', '-j', job_number, '--format=State', '--noheader'],
                capture_output=True,
                text=True,
            )
            state = result.stdout.strip().split()[0]
            if state in ['COMPLETED', 'FAILED', 'CANCELLED']:
                break

        output_text = slurm_out.read_text() if slurm_out.exists() else ''
        print('OUTPUT: ', output_text)
        return state, output_text

    return run
