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

        ci_log_dir = f'{dt.now().strftime("%Y-%m-%d-%H-%M")}_{get_commit_hash()}'
        log_dir = Path.home() / 'tgm_ci' / ci_log_dir
        log_dir.mkdir(parents=True, exist_ok=True)

        job_name = caller.name.replace('[', '_').replace(']', '').replace(':', '_')
        timestamp = dt.now().strftime('%Y-%m-%d-%H:%M:%S')
        slurm_out = log_dir / f'{job_name}_{timestamp}.out'
        slurm_err = log_dir / f'{job_name}_{timestamp}.err'

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
            print(result.stdout.strip())
            state = result.stdout.strip().split()[0]
            if state in ['COMPLETED', 'FAILED', 'CANCELLED']:
                break

        output_text = slurm_out.read_text() if slurm_out.exists() else ''
        return state, output_text

    return run
