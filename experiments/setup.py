import json
import subprocess
from datetime import datetime
from pathlib import Path

import psutil


def save_experiment_results_and_exit(results):
    exp_dir = Path(__file__).resolve().parents[1] / 'experiments' / 'results'
    exp_dir.mkdir(parents=True, exist_ok=True)
    save_path = exp_dir / f'{results["task"]}_{results["method"]}.json'
    with open(save_path, 'w') as f:
        json.dump(results, f)
    exit()


def setup_experiment(args, path):
    results = vars(args)
    results['task'] = path.parent.stem
    results['method'] = path.stem
    results['capture_gpu'] = bool(args.capture_gpu)
    results['git_hash'] = (
        subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    )
    results['current_time'] = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

    model = None
    with open('/proc/cpuinfo') as f:
        for line in f:
            if line.startswith('model name'):
                model = line.split(':', 1)[1].strip()
                break  # first occurrence is enough
    logical = sum(1 for line in open('/proc/cpuinfo') if line.startswith('processor'))
    results['cpu_info'] = f'{logical}-core {model}'
    results['ram_available_gb'] = psutil.virtual_memory().total / 1024**3

    try:
        nvidia_smi = subprocess.check_output(
            [
                'nvidia-smi',
                '--query-gpu=name,memory.total',
                '--format=csv,noheader',
            ],
            encoding='utf-8',
        )
        results['gpu_info'] = ':'.join(nvidia_smi.split(', '))
    except Exception:
        results['gpu_info'] = None
    return results
