import json
import subprocess
from datetime import datetime
from pathlib import Path

import psutil
import yaml


def read_experiment_configs():
    config_path = Path(__file__).parent / 'config.yaml'
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def save_experiment_results_and_exit(results):
    exp_dir = Path(__file__).resolve().parents[1] / 'experiments' / 'results'
    exp_dir.mkdir(parents=True, exist_ok=True)
    save_path = (
        exp_dir
        / f'{results["task"]}_{results["method"]}_{results["seed"]}_{results["dataset"].replace("-", "_")}.json'
    )
    with open(save_path, 'w') as f:
        json.dump(results, f)
    exit()


def run_experiment_as_subprocess(script, script_args):
    cmd = f'python {script} {" ".join(script_args)}'
    print('Running ', cmd)
    subprocess.run(cmd, shell=True)
    print('Done.')


def is_experiment_already_done(task, dataset, method, seed):
    exp_dir = Path(__file__).resolve().parents[1] / 'experiments' / 'results'
    save_path = exp_dir / f'{task}_{method}_{seed}_{dataset.replace("-", "_")}.json'
    return save_path.exists()


def setup_experiment(args, path):
    results = vars(args)
    results['task'] = path.parent.stem
    results['method'] = path.stem
    results['capture_gpu'] = bool(args.capture_gpu)
    results['current_time'] = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

    def _get_git_hash():
        return (
            subprocess.check_output(['git', 'rev-parse', 'HEAD'])
            .decode('ascii')
            .strip()
        )

    results['git_hash'] = _get_git_hash()

    def _get_cpu_info():
        model = None
        with open('/proc/cpuinfo') as f:
            for line in f:
                if line.startswith('model name'):
                    model = line.split(':', 1)[1].strip()
                    break  # first occurrence is enough
        cores = sum(1 for line in open('/proc/cpuinfo') if line.startswith('processor'))
        return f'{cores}-core {model}'

    def _get_gpu_info():
        try:
            nvidia_smi = subprocess.check_output(
                [
                    'nvidia-smi',
                    '--query-gpu=name,memory.total',
                    '--format=csv,noheader',
                ],
                encoding='utf-8',
            )
            return ':'.join(nvidia_smi.split(', '))
        except Exception:
            return None

    results['ram_available_gb'] = psutil.virtual_memory().total / 1024**3
    results['cpu_info'] = _get_cpu_info()
    results['gpu_info'] = _get_gpu_info()
    return results
