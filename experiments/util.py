import argparse
import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import psutil
import yaml

EXAMPLES_ROOT = Path(__file__).resolve().parents[1] / 'examples'

from tgm.util.perf import Profiling, Usage


def setup_basic_logging(
    log_file_path: str | Path | None = None,
    log_file_logging_level: int = logging.DEBUG,
    stream_logging_level: int = logging.INFO,
) -> None:
    handlers: List[logging.Handler] = []

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(stream_logging_level)
    stream_handler.setFormatter(
        logging.Formatter('[%(asctime)s] %(name)s - %(levelname)s %(message)s')
    )
    handlers.append(stream_handler)

    if log_file_path is not None:
        file_handler = logging.FileHandler(filename=log_file_path, mode='a')
        file_handler.setLevel(log_file_logging_level)
        file_handler.setFormatter(
            logging.Formatter(
                '[%(asctime)s] %(name)s - %(levelname)s [%(processName)s %(threadName)s %(name)s.%(funcName)s:%(lineno)d] %(message)s',
            )
        )
        handlers.append(file_handler)

    logging.basicConfig(
        level=logging.DEBUG,
        format='[%(asctime)s] %(name)s - %(levelname)s [%(processName)s %(threadName)s %(name)s.%(funcName)s:%(lineno)d] %(message)s',
        handlers=handlers,
    )


def read_experiment_configs() -> Dict[str, Any]:
    config_path = Path(__file__).parent / 'config.yaml'
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def save_experiment_results_and_exit(results: Dict[str, Any]) -> None:
    save_path = _get_experiment_save_path(
        task=results['task'],
        dataset=results['dataset'],
        method=results['method'],
        seed=results['seed'],
    )

    # Finish profiling
    results['u'].__exit__()
    results['peak_gpu_gb'] = results['u'].gpu_gb
    results.pop('u')
    if 'p' in results:
        results['p'].__exit__()
        results.pop('p')
    else:
        # Only save json info if cprofile was not in use
        with open(save_path, 'w') as f:
            json.dump(results, f)
    exit()


def run_experiment_as_subprocess(script: str, script_args: List[str]) -> None:
    cmd = f'python {script} {" ".join(script_args)}'
    logging.info(f'Running {cmd}')
    subprocess.run(cmd, shell=True)


def is_experiment_already_done(task: str, dataset: str, method: str, seed: int) -> bool:
    json_path = _get_experiment_save_path(task, dataset, method, seed)
    cprofile_path = json_path.with_suffix('.profile')
    return json_path.exists() or cprofile_path.exists()


def setup_experiment(args: argparse.Namespace, path: Path) -> dict:
    results = vars(args)
    results['task'] = path.parent.stem
    results['method'] = path.stem
    results['capture_gpu'] = bool(args.capture_gpu)
    results['current_time'] = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    results['git_hash'] = _get_git_hash()
    results['ram_available_gb'] = _get_ram_available_gb()
    results['cpu_info'] = _get_cpu_info()
    results['gpu_info'] = _get_gpu_info()

    # Start profiling
    if args.capture_cprofile:
        profile_path = _get_experiment_save_path(
            results['task'], results['dataset'], results['method'], results['seed']
        ).with_suffix('.profile')
        results['p'] = Profiling(str(profile_path)).__enter__()
    results['u'] = Usage(gpu=args.capture_gpu).__enter__()
    return results


def _get_experiment_save_path(task: str, dataset: str, method: str, seed: int) -> Path:
    exp_dir = Path(__file__).resolve().parents[1] / 'experiments' / 'results'
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir / f'{task}_{method}_{seed}_{dataset.replace("-", "_")}.json'


def _get_git_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()


def _get_ram_available_gb() -> float:
    return psutil.virtual_memory().total / 1024**3


def _get_cpu_info() -> str | None:
    model = None
    with open('/proc/cpuinfo') as f:
        for line in f:
            if line.startswith('model name'):
                model = line.split(':', 1)[1].strip()
                break  # first occurrence is enough
    cores = sum(1 for line in open('/proc/cpuinfo') if line.startswith('processor'))
    return f'{cores}-core {model}'


def _get_gpu_info() -> str | None:
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
