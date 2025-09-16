from pathlib import Path

import yaml

EXPERIMENTS_CONFIG = Path(__file__).parent / 'config.yaml'


with open(EXPERIMENTS_CONFIG, 'r') as f:
    configs = yaml.safe_load(f)


global_configs = configs['global_configs']
dataset_configs = configs['datasets']
experiment_configs = configs['experiments']

num_runs = global_configs['num_runs_per_experiment']
base_seed = global_configs['base_seed']
do_extra_run_with_gpu_profiler = global_configs['do_extra_run_with_gpu_profiler']

examples_root = Path(__file__).resolve().parents[1] / 'examples'


def run_experiment(script, script_args):
    print('Running ', script, script_args)


for task in ['linkproppred', 'nodeproppred']:
    for dataset in dataset_configs[task]:
        for experiment in experiment_configs[task]:
            script = examples_root / task / f'{experiment["method"]}.py'
            script_args = experiment['script_flags'].split('\n')
            if dataset in experiment['additional_data_specific_script_flags']:
                script_args += experiment['additional_data_specific_script_flags'][
                    dataset
                ].split('\n')

            for run in range(num_runs):
                script_args.append(f'--seed {base_seed + run}')
                run_experiment(script, script_args)
                script_args.pop()

            if do_extra_run_with_gpu_profiler:
                script_args.append(f'--seed {base_seed + num_runs}')
                script_args.append(f'--capture-gpu')
                run_experiment(script, script_args)
