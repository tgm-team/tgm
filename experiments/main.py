import logging
from typing import List

from experiments.util import (
    EXAMPLES_ROOT,
    is_experiment_already_done,
    read_experiment_configs,
    run_experiment_as_subprocess,
    setup_basic_logging,
)


def run_experiment(
    script: str, script_args: List[str], task: str, dataset: str, method: str, seed: int
) -> None:
    if is_experiment_already_done(task, dataset, method, seed):
        logging.info(f'Experiment for {script} {dataset} {seed} already done, skipping')
    else:
        script_args.append(f'--seed {seed}')
        run_experiment_as_subprocess(script, script_args)
        script_args.pop()


def main() -> None:
    configs = read_experiment_configs()
    global_configs = configs['global_configs']
    dataset_configs = configs['datasets']
    experiment_configs = configs['experiments']

    setup_basic_logging(global_configs['log_file'])

    for task in ['linkproppred', 'nodeproppred']:
        for dataset in dataset_configs[task]:
            for experiment in experiment_configs[task]:
                method = experiment['method']
                script = EXAMPLES_ROOT / task / f'{method}.py'
                script_args = experiment['script_flags'].split('\n')

                if global_configs['run_with_gpu_profiler']:
                    script_args.append(f'--capture-gpu')

                if dataset in experiment['additional_data_specific_script_flags']:
                    script_args += experiment['additional_data_specific_script_flags'][
                        dataset
                    ].split('\n')

                for run in range(global_configs['num_runs_per_experiment']):
                    seed = global_configs['base_seed'] + run
                    run_experiment(script, script_args, task, dataset, method, seed)

                if global_configs['do_extra_run_with_cprofiler']:
                    script_args.append(f'--capture-cprofile')
                    seed = (
                        global_configs['base_seed']
                        + global_configs['num_runs_per_experiment']
                    )
                    run_experiment(script, script_args, task, dataset, method, seed)
                    script_args.pop()


if __name__ == '__main__':
    main()
