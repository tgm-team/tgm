import os
from typing import Dict, Tuple

import pandas as pd

from .configs import EXPERIMENTS_ARTIFACT


class EarlyStopping:
    def __init__(
        self, higher_is_better: bool = True, patience: int = 50, tolerance: float = 0.05
    ):
        self.higher_is_better = higher_is_better
        self.patience = self.init_patience = patience
        self.tolerance = tolerance

        self.best_epoch = -1
        self.best_performance = float('-inf') if higher_is_better else float('inf')

    def __call__(self, epoch: int, performance: float) -> Tuple[bool, bool]:
        assert epoch > self.best_epoch
        is_best_epoch = False

        if self.higher_is_better:
            if performance > self.best_performance:
                self.patience = self.init_patience
                is_best_epoch = True
                self.best_epoch = epoch
                self.best_performance = performance

            elif self.best_epoch - performance > self.tolerance:
                self.patience -= 1
        else:
            if performance < self.best_performance:
                self.patience = self.init_patience
                is_best_epoch = True
                self.best_epoch = epoch
                self.best_performance = performance

            elif performance - self.best_epoch > self.tolerance:
                self.patience -= 1

        return is_best_epoch, self.patience <= 0


def save_results(experiment_id: str, results: Dict, intermediate_path: str = ''):
    partial_path = f'{EXPERIMENTS_ARTIFACT}/results/{intermediate_path}'
    if not os.path.exists(partial_path):
        os.makedirs(partial_path)

    result_path = f'{partial_path}/{experiment_id}.csv'
    if not os.path.exists(result_path):
        result_df = pd.DataFrame(columns=results.keys())
    else:
        result_df = pd.read_csv(result_path)

    result_df = result_df._append(results, ignore_index=True)
    result_df.to_csv(result_path, index=False)
