"""
Hyperparameter tuning for Direction Classifier using Optuna.

Uses walk-forward validation to prevent overfitting to specific time periods.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.metrics import f1_score, roc_auc_score

from src.utils.logger import get_logger

logger = get_logger(__name__)


class DirectionClassifierTuner:
    """
    Optuna-based hyperparameter tuner for Direction Classifier.

    Uses walk-forward validation for robust evaluation.
    """

    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 5,
        num_classes: int = 2,
        random_state: int = 42,
    ):
        """
        Initialize tuner with training data.

        Args:
            X: Feature DataFrame
            y: Labels Series
            n_splits: Number of walk-forward validation splits
            num_classes: 2 for binary, 3 for ternary
            random_state: Random seed
        """
        self.X = X
        self.y = y
        self.n_splits = n_splits
        self.num_classes = num_classes
        self.random_state = random_state
        self.best_params: Optional[Dict[str, Any]] = None
        self.study: Optional[optuna.Study] = None

        logger.info(
            f"Initialized DirectionClassifierTuner: "
            f"{len(X)} samples, {len(X.columns)} features, "
            f"{n_splits} folds, {num_classes} classes"
        )

    def _create_walk_forward_splits(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Create walk-forward validation splits."""
        n_samples = len(self.X)
        test_size = int(n_samples * 0.1)
        min_train = int(n_samples * 0.3)

        splits = []
        for i in range(self.n_splits):
            train_end = min_train + i * test_size
            test_start = train_end
            test_end = min(test_start + test_size, n_samples)

            if train_end >= n_samples or test_end > n_samples:
                continue

            train_idx = np.arange(0, train_end)
            test_idx = np.arange(test_start, test_end)
            splits.append((train_idx, test_idx))

        return splits

    def _objective(self, trial: optuna.Trial) -> float:
        """
        Optuna objective function.

        Args:
            trial: Optuna trial object

        Returns:
            Mean AUC across walk-forward folds
        """
        # Define hyperparameter search space
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.001, 1.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.001, 10.0, log=True),
            'random_state': self.random_state,
            'n_jobs': -1,
            'verbosity': 0,
        }

        # Set objective based on num_classes
        if self.num_classes == 2:
            params['objective'] = 'binary:logistic'
            params['eval_metric'] = 'auc'
        else:
            params['objective'] = 'multi:softprob'
            params['num_class'] = self.num_classes
            params['eval_metric'] = 'mlogloss'

        # Walk-forward validation
        splits = self._create_walk_forward_splits()
        auc_scores = []

        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            X_train = self.X.iloc[train_idx]
            y_train = self.y.iloc[train_idx]
            X_test = self.X.iloc[test_idx]
            y_test = self.y.iloc[test_idx]

            # Create validation set from end of training
            val_size = int(len(X_train) * 0.15)
            X_val = X_train.iloc[-val_size:]
            y_val = y_train.iloc[-val_size:]
            X_train_final = X_train.iloc[:-val_size]
            y_train_final = y_train.iloc[:-val_size]

            # Train model
            model = xgb.XGBClassifier(**params, early_stopping_rounds=20)
            model.fit(
                X_train_final, y_train_final,
                eval_set=[(X_val, y_val)],
                verbose=False
            )

            # Evaluate
            y_proba = model.predict_proba(X_test)

            try:
                if self.num_classes == 2:
                    auc = roc_auc_score(y_test, y_proba[:, 1])
                else:
                    auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')
                auc_scores.append(auc)
            except ValueError:
                auc_scores.append(0.5)

            # Pruning: stop early if this trial is clearly worse
            trial.report(np.mean(auc_scores), fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return np.mean(auc_scores)

    def tune(
        self,
        n_trials: int = 100,
        timeout: Optional[int] = None,
        show_progress: bool = True,
    ) -> Dict[str, Any]:
        """
        Run hyperparameter tuning.

        Args:
            n_trials: Number of trials to run
            timeout: Optional timeout in seconds
            show_progress: Whether to show progress bar

        Returns:
            Dictionary of best parameters
        """
        logger.info(f"Starting Optuna tuning with {n_trials} trials")

        # Create study
        sampler = TPESampler(seed=self.random_state)
        pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=2)

        # Suppress Optuna logs
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        self.study = optuna.create_study(
            direction='maximize',
            sampler=sampler,
            pruner=pruner,
        )

        # Run optimization
        self.study.optimize(
            self._objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=show_progress,
        )

        self.best_params = self.study.best_params

        logger.info(f"Best AUC: {self.study.best_value:.4f}")
        logger.info(f"Best params: {self.best_params}")

        return self.best_params

    def get_best_params_for_training(self) -> Dict[str, Any]:
        """
        Get best parameters formatted for DirectionClassifier.

        Returns:
            Parameters dict ready for model training
        """
        if self.best_params is None:
            raise ValueError("Must run tune() first")

        params = self.best_params.copy()

        # Add fixed params
        params['random_state'] = self.random_state
        params['n_jobs'] = -1
        params['early_stopping_rounds'] = 20

        if self.num_classes == 2:
            params['objective'] = 'binary:logistic'
            params['eval_metric'] = 'logloss'
        else:
            params['objective'] = 'multi:softprob'
            params['num_class'] = self.num_classes
            params['eval_metric'] = 'mlogloss'

        return params

    def save_best_params(self, path: str) -> None:
        """Save best parameters to JSON file."""
        if self.best_params is None:
            raise ValueError("Must run tune() first")

        save_data = {
            "best_params": self.best_params,
            "best_auc": self.study.best_value,
            "n_trials": len(self.study.trials),
            "num_classes": self.num_classes,
        }

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(save_data, f, indent=2)

        logger.info(f"Best params saved to {path}")

    @staticmethod
    def load_best_params(path: str) -> Dict[str, Any]:
        """Load best parameters from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return data["best_params"]

    def get_trial_summary(self) -> pd.DataFrame:
        """Get summary of all trials."""
        if self.study is None:
            raise ValueError("Must run tune() first")

        trials_data = []
        for trial in self.study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                row = {"trial": trial.number, "auc": trial.value}
                row.update(trial.params)
                trials_data.append(row)

        return pd.DataFrame(trials_data).sort_values("auc", ascending=False)


def quick_tune(
    X: pd.DataFrame,
    y: pd.Series,
    n_trials: int = 50,
    num_classes: int = 2,
) -> Dict[str, Any]:
    """
    Quick hyperparameter tuning with default settings.

    Args:
        X: Feature DataFrame
        y: Labels Series
        n_trials: Number of trials
        num_classes: 2 for binary, 3 for ternary

    Returns:
        Best parameters dictionary
    """
    tuner = DirectionClassifierTuner(X, y, num_classes=num_classes)
    best_params = tuner.tune(n_trials=n_trials)
    return tuner.get_best_params_for_training()
