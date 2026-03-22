"""Optuna hyperparameter optimizer for XGBoost models.

Uses Optuna's TPE sampler for efficient hyperparameter search.
Integrates with ExperimentTracker to log all trial results.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import optuna
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier, XGBRegressor

logger = logging.getLogger(__name__)

# Suppress Optuna's verbose info logs
optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass
class OptimizationResult:
    """Result of hyperparameter optimization."""

    best_params: dict[str, Any]
    best_score: float
    n_trials: int
    all_trials: list[dict[str, Any]] = field(default_factory=list)
    optimization_metric: str = "accuracy"

    def to_dict(self) -> dict[str, Any]:
        return {
            "best_params": self.best_params,
            "best_score": self.best_score,
            "n_trials": self.n_trials,
            "optimization_metric": self.optimization_metric,
            "top_5_trials": self.all_trials[:5],
        }


class OptunaOptimizer:
    """Hyperparameter optimization using Optuna TPE sampler.

    Supports:
    - XGBoost classification (accuracy, f1)
    - XGBoost regression (rmse, r2)
    - Custom search spaces
    - Cross-validation based scoring
    """

    def __init__(
        self,
        task: str = "classification",
        n_trials: int = 50,
        cv_folds: int = 5,
        metric: str = "accuracy",
        random_seed: int = 42,
    ) -> None:
        self.task = task
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.metric = metric
        self.random_seed = random_seed

    def optimize(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        search_space: dict[str, Any] | None = None,
    ) -> OptimizationResult:
        """Run hyperparameter optimization."""
        study = optuna.create_study(
            direction="maximize" if self.metric in ("accuracy", "f1", "r2") else "minimize",
            sampler=optuna.samplers.TPESampler(seed=self.random_seed),
        )

        def objective(trial: optuna.Trial) -> float:
            params = self._suggest_params(trial, search_space)

            if self.task == "classification":
                model = XGBClassifier(
                    **params,
                    random_state=self.random_seed,
                    eval_metric="logloss",
                    verbosity=0,
                )
                scoring = "accuracy" if self.metric == "accuracy" else "f1_weighted"
            else:
                model = XGBRegressor(
                    **params,
                    random_state=self.random_seed,
                    eval_metric="rmse",
                    verbosity=0,
                )
                scoring = "neg_root_mean_squared_error" if self.metric == "rmse" else "r2"

            scores = cross_val_score(
                model, x_train, y_train, cv=self.cv_folds, scoring=scoring, n_jobs=-1
            )
            return float(scores.mean())

        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)

        # Collect all trial results
        all_trials = []
        for trial in study.trials:
            all_trials.append({
                "number": trial.number,
                "params": trial.params,
                "score": trial.value,
            })
        def _trial_score(t: dict[str, Any]) -> float:
            s = t.get("score")
            return float(s) if s is not None else 0.0

        all_trials.sort(key=_trial_score, reverse=True)

        result = OptimizationResult(
            best_params=study.best_params,
            best_score=study.best_value,
            n_trials=len(study.trials),
            all_trials=all_trials,
            optimization_metric=self.metric,
        )

        logger.info(
            "Optimization complete: best %s=%.4f after %d trials",
            self.metric,
            result.best_score,
            result.n_trials,
        )
        return result

    def _suggest_params(
        self, trial: optuna.Trial, search_space: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Suggest hyperparameters for a trial."""
        if search_space:
            return self._custom_search(trial, search_space)

        # Default XGBoost search space
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }

    def _custom_search(self, trial: optuna.Trial, space: dict[str, Any]) -> dict[str, Any]:
        """Build params from user-defined search space."""
        params: dict[str, Any] = {}
        for name, config in space.items():
            if isinstance(config, dict):
                kind = config.get("type", "float")
                if kind == "int":
                    params[name] = trial.suggest_int(name, config["low"], config["high"])
                elif kind == "float":
                    params[name] = trial.suggest_float(
                        name, config["low"], config["high"], log=config.get("log", False)
                    )
                elif kind == "categorical":
                    params[name] = trial.suggest_categorical(name, config["choices"])
            else:
                params[name] = config  # Fixed value
        return params
