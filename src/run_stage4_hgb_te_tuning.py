from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline

from experiment_framework import (
    ID_COL,
    RANDOM_STATE,
    TARGET,
    apply_feature_set,
    blend_experiments,
    get_cv,
    load_data,
    make_preprocessor,
    project_root,
    rmse,
)


FEATURE_SET = "fs_07_all_safe"
MODEL_FAMILY = "hgb_te_tuned"
N_SPLITS = 5


HGB_TE_CONFIGS = [
    {
        "name": "base_recheck",
        "smoothing": 25.0,
        "max_iter": 800,
        "learning_rate": 0.030,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 20,
        "l2_regularization": 0.05,
    },
    {
        "name": "smooth_regularized",
        "smoothing": 80.0,
        "max_iter": 900,
        "learning_rate": 0.025,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 30,
        "l2_regularization": 0.20,
    },
    {
        "name": "compact_fast",
        "smoothing": 25.0,
        "max_iter": 650,
        "learning_rate": 0.045,
        "max_leaf_nodes": 15,
        "min_samples_leaf": 20,
        "l2_regularization": 0.03,
    },
    {
        "name": "deeper_regularized",
        "smoothing": 40.0,
        "max_iter": 900,
        "learning_rate": 0.025,
        "max_leaf_nodes": 63,
        "min_samples_leaf": 25,
        "l2_regularization": 0.15,
    },
    {
        "name": "low_lr_long",
        "smoothing": 25.0,
        "max_iter": 1100,
        "learning_rate": 0.020,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 20,
        "l2_regularization": 0.05,
    },
    {
        "name": "strong_l2",
        "smoothing": 80.0,
        "max_iter": 1000,
        "learning_rate": 0.025,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 40,
        "l2_regularization": 0.40,
    },
    {
        "name": "low_smoothing",
        "smoothing": 10.0,
        "max_iter": 800,
        "learning_rate": 0.030,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 20,
        "l2_regularization": 0.05,
    },
    {
        "name": "high_smoothing",
        "smoothing": 150.0,
        "max_iter": 800,
        "learning_rate": 0.030,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 20,
        "l2_regularization": 0.05,
    },
    {
        "name": "smaller_leaf",
        "smoothing": 25.0,
        "max_iter": 800,
        "learning_rate": 0.030,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 10,
        "l2_regularization": 0.08,
    },
]


def build_hgb_te_pipeline(config, numeric_features, categorical_features):
    preprocess = make_preprocessor(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        encoding="target",
        dense=True,
        smoothing=config["smoothing"],
    )
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        max_iter=config["max_iter"],
        learning_rate=config["learning_rate"],
        max_leaf_nodes=config["max_leaf_nodes"],
        min_samples_leaf=config["min_samples_leaf"],
        l2_regularization=config["l2_regularization"],
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=30,
        random_state=RANDOM_STATE,
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", model),
        ]
    )


def update_summary(row):
    root = project_root()
    report_dir = root / "reports" / "experiments"
    summary_path = report_dir / "summary.csv"
    new_summary = pd.DataFrame([row])

    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        summary = pd.concat([summary, new_summary], ignore_index=True)
        summary = summary.drop_duplicates(subset=["experiment_id"], keep="last")
    else:
        summary = new_summary

    summary.sort_values("cv_rmse").to_csv(summary_path, index=False)


def run_tuning_config(config, train_fe, test_fe, X, y, X_test, numeric_features, categorical_features, splitter, y_bins):
    root = project_root()
    report_dir = root / "reports" / "experiments"
    submission_dir = root / "submissions"
    report_dir.mkdir(parents=True, exist_ok=True)
    submission_dir.mkdir(exist_ok=True)

    experiment_id = f"exp_4{config['idx']:02d}_hgb_te_{config['name']}"
    print(f"\n=== {experiment_id} ===")
    print(config)

    oof = np.zeros(len(X), dtype=float)
    test_pred = np.zeros(len(X_test), dtype=float)
    fold_rows = []

    for fold, (train_idx, valid_idx) in enumerate(splitter.split(X, y_bins), start=1):
        model = build_hgb_te_pipeline(config, numeric_features, categorical_features)
        model = clone(model)

        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        model.fit(X_train, y_train)
        valid_pred = np.clip(model.predict(X_valid), 0, 10)
        fold_score = rmse(y_valid, valid_pred)

        oof[valid_idx] = valid_pred
        test_pred += np.clip(model.predict(X_test), 0, 10) / N_SPLITS
        fold_rows.append(
            {
                "experiment_id": experiment_id,
                "feature_set": FEATURE_SET,
                "model_name": MODEL_FAMILY,
                "config_name": config["name"],
                "fold": fold,
                "rmse": fold_score,
            }
        )
        print(f"  fold {fold}: {fold_score:.6f}")

    cv_score = rmse(y, oof)
    fold_scores = [row["rmse"] for row in fold_rows]
    print(f"OOF RMSE: {cv_score:.6f}")

    fold_path = report_dir / f"{experiment_id}_folds.csv"
    oof_path = report_dir / f"{experiment_id}_oof.csv"
    prediction_path = report_dir / f"{experiment_id}_test_predictions.csv"
    submission_path = submission_dir / f"{experiment_id}_{MODEL_FAMILY}_{FEATURE_SET}.csv"
    params_path = report_dir / f"{experiment_id}_params.csv"

    pd.DataFrame(fold_rows).to_csv(fold_path, index=False)
    pd.DataFrame({ID_COL: train_fe[ID_COL], TARGET: y, "oof_pred": oof}).to_csv(oof_path, index=False)
    pd.DataFrame({ID_COL: test_fe[ID_COL], "prediction": np.clip(test_pred, 0, 10)}).to_csv(
        prediction_path,
        index=False,
    )
    pd.DataFrame([config]).to_csv(params_path, index=False)
    pd.DataFrame({ID_COL: test_fe[ID_COL], TARGET: np.clip(test_pred, 0, 10)}).to_csv(
        submission_path,
        index=False,
    )

    update_summary(
        {
            "experiment_id": experiment_id,
            "feature_set": FEATURE_SET,
            "model_name": MODEL_FAMILY,
            "n_splits": N_SPLITS,
            "cv_rmse": cv_score,
            "fold_mean": float(np.mean(fold_scores)),
            "fold_std": float(np.std(fold_scores)),
            "oof_path": str(oof_path),
            "prediction_path": str(prediction_path),
            "submission_path": str(submission_path),
        }
    )

    return {
        "experiment_id": experiment_id,
        "config_name": config["name"],
        "cv_rmse": cv_score,
        "fold_mean": float(np.mean(fold_scores)),
        "fold_std": float(np.std(fold_scores)),
        "submission_path": str(submission_path),
    }


def main():
    train, test = load_data()
    train_fe = apply_feature_set(train, FEATURE_SET)
    test_fe = apply_feature_set(test, FEATURE_SET)

    feature_cols = [col for col in train_fe.columns if col not in [TARGET, ID_COL]]
    X = train_fe[feature_cols]
    y = train_fe[TARGET]
    X_test = test_fe[feature_cols]

    numeric_features = X.select_dtypes(include="number").columns.tolist()
    categorical_features = X.select_dtypes(exclude="number").columns.tolist()
    splitter, y_bins = get_cv(y, n_splits=N_SPLITS)

    print("Stage 4 - HGB Target Encoding Tuning")
    print("Feature set:", FEATURE_SET)
    print("features:", len(feature_cols), "numeric:", len(numeric_features), "categorical:", len(categorical_features))
    print("config count:", len(HGB_TE_CONFIGS))

    results = []
    for idx, config in enumerate(HGB_TE_CONFIGS, start=1):
        config = dict(config)
        config["idx"] = idx
        results.append(
            run_tuning_config(
                config=config,
                train_fe=train_fe,
                test_fe=test_fe,
                X=X,
                y=y,
                X_test=X_test,
                numeric_features=numeric_features,
                categorical_features=categorical_features,
                splitter=splitter,
                y_bins=y_bins,
            )
        )

    result_df = pd.DataFrame(results).sort_values("cv_rmse").reset_index(drop=True)
    print("\nTuning results:")
    print(result_df)

    top3_ids = result_df.head(3)["experiment_id"].tolist()
    print("\nTop 3 for blend:", top3_ids)

    blend_experiments(
        experiment_id="exp_490_hgb_te_tuned_top3_equal_blend",
        source_experiment_ids=top3_ids,
    )

    inv = 1 / result_df.head(3)["cv_rmse"].to_numpy()
    inv_weights = (inv / inv.sum()).tolist()
    blend_experiments(
        experiment_id="exp_491_hgb_te_tuned_top3_inverse_blend",
        source_experiment_ids=top3_ids,
        weights=inv_weights,
    )


if __name__ == "__main__":
    main()

