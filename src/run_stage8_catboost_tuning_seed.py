from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from catboost import CatBoostRegressor

from experiment_framework import (
    ID_COL,
    RANDOM_STATE,
    TARGET,
    apply_feature_set,
    blend_experiments,
    get_cv,
    load_data,
    project_root,
    rmse,
)


N_SPLITS = 5
PRIMARY_FEATURE_SET = "fs_02_missing_indicators"
MODEL_NAME = "cat_native_tuned"
SEEDS = [42, 11, 77, 2026, 3407]
CATBOOST_TASK_TYPE = os.getenv("CATBOOST_TASK_TYPE", "CPU").upper()
CATBOOST_DEVICES = os.getenv("CATBOOST_DEVICES", "0")
FAST_MODE = os.getenv("FAST_MODE", "0") == "1"


CAT_TUNING_CONFIGS = [
    {
        "name": "base_es",
        "iterations": 2500,
        "learning_rate": 0.025,
        "depth": 6,
        "l2_leaf_reg": 4.0,
        "random_strength": 1.0,
        "bagging_temperature": 1.0,
    },
    {
        "name": "depth5_reg",
        "iterations": 3000,
        "learning_rate": 0.020,
        "depth": 5,
        "l2_leaf_reg": 6.0,
        "random_strength": 1.0,
        "bagging_temperature": 0.8,
    },
    {
        "name": "depth6_reg",
        "iterations": 2800,
        "learning_rate": 0.022,
        "depth": 6,
        "l2_leaf_reg": 8.0,
        "random_strength": 1.2,
        "bagging_temperature": 0.8,
    },
    {
        "name": "depth7_reg",
        "iterations": 2400,
        "learning_rate": 0.020,
        "depth": 7,
        "l2_leaf_reg": 8.0,
        "random_strength": 1.5,
        "bagging_temperature": 1.0,
    },
    {
        "name": "depth4_long",
        "iterations": 3500,
        "learning_rate": 0.018,
        "depth": 4,
        "l2_leaf_reg": 5.0,
        "random_strength": 1.0,
        "bagging_temperature": 0.8,
    },
    {
        "name": "faster_depth6",
        "iterations": 1800,
        "learning_rate": 0.035,
        "depth": 6,
        "l2_leaf_reg": 4.0,
        "random_strength": 1.0,
        "bagging_temperature": 1.0,
    },
    {
        "name": "low_random_strength",
        "iterations": 2500,
        "learning_rate": 0.025,
        "depth": 6,
        "l2_leaf_reg": 4.0,
        "random_strength": 0.3,
        "bagging_temperature": 1.0,
    },
    {
        "name": "high_random_strength",
        "iterations": 2500,
        "learning_rate": 0.025,
        "depth": 6,
        "l2_leaf_reg": 4.0,
        "random_strength": 2.0,
        "bagging_temperature": 1.2,
    },
]


def catboost_runtime_params():
    if CATBOOST_TASK_TYPE == "GPU":
        return {
            "task_type": "GPU",
            "devices": CATBOOST_DEVICES,
        }
    return {}


def prepare_catboost_frame(train_part, valid_part=None, test_part=None, numeric_features=None, categorical_features=None):
    numeric_features = numeric_features or []
    categorical_features = categorical_features or []

    train_out = train_part.copy()
    valid_out = valid_part.copy() if valid_part is not None else None
    test_out = test_part.copy() if test_part is not None else None

    medians = train_out[numeric_features].median()
    train_out[numeric_features] = train_out[numeric_features].fillna(medians)
    if valid_out is not None:
        valid_out[numeric_features] = valid_out[numeric_features].fillna(medians)
    if test_out is not None:
        test_out[numeric_features] = test_out[numeric_features].fillna(medians)

    for col in categorical_features:
        train_out[col] = train_out[col].astype("object").where(train_out[col].notna(), "Bilinmiyor").astype(str)
        if valid_out is not None:
            valid_out[col] = valid_out[col].astype("object").where(valid_out[col].notna(), "Bilinmiyor").astype(str)
        if test_out is not None:
            test_out[col] = test_out[col].astype("object").where(test_out[col].notna(), "Bilinmiyor").astype(str)

    return train_out, valid_out, test_out


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


def save_outputs(experiment_id, feature_set, train_ids, test_ids, y, oof, test_pred, fold_rows, config):
    root = project_root()
    report_dir = root / "reports" / "experiments"
    submission_dir = root / "submissions"
    report_dir.mkdir(parents=True, exist_ok=True)
    submission_dir.mkdir(exist_ok=True)

    fold_scores = [row["rmse"] for row in fold_rows]
    cv_score = rmse(y, oof)

    fold_path = report_dir / f"{experiment_id}_folds.csv"
    oof_path = report_dir / f"{experiment_id}_oof.csv"
    prediction_path = report_dir / f"{experiment_id}_test_predictions.csv"
    submission_path = submission_dir / f"{experiment_id}_{MODEL_NAME}_{feature_set}.csv"
    params_path = report_dir / f"{experiment_id}_params.csv"

    pd.DataFrame(fold_rows).to_csv(fold_path, index=False)
    pd.DataFrame({ID_COL: train_ids, TARGET: y, "oof_pred": np.clip(oof, 0, 10)}).to_csv(
        oof_path,
        index=False,
    )
    pd.DataFrame({ID_COL: test_ids, "prediction": np.clip(test_pred, 0, 10)}).to_csv(
        prediction_path,
        index=False,
    )
    pd.DataFrame([config]).to_csv(params_path, index=False)
    pd.DataFrame({ID_COL: test_ids, TARGET: np.clip(test_pred, 0, 10)}).to_csv(
        submission_path,
        index=False,
    )

    update_summary(
        {
            "experiment_id": experiment_id,
            "feature_set": feature_set,
            "model_name": MODEL_NAME,
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
        "feature_set": feature_set,
        "config_name": config["name"],
        "cv_rmse": cv_score,
        "fold_mean": float(np.mean(fold_scores)),
        "fold_std": float(np.std(fold_scores)),
        "submission_path": str(submission_path),
    }


def run_catboost_cv(experiment_id, feature_set, config, random_seed):
    train, test = load_data()
    train_fe = apply_feature_set(train, feature_set)
    test_fe = apply_feature_set(test, feature_set)

    feature_cols = [col for col in train_fe.columns if col not in [TARGET, ID_COL]]
    X = train_fe[feature_cols]
    y = train_fe[TARGET]
    X_test = test_fe[feature_cols]

    numeric_features = X.select_dtypes(include="number").columns.tolist()
    categorical_features = X.select_dtypes(exclude="number").columns.tolist()
    feature_order = numeric_features + categorical_features
    cat_indices = [feature_order.index(col) for col in categorical_features]

    X_ordered = X[feature_order]
    X_test_ordered = X_test[feature_order]
    splitter, y_bins = get_cv(y, n_splits=N_SPLITS)

    oof = np.zeros(len(X), dtype=float)
    test_pred = np.zeros(len(X_test), dtype=float)
    fold_rows = []

    print(f"\n=== {experiment_id} ===")
    print("feature_set:", feature_set)
    print("seed:", random_seed)
    print(config)
    print("features:", len(feature_cols), "numeric:", len(numeric_features), "categorical:", len(categorical_features))

    for fold, (train_idx, valid_idx) in enumerate(splitter.split(X_ordered, y_bins), start=1):
        X_train_raw = X_ordered.iloc[train_idx]
        X_valid_raw = X_ordered.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        X_train, X_valid, X_test_prepared = prepare_catboost_frame(
            X_train_raw,
            X_valid_raw,
            X_test_ordered,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
        )

        model = CatBoostRegressor(
            loss_function="RMSE",
            random_seed=random_seed,
            allow_writing_files=False,
            verbose=False,
            od_type="Iter",
            od_wait=120,
            use_best_model=True,
            **catboost_runtime_params(),
            **{key: value for key, value in config.items() if key != "name"},
        )
        model.fit(
            X_train,
            y_train,
            cat_features=cat_indices,
            eval_set=(X_valid, y_valid),
        )
        valid_pred = np.clip(model.predict(X_valid), 0, 10)
        fold_score = rmse(y_valid, valid_pred)

        oof[valid_idx] = valid_pred
        test_pred += np.clip(model.predict(X_test_prepared), 0, 10) / N_SPLITS
        fold_rows.append(
            {
                "experiment_id": experiment_id,
                "feature_set": feature_set,
                "model_name": MODEL_NAME,
                "config_name": config["name"],
                "seed": random_seed,
                "fold": fold,
                "rmse": fold_score,
            }
        )
        print(f"  fold {fold}: {fold_score:.6f}")

    return save_outputs(
        experiment_id=experiment_id,
        feature_set=feature_set,
        train_ids=train_fe[ID_COL],
        test_ids=test_fe[ID_COL],
        y=y,
        oof=oof,
        test_pred=test_pred,
        fold_rows=fold_rows,
        config={**config, "feature_set": feature_set, "seed": random_seed},
    )


def main():
    print("Stage 8 - CatBoost Tuning + Seed Averaging")
    print("Primary feature set:", PRIMARY_FEATURE_SET)
    print("CatBoost task type:", CATBOOST_TASK_TYPE)
    if CATBOOST_TASK_TYPE == "GPU":
        print("CatBoost devices:", CATBOOST_DEVICES)
    print("FAST_MODE:", FAST_MODE)

    configs_to_run = CAT_TUNING_CONFIGS[:3] if FAST_MODE else CAT_TUNING_CONFIGS
    seeds_to_run = SEEDS[:2] if FAST_MODE else SEEDS

    print("config count:", len(configs_to_run))
    print("seed count:", len(seeds_to_run))

    tuning_results = []
    for idx, config in enumerate(configs_to_run, start=1):
        experiment_id = f"exp_8{idx:02d}_cat_tuned_{config['name']}"
        tuning_results.append(
            run_catboost_cv(
                experiment_id=experiment_id,
                feature_set=PRIMARY_FEATURE_SET,
                config=config,
                random_seed=RANDOM_STATE,
            )
        )

    result_df = pd.DataFrame(tuning_results).sort_values("cv_rmse").reset_index(drop=True)
    print("\nCatBoost tuning results:")
    print(result_df[["experiment_id", "config_name", "cv_rmse", "fold_mean", "fold_std", "submission_path"]])

    best_config_name = result_df.iloc[0]["config_name"]
    best_config = next(config for config in configs_to_run if config["name"] == best_config_name)
    print("\nBest config for seed averaging:", best_config_name)

    seed_experiment_ids = []
    for seed in seeds_to_run:
        experiment_id = f"exp_850_cat_seed_{seed}_{best_config_name}"
        run_catboost_cv(
            experiment_id=experiment_id,
            feature_set=PRIMARY_FEATURE_SET,
            config=best_config,
            random_seed=seed,
        )
        seed_experiment_ids.append(experiment_id)

    blend_experiments(
        experiment_id="exp_890_cat_seed_average_blend",
        source_experiment_ids=seed_experiment_ids,
    )

    with_hgb = ["exp_491_hgb_te_tuned_top3_inverse_blend", "exp_890_cat_seed_average_blend"]
    blend_experiments(
        experiment_id="exp_891_hgb_cat_seed_average_blend",
        source_experiment_ids=with_hgb,
    )

    with_feature_blend = [
        "exp_491_hgb_te_tuned_top3_inverse_blend",
        "exp_890_cat_seed_average_blend",
        "exp_790_cat_feature_top3_equal_blend",
    ]
    blend_experiments(
        experiment_id="exp_892_hgb_cat_seed_feature_blend",
        source_experiment_ids=with_feature_blend,
    )


if __name__ == "__main__":
    main()
