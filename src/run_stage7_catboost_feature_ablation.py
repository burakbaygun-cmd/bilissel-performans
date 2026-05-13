from __future__ import annotations

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
    available_feature_sets,
    blend_experiments,
    get_cv,
    load_data,
    project_root,
    rmse,
)


N_SPLITS = 5

CAT_BASE_CONFIG = {
    "iterations": 1600,
    "learning_rate": 0.025,
    "depth": 6,
    "l2_leaf_reg": 4.0,
    "random_strength": 1.0,
    "bagging_temperature": 1.0,
}


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


def run_catboost_feature_set(feature_set: str, idx: int):
    root = project_root()
    report_dir = root / "reports" / "experiments"
    submission_dir = root / "submissions"
    report_dir.mkdir(parents=True, exist_ok=True)
    submission_dir.mkdir(exist_ok=True)

    experiment_id = f"exp_7{idx:02d}_cat_native_base_{feature_set}"
    model_name = "cat_native_base"

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
            random_seed=RANDOM_STATE,
            allow_writing_files=False,
            verbose=False,
            **CAT_BASE_CONFIG,
        )
        model.fit(X_train, y_train, cat_features=cat_indices)
        valid_pred = np.clip(model.predict(X_valid), 0, 10)
        fold_score = rmse(y_valid, valid_pred)

        oof[valid_idx] = valid_pred
        test_pred += np.clip(model.predict(X_test_prepared), 0, 10) / N_SPLITS
        fold_rows.append(
            {
                "experiment_id": experiment_id,
                "feature_set": feature_set,
                "model_name": model_name,
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
    submission_path = submission_dir / f"{experiment_id}_{model_name}_{feature_set}.csv"
    params_path = report_dir / f"{experiment_id}_params.csv"

    pd.DataFrame(fold_rows).to_csv(fold_path, index=False)
    pd.DataFrame({ID_COL: train_fe[ID_COL], TARGET: y, "oof_pred": np.clip(oof, 0, 10)}).to_csv(
        oof_path,
        index=False,
    )
    pd.DataFrame({ID_COL: test_fe[ID_COL], "prediction": np.clip(test_pred, 0, 10)}).to_csv(
        prediction_path,
        index=False,
    )
    pd.DataFrame([{**CAT_BASE_CONFIG, "feature_set": feature_set}]).to_csv(params_path, index=False)
    pd.DataFrame({ID_COL: test_fe[ID_COL], TARGET: np.clip(test_pred, 0, 10)}).to_csv(
        submission_path,
        index=False,
    )

    update_summary(
        {
            "experiment_id": experiment_id,
            "feature_set": feature_set,
            "model_name": model_name,
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
        "cv_rmse": cv_score,
        "submission_path": str(submission_path),
    }


def main():
    feature_sets = available_feature_sets()
    print("Stage 7 - CatBoost Feature Set Ablation")
    print("Feature sets:", feature_sets)

    results = []
    for idx, feature_set in enumerate(feature_sets, start=1):
        results.append(run_catboost_feature_set(feature_set=feature_set, idx=idx))

    result_df = pd.DataFrame(results).sort_values("cv_rmse").reset_index(drop=True)
    print("\nCatBoost feature set results:")
    print(result_df)

    top3_ids = result_df.head(3)["experiment_id"].tolist()
    print("\nTop 3 CatBoost feature sets:", top3_ids)
    blend_experiments(
        experiment_id="exp_790_cat_feature_top3_equal_blend",
        source_experiment_ids=top3_ids,
    )

    with_hgb = ["exp_491_hgb_te_tuned_top3_inverse_blend"] + top3_ids
    blend_experiments(
        experiment_id="exp_791_hgb_cat_feature_top3_equal_blend",
        source_experiment_ids=with_hgb,
    )


if __name__ == "__main__":
    main()

