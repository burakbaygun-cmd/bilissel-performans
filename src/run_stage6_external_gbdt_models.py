from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sklearn.base import clone
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
N_SPLITS = 5


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


def save_experiment_outputs(
    experiment_id,
    model_name,
    train_ids,
    test_ids,
    y,
    oof,
    test_pred,
    fold_rows,
    config,
):
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
    submission_path = submission_dir / f"{experiment_id}_{model_name}_{FEATURE_SET}.csv"
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
            "feature_set": FEATURE_SET,
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
        "model_name": model_name,
        "cv_rmse": cv_score,
        "fold_mean": float(np.mean(fold_scores)),
        "fold_std": float(np.std(fold_scores)),
        "submission_path": str(submission_path),
    }


def build_pipeline_model(config, numeric_features, categorical_features):
    model_name = config["model_name"]
    encoding = config["encoding"]

    if model_name.startswith("lgbm"):
        from lightgbm import LGBMRegressor

        estimator = LGBMRegressor(
            objective="regression",
            n_estimators=config["n_estimators"],
            learning_rate=config["learning_rate"],
            num_leaves=config["num_leaves"],
            max_depth=config["max_depth"],
            min_child_samples=config["min_child_samples"],
            subsample=config["subsample"],
            colsample_bytree=config["colsample_bytree"],
            reg_alpha=config["reg_alpha"],
            reg_lambda=config["reg_lambda"],
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
            force_col_wise=True,
        )
    elif model_name.startswith("xgb"):
        from xgboost import XGBRegressor

        estimator = XGBRegressor(
            objective="reg:squarederror",
            eval_metric="rmse",
            n_estimators=config["n_estimators"],
            learning_rate=config["learning_rate"],
            max_depth=config["max_depth"],
            min_child_weight=config["min_child_weight"],
            subsample=config["subsample"],
            colsample_bytree=config["colsample_bytree"],
            reg_alpha=config["reg_alpha"],
            reg_lambda=config["reg_lambda"],
            random_state=RANDOM_STATE,
            n_jobs=-1,
            tree_method="hist",
        )
    else:
        raise ValueError(f"Unsupported pipeline model: {model_name}")

    preprocess = make_preprocessor(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        encoding=encoding,
        dense=False if encoding == "onehot" else True,
        smoothing=config.get("smoothing", 25.0),
    )
    return Pipeline(steps=[("preprocess", preprocess), ("model", estimator)])


def run_pipeline_config(config, train_fe, test_fe, X, y, X_test, numeric_features, categorical_features, splitter, y_bins):
    experiment_id = config["experiment_id"]
    model_name = config["model_name"]
    print(f"\n=== {experiment_id} ===")
    print(config)

    oof = np.zeros(len(X), dtype=float)
    test_pred = np.zeros(len(X_test), dtype=float)
    fold_rows = []

    for fold, (train_idx, valid_idx) in enumerate(splitter.split(X, y_bins), start=1):
        model = build_pipeline_model(config, numeric_features, categorical_features)
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
                "model_name": model_name,
                "fold": fold,
                "rmse": fold_score,
            }
        )
        print(f"  fold {fold}: {fold_score:.6f}")

    return save_experiment_outputs(
        experiment_id=experiment_id,
        model_name=model_name,
        train_ids=train_fe[ID_COL],
        test_ids=test_fe[ID_COL],
        y=y,
        oof=oof,
        test_pred=test_pred,
        fold_rows=fold_rows,
        config=config,
    )


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


def run_catboost_native_config(config, train_fe, test_fe, X, y, X_test, numeric_features, categorical_features, splitter, y_bins):
    from catboost import CatBoostRegressor

    experiment_id = config["experiment_id"]
    model_name = config["model_name"]
    print(f"\n=== {experiment_id} ===")
    print(config)

    feature_order = numeric_features + categorical_features
    cat_indices = [feature_order.index(col) for col in categorical_features]

    X_ordered = X[feature_order]
    X_test_ordered = X_test[feature_order]

    oof = np.zeros(len(X), dtype=float)
    test_pred = np.zeros(len(X_test), dtype=float)
    fold_rows = []

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
            iterations=config["iterations"],
            learning_rate=config["learning_rate"],
            depth=config["depth"],
            l2_leaf_reg=config["l2_leaf_reg"],
            random_strength=config["random_strength"],
            bagging_temperature=config["bagging_temperature"],
            random_seed=RANDOM_STATE,
            allow_writing_files=False,
            verbose=False,
        )
        model.fit(X_train, y_train, cat_features=cat_indices)
        valid_pred = np.clip(model.predict(X_valid), 0, 10)
        fold_score = rmse(y_valid, valid_pred)

        oof[valid_idx] = valid_pred
        test_pred += np.clip(model.predict(X_test_prepared), 0, 10) / N_SPLITS
        fold_rows.append(
            {
                "experiment_id": experiment_id,
                "feature_set": FEATURE_SET,
                "model_name": model_name,
                "fold": fold,
                "rmse": fold_score,
            }
        )
        print(f"  fold {fold}: {fold_score:.6f}")

    return save_experiment_outputs(
        experiment_id=experiment_id,
        model_name=model_name,
        train_ids=train_fe[ID_COL],
        test_ids=test_fe[ID_COL],
        y=y,
        oof=oof,
        test_pred=test_pred,
        fold_rows=fold_rows,
        config=config,
    )


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

    print("Stage 6 - External GBDT Models")
    print("Feature set:", FEATURE_SET)
    print("features:", len(feature_cols), "numeric:", len(numeric_features), "categorical:", len(categorical_features))

    pipeline_configs = [
        {
            "experiment_id": "exp_601_lgbm_ohe_base",
            "model_name": "lgbm_ohe_base",
            "encoding": "onehot",
            "n_estimators": 1800,
            "learning_rate": 0.025,
            "num_leaves": 31,
            "max_depth": -1,
            "min_child_samples": 25,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "reg_alpha": 0.0,
            "reg_lambda": 0.2,
        },
        {
            "experiment_id": "exp_602_lgbm_te_base",
            "model_name": "lgbm_te_base",
            "encoding": "target",
            "smoothing": 25.0,
            "n_estimators": 1800,
            "learning_rate": 0.025,
            "num_leaves": 31,
            "max_depth": -1,
            "min_child_samples": 25,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "reg_alpha": 0.0,
            "reg_lambda": 0.2,
        },
        {
            "experiment_id": "exp_603_lgbm_te_regularized",
            "model_name": "lgbm_te_regularized",
            "encoding": "target",
            "smoothing": 80.0,
            "n_estimators": 2200,
            "learning_rate": 0.018,
            "num_leaves": 24,
            "max_depth": -1,
            "min_child_samples": 45,
            "subsample": 0.80,
            "colsample_bytree": 0.80,
            "reg_alpha": 0.05,
            "reg_lambda": 0.8,
        },
        {
            "experiment_id": "exp_604_xgb_ohe_base",
            "model_name": "xgb_ohe_base",
            "encoding": "onehot",
            "n_estimators": 1100,
            "learning_rate": 0.025,
            "max_depth": 4,
            "min_child_weight": 8,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "reg_alpha": 0.0,
            "reg_lambda": 1.5,
        },
        {
            "experiment_id": "exp_605_xgb_te_base",
            "model_name": "xgb_te_base",
            "encoding": "target",
            "smoothing": 25.0,
            "n_estimators": 1100,
            "learning_rate": 0.025,
            "max_depth": 4,
            "min_child_weight": 8,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "reg_alpha": 0.0,
            "reg_lambda": 1.5,
        },
    ]

    catboost_configs = [
        {
            "experiment_id": "exp_606_cat_native_base",
            "model_name": "cat_native_base",
            "iterations": 1600,
            "learning_rate": 0.025,
            "depth": 6,
            "l2_leaf_reg": 4.0,
            "random_strength": 1.0,
            "bagging_temperature": 1.0,
        },
        {
            "experiment_id": "exp_607_cat_native_depth5",
            "model_name": "cat_native_depth5",
            "iterations": 1800,
            "learning_rate": 0.025,
            "depth": 5,
            "l2_leaf_reg": 6.0,
            "random_strength": 1.0,
            "bagging_temperature": 0.8,
        },
        {
            "experiment_id": "exp_608_cat_native_depth7_reg",
            "model_name": "cat_native_depth7_reg",
            "iterations": 1400,
            "learning_rate": 0.025,
            "depth": 7,
            "l2_leaf_reg": 8.0,
            "random_strength": 1.5,
            "bagging_temperature": 1.0,
        },
    ]

    results = []
    for config in pipeline_configs:
        results.append(
            run_pipeline_config(
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

    for config in catboost_configs:
        results.append(
            run_catboost_native_config(
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

    result_df = pd.DataFrame(results).sort_values("cv_rmse")
    print("\nExternal model results:")
    print(result_df[["experiment_id", "model_name", "cv_rmse", "fold_mean", "fold_std", "submission_path"]])

    top3_external = result_df.head(3)["experiment_id"].tolist()
    print("\nTop 3 external:", top3_external)
    blend_experiments(
        experiment_id="exp_690_external_top3_equal_blend",
        source_experiment_ids=top3_external,
    )

    with_hgb = ["exp_491_hgb_te_tuned_top3_inverse_blend"] + top3_external
    blend_experiments(
        experiment_id="exp_691_hgb_external_top3_equal_blend",
        source_experiment_ids=with_hgb,
    )

    summary = pd.read_csv(project_root() / "reports" / "experiments" / "summary.csv")
    scores = []
    for experiment_id in with_hgb:
        score = float(summary.loc[summary["experiment_id"] == experiment_id, "cv_rmse"].iloc[0])
        scores.append(score)
    inv = 1 / np.array(scores)
    inv_weights = (inv / inv.sum()).tolist()
    blend_experiments(
        experiment_id="exp_692_hgb_external_top3_inverse_blend",
        source_experiment_ids=with_hgb,
        weights=inv_weights,
    )


if __name__ == "__main__":
    main()

