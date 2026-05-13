from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from experiment_framework import ID_COL, TARGET, project_root, rmse


RANDOM_STATE = 42
RANDOM_ITER = int(os.getenv("STAGE9_RANDOM_ITER", "120000"))

CORE3_IDS = [
    "exp_491_hgb_te_tuned_top3_inverse_blend",
    "exp_890_cat_seed_average_blend",
    "exp_790_cat_feature_top3_equal_blend",
]

SELECTED_IDS = [
    "exp_491_hgb_te_tuned_top3_inverse_blend",
    "exp_890_cat_seed_average_blend",
    "exp_790_cat_feature_top3_equal_blend",
    "exp_802_cat_tuned_depth5_reg",
    "exp_808_cat_tuned_high_random_strength",
    "exp_605_xgb_te_base",
    "exp_603_lgbm_te_regularized",
]

DIVERSE_IDS = [
    "exp_491_hgb_te_tuned_top3_inverse_blend",
    "exp_890_cat_seed_average_blend",
    "exp_790_cat_feature_top3_equal_blend",
    "exp_802_cat_tuned_depth5_reg",
    "exp_808_cat_tuned_high_random_strength",
    "exp_605_xgb_te_base",
    "exp_604_xgb_ohe_base",
    "exp_603_lgbm_te_regularized",
    "exp_606_cat_native_base",
]


def load_summary() -> pd.DataFrame:
    summary_path = project_root() / "reports" / "experiments" / "summary.csv"
    summary = pd.read_csv(summary_path)
    summary["cv_rmse"] = summary["cv_rmse"].astype(float)
    return summary


def filter_existing_ids(summary: pd.DataFrame, source_ids: list[str]) -> list[str]:
    available = set(summary["experiment_id"])
    existing = [experiment_id for experiment_id in source_ids if experiment_id in available]
    missing = [experiment_id for experiment_id in source_ids if experiment_id not in available]
    if missing:
        print("Skipped missing experiments:")
        for experiment_id in missing:
            print(" ", experiment_id)
    return existing


def load_predictions(summary: pd.DataFrame, source_ids: list[str]):
    oof_frames = []
    pred_frames = []

    for experiment_id in source_ids:
        row = summary.loc[summary["experiment_id"] == experiment_id].iloc[0]
        oof = pd.read_csv(row["oof_path"])[[ID_COL, TARGET, "oof_pred"]].rename(
            columns={"oof_pred": experiment_id}
        )
        pred = pd.read_csv(row["prediction_path"])[[ID_COL, "prediction"]].rename(
            columns={"prediction": experiment_id}
        )
        oof_frames.append(oof)
        pred_frames.append(pred)

    base_oof = oof_frames[0][[ID_COL, TARGET]].copy()
    base_pred = pred_frames[0][[ID_COL]].copy()
    oof_matrix = np.column_stack(
        [frame[experiment_id].to_numpy() for frame, experiment_id in zip(oof_frames, source_ids)]
    )
    pred_matrix = np.column_stack(
        [frame[experiment_id].to_numpy() for frame, experiment_id in zip(pred_frames, source_ids)]
    )
    return base_oof, base_pred, oof_matrix, pred_matrix


def score_weights(y: np.ndarray, oof_matrix: np.ndarray, weights: np.ndarray) -> float:
    pred = np.clip(oof_matrix @ weights, 0, 10)
    return rmse(y, pred)


def save_blend(
    experiment_id: str,
    source_ids: list[str],
    weights: np.ndarray,
    base_oof: pd.DataFrame,
    base_pred: pd.DataFrame,
    oof_matrix: np.ndarray,
    pred_matrix: np.ndarray,
) -> dict:
    root = project_root()
    report_dir = root / "reports" / "experiments"
    submission_dir = root / "submissions"
    report_dir.mkdir(parents=True, exist_ok=True)
    submission_dir.mkdir(exist_ok=True)

    weights = np.array(weights, dtype=float)
    weights = weights / weights.sum()

    oof_pred = np.clip(oof_matrix @ weights, 0, 10)
    test_pred = np.clip(pred_matrix @ weights, 0, 10)
    score = rmse(base_oof[TARGET], oof_pred)

    oof_path = report_dir / f"{experiment_id}_oof.csv"
    prediction_path = report_dir / f"{experiment_id}_test_predictions.csv"
    weights_path = report_dir / f"{experiment_id}_weights.csv"
    submission_path = submission_dir / f"{experiment_id}_blend.csv"

    oof = base_oof.copy()
    oof["oof_pred"] = oof_pred
    pred = base_pred.copy()
    pred["prediction"] = test_pred

    oof.to_csv(oof_path, index=False)
    pred.to_csv(prediction_path, index=False)
    pd.DataFrame({"source_experiment_id": source_ids, "weight": weights}).to_csv(weights_path, index=False)
    pd.DataFrame({ID_COL: pred[ID_COL], TARGET: pred["prediction"]}).to_csv(submission_path, index=False)

    summary_path = report_dir / "summary.csv"
    summary = pd.read_csv(summary_path)
    new_row = pd.DataFrame(
        [
            {
                "experiment_id": experiment_id,
                "feature_set": "blend",
                "model_name": "blend",
                "n_splits": np.nan,
                "cv_rmse": score,
                "fold_mean": np.nan,
                "fold_std": np.nan,
                "oof_path": str(oof_path),
                "prediction_path": str(prediction_path),
                "submission_path": str(submission_path),
            }
        ]
    )
    summary = pd.concat([summary, new_row], ignore_index=True)
    summary = summary.drop_duplicates(subset=["experiment_id"], keep="last")
    summary.sort_values("cv_rmse").to_csv(summary_path, index=False)

    return {
        "experiment_id": experiment_id,
        "cv_rmse": score,
        "submission_path": str(submission_path),
        "weights_path": str(weights_path),
    }


def core3_grid_search(y: np.ndarray, oof_matrix: np.ndarray, step: float = 0.0025):
    best_score = float("inf")
    best_weights = None
    n_steps = int(round(1 / step))

    for i in range(n_steps + 1):
        w1 = i * step
        remaining = 1 - w1
        max_j = int(round(remaining / step))
        for j in range(max_j + 1):
            w2 = j * step
            w3 = 1 - w1 - w2
            if w3 < -1e-12:
                continue
            weights = np.array([w1, w2, w3], dtype=float)
            score = score_weights(y, oof_matrix, weights)
            if score < best_score:
                best_score = score
                best_weights = weights

    return best_score, best_weights


def random_dirichlet_search(
    y: np.ndarray,
    oof_matrix: np.ndarray,
    n_iter: int,
    alpha: float,
    max_weight: float | None = None,
):
    rng = np.random.default_rng(RANDOM_STATE + int(alpha * 1000))
    n_models = oof_matrix.shape[1]
    best_score = float("inf")
    best_weights = None

    for _ in range(n_iter):
        weights = rng.dirichlet(np.ones(n_models) * alpha)
        if max_weight is not None and weights.max() > max_weight:
            continue
        score = score_weights(y, oof_matrix, weights)
        if score < best_score:
            best_score = score
            best_weights = weights

    return best_score, best_weights


def pairwise_refine(
    y: np.ndarray,
    oof_matrix: np.ndarray,
    initial_weights: np.ndarray,
    step_sizes=(0.02, 0.01, 0.005, 0.002, 0.001),
):
    weights = initial_weights.copy().astype(float)
    weights = weights / weights.sum()
    best_score = score_weights(y, oof_matrix, weights)

    for step in step_sizes:
        improved = True
        while improved:
            improved = False
            best_candidate = None
            for source_idx in range(len(weights)):
                if weights[source_idx] < step:
                    continue
                for target_idx in range(len(weights)):
                    if source_idx == target_idx:
                        continue
                    candidate = weights.copy()
                    candidate[source_idx] -= step
                    candidate[target_idx] += step
                    score = score_weights(y, oof_matrix, candidate)
                    if score < best_score - 1e-10:
                        best_candidate = candidate
                        best_score = score
            if best_candidate is not None:
                weights = best_candidate
                improved = True

    return best_score, weights


def run_pool(pool_name: str, experiment_id_prefix: str, source_ids: list[str], summary: pd.DataFrame):
    source_ids = filter_existing_ids(summary, source_ids)
    base_oof, base_pred, oof_matrix, pred_matrix = load_predictions(summary, source_ids)
    y = base_oof[TARGET].to_numpy()

    print(f"\n=== {pool_name} ===")
    for source_id in source_ids:
        score = summary.loc[summary["experiment_id"] == source_id, "cv_rmse"].iloc[0]
        print(f"  {source_id}: {score:.6f}")

    results = []

    equal_weights = np.ones(len(source_ids)) / len(source_ids)
    results.append(
        save_blend(
            experiment_id=f"{experiment_id_prefix}_equal_blend",
            source_ids=source_ids,
            weights=equal_weights,
            base_oof=base_oof,
            base_pred=base_pred,
            oof_matrix=oof_matrix,
            pred_matrix=pred_matrix,
        )
    )

    best_random_score = float("inf")
    best_random_weights = None
    for alpha in [0.4, 0.8, 1.5, 3.0]:
        score, weights = random_dirichlet_search(
            y=y,
            oof_matrix=oof_matrix,
            n_iter=RANDOM_ITER,
            alpha=alpha,
            max_weight=0.70,
        )
        print(f"  random alpha={alpha}: {score:.6f}")
        if score < best_random_score:
            best_random_score = score
            best_random_weights = weights

    results.append(
        save_blend(
            experiment_id=f"{experiment_id_prefix}_random_blend",
            source_ids=source_ids,
            weights=best_random_weights,
            base_oof=base_oof,
            base_pred=base_pred,
            oof_matrix=oof_matrix,
            pred_matrix=pred_matrix,
        )
    )

    refined_score, refined_weights = pairwise_refine(
        y=y,
        oof_matrix=oof_matrix,
        initial_weights=best_random_weights,
    )
    print(f"  pairwise refined: {refined_score:.6f}")
    results.append(
        save_blend(
            experiment_id=f"{experiment_id_prefix}_refined_blend",
            source_ids=source_ids,
            weights=refined_weights,
            base_oof=base_oof,
            base_pred=base_pred,
            oof_matrix=oof_matrix,
            pred_matrix=pred_matrix,
        )
    )

    return results


def main():
    summary = load_summary()
    all_results = []

    core_ids = filter_existing_ids(summary, CORE3_IDS)
    base_oof, base_pred, oof_matrix, pred_matrix = load_predictions(summary, core_ids)
    y = base_oof[TARGET].to_numpy()

    print("Stage 9 - Selected Blend Optimization")
    print("RANDOM_ITER:", RANDOM_ITER)
    print("\n=== core3_grid ===")
    grid_score, grid_weights = core3_grid_search(y, oof_matrix)
    print(f"core3 grid: {grid_score:.6f}")
    all_results.append(
        save_blend(
            experiment_id="exp_900_core3_grid_blend",
            source_ids=core_ids,
            weights=grid_weights,
            base_oof=base_oof,
            base_pred=base_pred,
            oof_matrix=oof_matrix,
            pred_matrix=pred_matrix,
        )
    )

    all_results.extend(run_pool("core3_random", "exp_901_core3", CORE3_IDS, summary))
    all_results.extend(run_pool("selected_random", "exp_902_selected", SELECTED_IDS, summary))
    all_results.extend(run_pool("diverse_random", "exp_903_diverse", DIVERSE_IDS, summary))

    result_df = pd.DataFrame(all_results).sort_values("cv_rmse")
    print("\nStage 9 results:")
    print(result_df[["experiment_id", "cv_rmse", "submission_path", "weights_path"]])


if __name__ == "__main__":
    main()
