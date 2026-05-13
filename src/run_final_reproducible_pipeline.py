from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from experiment_framework import blend_experiments, project_root
from run_stage4_hgb_te_tuning import main as run_stage4_hgb_te_tuning
from run_stage6_external_gbdt_models import main as run_stage6_external_gbdt_models
from run_stage7_catboost_feature_ablation import run_catboost_feature_set
from run_stage8_catboost_tuning_seed import main as run_stage8_catboost_tuning_seed
from run_stage9_selected_blend_optimization import main as run_stage9_selected_blend_optimization


FINAL_SUBMISSION_NAME = "exp_903_diverse_refined_blend_blend.csv"


def run_required_catboost_feature_blend() -> None:
    """Recreate the exact CatBoost feature blend used by the final Stage 9 blend."""

    required_feature_runs = [
        ("fs_01_clean_country", 2),          # exp_702
        ("fs_02_missing_indicators", 3),    # exp_703
        ("fs_06_bins", 7),                  # exp_707
    ]

    for feature_set, original_index in required_feature_runs:
        run_catboost_feature_set(feature_set=feature_set, idx=original_index)

    blend_experiments(
        experiment_id="exp_790_cat_feature_top3_equal_blend",
        source_experiment_ids=[
            "exp_703_cat_native_base_fs_02_missing_indicators",
            "exp_707_cat_native_base_fs_06_bins",
            "exp_702_cat_native_base_fs_01_clean_country",
        ],
    )


def copy_final_submission() -> Path:
    root = project_root()
    src = root / "submissions" / FINAL_SUBMISSION_NAME
    dst = root / "submissions" / "final_submission.csv"

    if not src.exists():
        raise FileNotFoundError(f"Final submission was not produced: {src}")

    shutil.copyfile(src, dst)
    return dst


def main() -> None:
    root = project_root()
    (root / "reports" / "experiments").mkdir(parents=True, exist_ok=True)
    (root / "submissions").mkdir(parents=True, exist_ok=True)

    print("Final reproducible pipeline")
    print("Project root:", root)

    print("\n[1/5] HGB target encoding tuning")
    run_stage4_hgb_te_tuning()

    print("\n[2/5] External GBDT models")
    run_stage6_external_gbdt_models()

    print("\n[3/5] Required CatBoost feature blend")
    run_required_catboost_feature_blend()

    print("\n[4/5] CatBoost tuning + seed averaging")
    run_stage8_catboost_tuning_seed()

    print("\n[5/5] Selected blend optimization")
    run_stage9_selected_blend_optimization()

    final_path = copy_final_submission()
    print("\nFinal submission written to:")
    print(final_path)


if __name__ == "__main__":
    main()
