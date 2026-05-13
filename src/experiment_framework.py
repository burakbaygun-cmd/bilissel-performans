from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET = "bilissel_performans_skoru"
ID_COL = "id"
RANDOM_STATE = 42


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def package_exists(package_name: str) -> bool:
    return importlib.util.find_spec(package_name) is not None


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_data():
    root = project_root()
    train = pd.read_csv(root / "data" / "train.csv")
    test = pd.read_csv(root / "data" / "test_x.csv")
    return train, test


class TargetMeanEncoder(BaseEstimator, TransformerMixin):
    """Cross-validation-safe target mean encoder when used inside sklearn pipelines."""

    def __init__(self, smoothing: float = 25.0):
        self.smoothing = smoothing

    def fit(self, X, y):
        X = pd.DataFrame(X).copy().reset_index(drop=True)
        y = pd.Series(y).reset_index(drop=True)

        self.columns_ = X.columns.tolist()
        self.global_mean_ = float(y.mean())
        self.maps_ = {}

        for col in self.columns_:
            values = X[col].astype("object").where(X[col].notna(), "__MISSING__")
            stats = (
                pd.DataFrame({"value": values, "target": y})
                .groupby("value")["target"]
                .agg(["mean", "count"])
            )
            smooth_mean = (
                stats["mean"] * stats["count"] + self.global_mean_ * self.smoothing
            ) / (stats["count"] + self.smoothing)
            self.maps_[col] = smooth_mean.to_dict()

        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        if len(X.columns) == len(self.columns_):
            X.columns = self.columns_

        encoded_cols = []
        for col in self.columns_:
            values = X[col].astype("object").where(X[col].notna(), "__MISSING__")
            encoded = values.map(self.maps_[col]).fillna(self.global_mean_).astype(float)
            encoded_cols.append(encoded.to_numpy())

        return np.column_stack(encoded_cols)


def normalize_country(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ulke_map = {
        "South Korea": "Guney Kore",
        "Spain": "Ispanya",
        "Sweden": "Isvec",
    }
    df["ulke"] = df["ulke"].replace(ulke_map)
    return df


def add_missing_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    missing_cols = [
        "meslek",
        "vucut_kitle_indeksi",
        "uyku_oncesi_kafein_mg",
        "stres_skoru",
        "kronotip",
        "ruh_sagligi_durumu",
    ]
    for col in missing_cols:
        df[f"{col}_missing"] = df[col].isna().astype(int)
    return df


def add_sleep_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["toplam_kaliteli_uyku_yuzdesi"] = df["rem_yuzdesi"] + df["derin_uyku_yuzdesi"]
    df["hafif_uyku_yuzdesi"] = 100 - df["rem_yuzdesi"] - df["derin_uyku_yuzdesi"]
    df["uyku_kalite_orani"] = df["toplam_kaliteli_uyku_yuzdesi"] / (
        df["hafif_uyku_yuzdesi"].clip(lower=1) + 1
    )
    df["rem_derin_orani"] = df["rem_yuzdesi"] / (df["derin_uyku_yuzdesi"] + 1)
    df["derin_rem_orani"] = df["derin_uyku_yuzdesi"] / (df["rem_yuzdesi"] + 1)
    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["kaliteli_uyku_stres_orani"] = (
        df["rem_yuzdesi"] + df["derin_uyku_yuzdesi"]
    ) / (df["stres_skoru"] + 1)
    df["stres_calisma_etkilesimi"] = df["stres_skoru"] * df["gunluk_calisma_saati"]
    df["stres_uyanma_etkilesimi"] = df["stres_skoru"] * df["gecelik_uyanma_sayisi"]
    df["stres_gecikme_etkilesimi"] = df["stres_skoru"] * df["uykuya_dalma_suresi_dk"]
    df["uyku_bolunme_yuku"] = df["gecelik_uyanma_sayisi"] * df["uykuya_dalma_suresi_dk"]
    df["ekran_kafein_etkilesimi"] = (
        df["uyku_oncesi_ekran_suresi_dk"] * df["uyku_oncesi_kafein_mg"]
    )
    df["aktivite_stres_orani"] = df["gunluk_adim_sayisi"] / (df["stres_skoru"] + 1)
    df["adim_calisma_orani"] = df["gunluk_adim_sayisi"] / (df["gunluk_calisma_saati"] + 1)
    df["uyku_hijyen_riski"] = (
        df["uyku_oncesi_ekran_suresi_dk"] / 60
        + df["uyku_oncesi_kafein_mg"] / 100
        + df["uykuya_dalma_suresi_dk"] / 30
        + df["gecelik_uyanma_sayisi"]
    )
    return df


def add_health_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hafta_sonu_mu"] = (df["gun_tipi"] == "Hafta sonu").astype(int)
    df["saglikli_mi"] = (df["ruh_sagligi_durumu"] == "Saglikli").astype(int)
    df["depresyon_var_mi"] = df["ruh_sagligi_durumu"].isin(
        ["Depresyon", "Anksiyete ve depresyon"]
    ).astype(int)
    df["anksiyete_var_mi"] = df["ruh_sagligi_durumu"].isin(
        ["Anksiyete", "Anksiyete ve depresyon"]
    ).astype(int)
    df["sabah_insani_mi"] = (df["kronotip"] == "Sabah insani").astype(int)
    df["gece_insani_mi"] = (df["kronotip"] == "Gece insani").astype(int)
    return df


def add_binned_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["yas_grup"] = pd.cut(
        df["yas"],
        bins=[17, 24, 34, 44, 54, 69, 120],
        labels=["18-24", "25-34", "35-44", "45-54", "55-69", "70+"],
    ).astype("object")
    df["bmi_grup"] = pd.cut(
        df["vucut_kitle_indeksi"],
        bins=[0, 18.5, 25, 30, 35, 100],
        labels=["zayif", "normal", "fazla_kilo", "obez_1", "obez_2"],
    ).astype("object")
    df["stres_grup"] = pd.cut(
        df["stres_skoru"],
        bins=[0, 3, 6, 8, 10],
        labels=["dusuk", "orta", "yuksek", "cok_yuksek"],
        include_lowest=True,
    ).astype("object")
    df["calisma_grup"] = pd.cut(
        df["gunluk_calisma_saati"],
        bins=[-1, 4, 8, 12, 24],
        labels=["az", "normal", "uzun", "cok_uzun"],
    ).astype("object")
    df["ekran_grup"] = pd.cut(
        df["uyku_oncesi_ekran_suresi_dk"],
        bins=[0, 30, 60, 120, 240],
        labels=["dusuk", "orta", "yuksek", "cok_yuksek"],
        include_lowest=True,
    ).astype("object")
    df["adim_grup"] = pd.cut(
        df["gunluk_adim_sayisi"],
        bins=[0, 3000, 7000, 10000, 30000],
        labels=["dusuk", "orta", "iyi", "yuksek"],
        include_lowest=True,
    ).astype("object")
    return df


def _combo_value(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    values = []
    for col in cols:
        values.append(df[col].astype("object").where(df[col].notna(), "__MISSING__").astype(str))
    return values[0].str.cat(values[1:], sep="__")


def add_forensics_group_interactions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    combos = [
        ["meslek", "ruh_sagligi_durumu"],
        ["meslek", "ruh_sagligi_durumu", "gun_tipi"],
        ["meslek", "ruh_sagligi_durumu", "mevsim"],
        ["cinsiyet", "meslek", "ruh_sagligi_durumu"],
        ["meslek", "kronotip", "ruh_sagligi_durumu"],
        ["meslek", "ulke", "ruh_sagligi_durumu"],
        ["kronotip", "ruh_sagligi_durumu", "gun_tipi"],
        ["ruh_sagligi_durumu", "gun_tipi"],
        ["ruh_sagligi_durumu", "mevsim", "gun_tipi"],
        ["meslek", "gun_tipi"],
        ["meslek", "mevsim", "gun_tipi"],
        ["meslek", "kronotip", "gun_tipi"],
        ["ulke", "ruh_sagligi_durumu", "gun_tipi"],
    ]
    for cols in combos:
        name = "combo__" + "__".join(cols)
        df[name] = _combo_value(df, cols)
    return df


def add_forensics_risk_bins(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["stres_detay_grup"] = pd.cut(
        df["stres_skoru"],
        bins=[-0.01, 3.0, 5.0, 6.2, 6.62, 7.1, 7.74, 10.0],
        labels=["s_0_3", "s_3_5", "s_5_62", "s_62_66", "s_66_71", "s_71_77", "s_77_10"],
        include_lowest=True,
    ).astype("object")
    df["calisma_detay_grup"] = pd.cut(
        df["gunluk_calisma_saati"],
        bins=[-0.01, 4.0, 7.36, 8.28, 9.21, 10.22, 11.52, 24.0],
        labels=["c_0_4", "c_4_736", "c_736_828", "c_828_921", "c_921_1022", "c_1022_1152", "c_1152_plus"],
        include_lowest=True,
    ).astype("object")
    df["rem_detay_grup"] = pd.cut(
        df["rem_yuzdesi"],
        bins=[-0.01, 15.78, 17.36, 18.48, 19.44, 20.30, 23.0, 100.0],
        labels=["r_low", "r_158_174", "r_174_185", "r_185_194", "r_194_203", "r_203_23", "r_high"],
        include_lowest=True,
    ).astype("object")
    df["uykuya_dalma_detay_grup"] = pd.cut(
        df["uykuya_dalma_suresi_dk"],
        bins=[-0.01, 10, 15, 20, 24, 27, 31, 60, 240],
        labels=["lat_0_10", "lat_10_15", "lat_15_20", "lat_20_24", "lat_24_27", "lat_27_31", "lat_31_60", "lat_60_plus"],
        include_lowest=True,
    ).astype("object")
    df["uyanma_detay_grup"] = pd.cut(
        df["gecelik_uyanma_sayisi"],
        bins=[-0.01, 1, 2, 3, 4, 5, 6, 8, 20],
        labels=["w_0_1", "w_1_2", "w_2_3", "w_3_4", "w_4_5", "w_5_6", "w_6_8", "w_8_plus"],
        include_lowest=True,
    ).astype("object")
    df["kafein_detay_grup"] = pd.cut(
        df["uyku_oncesi_kafein_mg"],
        bins=[-0.01, 0, 25, 75, 150, 400, 1000],
        labels=["caf_0", "caf_1_25", "caf_25_75", "caf_75_150", "caf_150_400", "caf_400_plus"],
        include_lowest=True,
    ).astype("object")
    return df


def add_forensics_risk_interactions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    combos = [
        ["ruh_sagligi_durumu", "stres_detay_grup"],
        ["meslek", "stres_detay_grup"],
        ["meslek", "ruh_sagligi_durumu", "stres_detay_grup"],
        ["meslek", "calisma_detay_grup"],
        ["gun_tipi", "calisma_detay_grup"],
        ["ruh_sagligi_durumu", "calisma_detay_grup"],
        ["stres_detay_grup", "calisma_detay_grup"],
        ["stres_detay_grup", "uyanma_detay_grup"],
        ["ruh_sagligi_durumu", "uyanma_detay_grup"],
        ["rem_detay_grup", "stres_detay_grup"],
        ["uykuya_dalma_detay_grup", "stres_detay_grup"],
        ["kafein_detay_grup", "stres_detay_grup"],
    ]
    for cols in combos:
        name = "combo__" + "__".join(cols)
        df[name] = _combo_value(df, cols)
    return df


FEATURE_SET_STEPS = {
    "fs_00_raw": [],
    "fs_01_clean_country": [normalize_country],
    "fs_02_missing_indicators": [normalize_country, add_missing_indicators],
    "fs_03_sleep_features": [normalize_country, add_sleep_features],
    "fs_04_interactions": [normalize_country, add_sleep_features, add_interaction_features],
    "fs_05_health_flags": [normalize_country, add_health_flags],
    "fs_06_bins": [normalize_country, add_binned_features],
    "fs_07_all_safe": [
        normalize_country,
        add_missing_indicators,
        add_sleep_features,
        add_interaction_features,
        add_health_flags,
        add_binned_features,
    ],
    "fs_08_forensic_combos": [
        normalize_country,
        add_missing_indicators,
        add_forensics_group_interactions,
    ],
    "fs_09_forensic_risk_bins": [
        normalize_country,
        add_missing_indicators,
        add_forensics_risk_bins,
    ],
    "fs_10_forensic_combo_risk": [
        normalize_country,
        add_missing_indicators,
        add_forensics_group_interactions,
        add_forensics_risk_bins,
        add_forensics_risk_interactions,
    ],
}


def apply_feature_set(df: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    if feature_set not in FEATURE_SET_STEPS:
        valid = ", ".join(FEATURE_SET_STEPS)
        raise ValueError(f"Unknown feature_set={feature_set}. Valid values: {valid}")

    out = df.copy()
    for step in FEATURE_SET_STEPS[feature_set]:
        out = step(out)
    return out


def make_onehot_encoder(dense: bool):
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=not dense)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=not dense)


def make_preprocessor(
    numeric_features,
    categorical_features,
    encoding: str,
    dense: bool,
    smoothing: float = 25.0,
    scale_numeric: bool = False,
):
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    if encoding == "onehot":
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="Bilinmiyor")),
                ("one_hot", make_onehot_encoder(dense=dense)),
            ]
        )
    elif encoding == "target":
        categorical_pipeline = TargetMeanEncoder(smoothing=smoothing)
    else:
        raise ValueError("encoding must be 'onehot' or 'target'")

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=numeric_steps), numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        sparse_threshold=0.0 if dense else 0.3,
    )


@dataclass(frozen=True)
class ModelConfig:
    name: str
    encoding: str
    dense: bool
    estimator: object
    smoothing: float = 25.0
    scale_numeric: bool = False


def get_model_config(model_name: str) -> ModelConfig:
    configs = {
        "ridge_ohe": ModelConfig(
            name="ridge_ohe",
            encoding="onehot",
            dense=False,
            scale_numeric=True,
            estimator=Ridge(alpha=5.0, random_state=RANDOM_STATE),
        ),
        "hgb_ohe_base": ModelConfig(
            name="hgb_ohe_base",
            encoding="onehot",
            dense=True,
            estimator=HistGradientBoostingRegressor(
                max_iter=700,
                learning_rate=0.035,
                max_leaf_nodes=31,
                min_samples_leaf=20,
                l2_regularization=0.05,
                random_state=RANDOM_STATE,
            ),
        ),
        "hgb_ohe_compact": ModelConfig(
            name="hgb_ohe_compact",
            encoding="onehot",
            dense=True,
            estimator=HistGradientBoostingRegressor(
                max_iter=650,
                learning_rate=0.045,
                max_leaf_nodes=15,
                min_samples_leaf=20,
                l2_regularization=0.03,
                random_state=RANDOM_STATE,
            ),
        ),
        "hgb_te_base": ModelConfig(
            name="hgb_te_base",
            encoding="target",
            dense=True,
            smoothing=25.0,
            estimator=HistGradientBoostingRegressor(
                max_iter=800,
                learning_rate=0.03,
                max_leaf_nodes=31,
                min_samples_leaf=20,
                l2_regularization=0.05,
                random_state=RANDOM_STATE,
            ),
        ),
        "hgb_te_smooth": ModelConfig(
            name="hgb_te_smooth",
            encoding="target",
            dense=True,
            smoothing=80.0,
            estimator=HistGradientBoostingRegressor(
                max_iter=900,
                learning_rate=0.025,
                max_leaf_nodes=31,
                min_samples_leaf=30,
                l2_regularization=0.20,
                random_state=RANDOM_STATE,
            ),
        ),
        "extra_trees_ohe": ModelConfig(
            name="extra_trees_ohe",
            encoding="onehot",
            dense=False,
            estimator=ExtraTreesRegressor(
                n_estimators=650,
                min_samples_leaf=2,
                max_features=0.85,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
        "random_forest_ohe": ModelConfig(
            name="random_forest_ohe",
            encoding="onehot",
            dense=False,
            estimator=RandomForestRegressor(
                n_estimators=500,
                min_samples_leaf=3,
                max_features=0.85,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
    }

    if model_name in configs:
        return configs[model_name]

    if model_name == "lgbm_ohe" and package_exists("lightgbm"):
        from lightgbm import LGBMRegressor

        return ModelConfig(
            name="lgbm_ohe",
            encoding="onehot",
            dense=False,
            estimator=LGBMRegressor(
                n_estimators=1500,
                learning_rate=0.025,
                num_leaves=31,
                min_child_samples=25,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_lambda=0.2,
                objective="regression",
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbose=-1,
            ),
        )

    if model_name == "xgb_ohe" and package_exists("xgboost"):
        from xgboost import XGBRegressor

        return ModelConfig(
            name="xgb_ohe",
            encoding="onehot",
            dense=False,
            estimator=XGBRegressor(
                n_estimators=1200,
                learning_rate=0.025,
                max_depth=4,
                min_child_weight=8,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_lambda=1.5,
                objective="reg:squarederror",
                eval_metric="rmse",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        )

    if model_name == "catboost_ohe" and package_exists("catboost"):
        from catboost import CatBoostRegressor

        return ModelConfig(
            name="catboost_ohe",
            encoding="onehot",
            dense=True,
            estimator=CatBoostRegressor(
                iterations=1200,
                learning_rate=0.025,
                depth=6,
                l2_leaf_reg=4.0,
                loss_function="RMSE",
                random_seed=RANDOM_STATE,
                verbose=False,
            ),
        )

    raise ValueError(f"Unknown or unavailable model_name={model_name}")


def build_pipeline(model_name: str, numeric_features, categorical_features) -> Pipeline:
    config = get_model_config(model_name)
    preprocess = make_preprocessor(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        encoding=config.encoding,
        dense=config.dense,
        smoothing=config.smoothing,
        scale_numeric=config.scale_numeric,
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", config.estimator),
        ]
    )


def get_cv(y, n_splits: int = 5):
    y_bins = pd.qcut(y, q=10, labels=False, duplicates="drop")
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    return splitter, y_bins


@dataclass
class ExperimentOutput:
    experiment_id: str
    feature_set: str
    model_name: str
    cv_rmse: float
    fold_scores: list[float]
    oof_path: Path
    prediction_path: Path
    submission_path: Path


def run_cv_experiment(
    experiment_id: str,
    feature_set: str,
    model_name: str,
    n_splits: int = 5,
    save_submission: bool = True,
) -> ExperimentOutput:
    root = project_root()
    report_dir = root / "reports" / "experiments"
    submission_dir = root / "submissions"
    report_dir.mkdir(parents=True, exist_ok=True)
    submission_dir.mkdir(exist_ok=True)

    train, test = load_data()
    train_fe = apply_feature_set(train, feature_set)
    test_fe = apply_feature_set(test, feature_set)

    feature_cols = [col for col in train_fe.columns if col not in [TARGET, ID_COL]]
    X = train_fe[feature_cols]
    y = train_fe[TARGET]
    X_test = test_fe[feature_cols]

    numeric_features = X.select_dtypes(include="number").columns.tolist()
    categorical_features = X.select_dtypes(exclude="number").columns.tolist()

    splitter, y_bins = get_cv(y, n_splits=n_splits)
    oof = np.zeros(len(X), dtype=float)
    test_pred = np.zeros(len(X_test), dtype=float)
    fold_rows = []

    print(f"Experiment: {experiment_id}")
    print(f"feature_set={feature_set} model_name={model_name}")
    print(f"features={len(feature_cols)} numeric={len(numeric_features)} categorical={len(categorical_features)}")

    for fold, (train_idx, valid_idx) in enumerate(splitter.split(X, y_bins), start=1):
        model = build_pipeline(model_name, numeric_features, categorical_features)
        model = clone(model)

        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        model.fit(X_train, y_train)
        valid_pred = np.clip(model.predict(X_valid), 0, 10)
        fold_score = rmse(y_valid, valid_pred)

        oof[valid_idx] = valid_pred
        test_pred += np.clip(model.predict(X_test), 0, 10) / n_splits

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
    print(f"OOF RMSE: {cv_score:.6f}")

    fold_path = report_dir / f"{experiment_id}_folds.csv"
    oof_path = report_dir / f"{experiment_id}_oof.csv"
    prediction_path = report_dir / f"{experiment_id}_test_predictions.csv"
    submission_path = submission_dir / f"{experiment_id}_{model_name}_{feature_set}.csv"

    pd.DataFrame(fold_rows).to_csv(fold_path, index=False)
    pd.DataFrame({ID_COL: train_fe[ID_COL], TARGET: y, "oof_pred": oof}).to_csv(oof_path, index=False)
    pd.DataFrame({ID_COL: test_fe[ID_COL], "prediction": np.clip(test_pred, 0, 10)}).to_csv(
        prediction_path,
        index=False,
    )

    if save_submission:
        pd.DataFrame(
            {
                ID_COL: test_fe[ID_COL],
                TARGET: np.clip(test_pred, 0, 10),
            }
        ).to_csv(submission_path, index=False)

    summary_path = report_dir / "summary.csv"
    new_summary = pd.DataFrame(
        [
            {
                "experiment_id": experiment_id,
                "feature_set": feature_set,
                "model_name": model_name,
                "n_splits": n_splits,
                "cv_rmse": cv_score,
                "fold_mean": float(np.mean([row["rmse"] for row in fold_rows])),
                "fold_std": float(np.std([row["rmse"] for row in fold_rows])),
                "oof_path": str(oof_path),
                "prediction_path": str(prediction_path),
                "submission_path": str(submission_path),
            }
        ]
    )
    if summary_path.exists():
        summary = pd.concat([pd.read_csv(summary_path), new_summary], ignore_index=True)
        summary = summary.drop_duplicates(subset=["experiment_id"], keep="last")
    else:
        summary = new_summary
    summary.sort_values("cv_rmse").to_csv(summary_path, index=False)

    return ExperimentOutput(
        experiment_id=experiment_id,
        feature_set=feature_set,
        model_name=model_name,
        cv_rmse=cv_score,
        fold_scores=[row["rmse"] for row in fold_rows],
        oof_path=oof_path,
        prediction_path=prediction_path,
        submission_path=submission_path,
    )


def blend_experiments(
    experiment_id: str,
    source_experiment_ids: list[str],
    weights: list[float] | None = None,
):
    root = project_root()
    report_dir = root / "reports" / "experiments"
    submission_dir = root / "submissions"
    report_dir.mkdir(parents=True, exist_ok=True)
    submission_dir.mkdir(exist_ok=True)

    summary_path = report_dir / "summary.csv"
    summary = pd.read_csv(summary_path)

    if weights is None:
        weights = [1 / len(source_experiment_ids)] * len(source_experiment_ids)
    weights = np.array(weights, dtype=float)
    weights = weights / weights.sum()

    oof_frames = []
    pred_frames = []
    for source_id in source_experiment_ids:
        row = summary.loc[summary["experiment_id"] == source_id].iloc[0]
        oof_frames.append(pd.read_csv(row["oof_path"]).rename(columns={"oof_pred": source_id}))
        pred_frames.append(pd.read_csv(row["prediction_path"]).rename(columns={"prediction": source_id}))

    oof = oof_frames[0][[ID_COL, TARGET]].copy()
    pred = pred_frames[0][[ID_COL]].copy()
    oof_matrix = []
    pred_matrix = []

    for frame, source_id in zip(oof_frames, source_experiment_ids):
        oof_matrix.append(frame[source_id].to_numpy())
    for frame, source_id in zip(pred_frames, source_experiment_ids):
        pred_matrix.append(frame[source_id].to_numpy())

    oof_pred = np.clip(np.column_stack(oof_matrix) @ weights, 0, 10)
    test_pred = np.clip(np.column_stack(pred_matrix) @ weights, 0, 10)

    oof["oof_pred"] = oof_pred
    pred["prediction"] = test_pred

    cv_score = rmse(oof[TARGET], oof["oof_pred"])
    oof_path = report_dir / f"{experiment_id}_oof.csv"
    prediction_path = report_dir / f"{experiment_id}_test_predictions.csv"
    submission_path = submission_dir / f"{experiment_id}_blend.csv"
    weights_path = report_dir / f"{experiment_id}_weights.csv"

    oof.to_csv(oof_path, index=False)
    pred.to_csv(prediction_path, index=False)
    pd.DataFrame({"source_experiment_id": source_experiment_ids, "weight": weights}).to_csv(
        weights_path,
        index=False,
    )
    pd.DataFrame({ID_COL: pred[ID_COL], TARGET: test_pred}).to_csv(submission_path, index=False)

    new_summary = pd.DataFrame(
        [
            {
                "experiment_id": experiment_id,
                "feature_set": "blend",
                "model_name": "blend",
                "n_splits": np.nan,
                "cv_rmse": cv_score,
                "fold_mean": np.nan,
                "fold_std": np.nan,
                "oof_path": str(oof_path),
                "prediction_path": str(prediction_path),
                "submission_path": str(submission_path),
            }
        ]
    )
    summary = pd.concat([summary, new_summary], ignore_index=True)
    summary = summary.drop_duplicates(subset=["experiment_id"], keep="last")
    summary.sort_values("cv_rmse").to_csv(summary_path, index=False)

    print(f"Blend {experiment_id} OOF RMSE: {cv_score:.6f}")
    return {
        "experiment_id": experiment_id,
        "cv_rmse": cv_score,
        "submission_path": submission_path,
        "weights_path": weights_path,
    }


def available_feature_sets() -> list[str]:
    return list(FEATURE_SET_STEPS.keys())


def available_model_names() -> list[str]:
    names = [
        "ridge_ohe",
        "hgb_ohe_base",
        "hgb_ohe_compact",
        "hgb_te_base",
        "hgb_te_smooth",
        "extra_trees_ohe",
        "random_forest_ohe",
    ]
    for optional_name, package_name in [
        ("lgbm_ohe", "lightgbm"),
        ("xgb_ohe", "xgboost"),
        ("catboost_ohe", "catboost"),
    ]:
        if package_exists(package_name):
            names.append(optional_name)
    return names
