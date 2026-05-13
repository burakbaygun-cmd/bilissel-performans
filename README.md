# 🏆 Datathon — Bilişsel Performans Skoru Tahmini

**Görev:** Regresyon  
**Hedef değişken:** `bilissel_performans_skoru`  
**Metrik:** RMSE  
**En iyi Public RMSE:** `1.20244`  
**Local CV RMSE:** `1.21350`

---

## Yaklaşım Özeti

Final model tek bir model değil, birden fazla modelin OOF (Out-of-Fold) validation tahminleri üzerinden optimize edilmiş bir blend'idir.

### Model Aileleri

| Model | Encoding | Özellik |
|---|---|---|
| HistGradientBoosting | Target Encoding | sklearn native, erken durdurma |
| LightGBM | OHE / Target Encoding | Regularize edilmiş config |
| XGBoost | OHE / Target Encoding | Histogram tabanlı |
| CatBoost | Native categorical | En güçlü aile, seed averaging |

### Final Blend Ağırlıkları

| Kaynak | Ağırlık |
|---|---:|
| exp_802_cat_tuned_depth5_reg | 0.4138 |
| exp_606_cat_native_base | 0.2752 |
| exp_890_cat_seed_average_blend | 0.1707 |
| exp_790_cat_feature_top3_equal_blend | 0.0631 |
| exp_491_hgb_te_tuned_top3_inverse_blend | 0.0383 |
| exp_603_lgbm_te_regularized | 0.0377 |
| exp_604_xgb_ohe_base | 0.0007 |
| exp_808_cat_tuned_high_random_strength | 0.0002 |
| exp_605_xgb_te_base | 0.0002 |

### Pipeline Aşamaları

```
Stage 4 → HGB + Target Encoding hiperparametre taraması (9 config)
Stage 6 → LightGBM / XGBoost / CatBoost native modelleri (8 config)
Stage 7 → CatBoost feature set ablasyonu → top-3 blend
Stage 8 → CatBoost fine-tuning + 5 seed averaging
Stage 9 → OOF tabanlı seçilmiş blend optimizasyonu (grid + Dirichlet + pairwise)
```

### Validation Stratejisi

Stratified 5-fold CV. Hedef değişken sürekli olduğu için fold'lar quantile bin'leriyle dengelendi.

### Önemli Bulgular

- CatBoost bu veri setinde en güçlü model ailesi oldu.
- En büyük iyileşme tek büyük modelden değil, güvenilir OOF blend optimizasyonundan geldi.
- `id` leakage, duplicate ve train-test dağılım farkı tespit edilmedi.
- XGBoost tek başına güçlüydü ama final blend içinde ağırlığı çok düşük kaldı.

---

## Kurulum

Python 3.11 önerilir.

```bash
pip install -r requirements.txt
```

---

## Veri

Veri dosyaları Kaggle'dan indirilerek `data/` klasörüne koyulmalıdır:

```
data/train.csv
data/test_x.csv
data/sample_submission.csv
```

---

## Çalıştırma

### Notebook (önerilen)

```
notebooks/datathon_final_pipeline.ipynb
```

Tüm aşamalar açıklamalar ve çıktılarla birlikte tek notebook'ta.

### Script (komut satırı)

```bash
python src/run_final_reproducible_pipeline.py
```

Bu script sırasıyla şunları çalıştırır:

1. HGB target encoding tuning (Stage 4)
2. LightGBM / XGBoost / CatBoost modelleri (Stage 6)
3. CatBoost feature blend (Stage 7)
4. CatBoost tuning + seed averaging (Stage 8)
5. Selected blend optimization (Stage 9)

Çıktı: `submissions/final_submission.csv`

### GPU (opsiyonel)

```bash
set CATBOOST_TASK_TYPE=GPU
set CATBOOST_DEVICES=0
python src/run_final_reproducible_pipeline.py
```

---

## Proje Yapısı

```
├── notebooks/
│   └── datathon_final_pipeline.ipynb   # Ana notebook (tüm pipeline)
├── src/
│   ├── experiment_framework.py          # Ortak altyapı (encoder, feature eng, CV)
│   ├── run_final_reproducible_pipeline.py
│   ├── run_stage4_hgb_te_tuning.py
│   ├── run_stage6_external_gbdt_models.py
│   ├── run_stage7_catboost_feature_ablation.py
│   ├── run_stage8_catboost_tuning_seed.py
│   └── run_stage9_selected_blend_optimization.py
├── data/
│   └── README.md                        # Veri indirme talimatları
├── requirements.txt
└── README.md
```
