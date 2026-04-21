"""
ml/train_model.py
Huấn luyện ML model phân loại 6 ngành PTIT.

Thuật toán: Gradient Boosting — phù hợp nhất cho bài toán tabular data
nhỏ (6 features, ~21k mẫu). Tìm tham số tối ưu bằng RandomizedSearchCV.

Yêu cầu đặc biệt để tương thích với KBS:
  1. Model phải có predict_proba() → KBS cần probas[top1] - probas[top2]
  2. Xác suất phải được calibrate → ngưỡng confidence 0.10 của KBS mới có ý nghĩa
  3. Dùng confidence (Pearson r gap) làm sample_weight → model học nhiều hơn
     từ mẫu nhãn rõ ràng
  4. Input features: 6 X_norm [0,1] — đúng thang như UserProfile trong KBS

Đầu vào : data/processed/train.csv, data/processed/test.csv
Đầu ra  : models/classifier.joblib, models/label_encoder.joblib,
           models/feature_cols.joblib, models/tuning_report.json
"""

import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path
from scipy.stats import randint, uniform

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, f1_score,
    classification_report, confusion_matrix
)

TRAIN_PATH = Path("data/processed/train.csv")
TEST_PATH  = Path("data/processed/test.csv")
MODEL_DIR  = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

DIMS   = list("RIASEC")
MAJORS = ["CNTT", "ATTT", "MKT", "TMDT", "TTDPT", "KETOAN"]

# ── Features đưa vào ML model ────────────────────────────────────────────────
FEATURE_COLS = [f"{d}_norm" for d in DIMS]
TARGET_COL   = "assigned_major"
WEIGHT_COL   = "confidence"


# ── 1. Load dữ liệu ──────────────────────────────────────────────────────────
def load_data():
    train = pd.read_csv(TRAIN_PATH, index_col="sample_id")
    test  = pd.read_csv(TEST_PATH,  index_col="sample_id")

    X_train = train[FEATURE_COLS].values
    y_train = train[TARGET_COL].values
    w_train = train[WEIGHT_COL].values

    X_test  = test[FEATURE_COLS].values
    y_test  = test[TARGET_COL].values

    print(f"[load]  Train: {len(X_train):,}  |  Test: {len(X_test):,}")
    print(f"        Features: {FEATURE_COLS}")
    return X_train, y_train, w_train, X_test, y_test


# ── 2. Encode nhãn ───────────────────────────────────────────────────────────
def encode_labels(y_train, y_test):
    le = LabelEncoder()
    le.fit(MAJORS)
    return le.transform(y_train), le.transform(y_test), le


# ── 3. Tìm tham số tối ưu bằng RandomizedSearchCV ────────────────────────────
def tune_hyperparameters(X_train, y_train_enc, w_train):
    """
    Tại sao dùng Gradient Boosting?
      - 6 features, ~21k mẫu → tabular data nhỏ
      - GB học sequential (mỗi cây sửa lỗi cây trước) → tận dụng tốt 6 features
      - Trên tabular data nhỏ, GB thường cho F1 cao hơn RF
        (Grinsztajn et al., 2022; Shwartz-Ziv & Armon, 2022)
      - Hỗ trợ predict_proba, sample_weight, calibrate được

    Tại sao dùng RandomizedSearchCV?
      - GridSearchCV duyệt toàn bộ tổ hợp → quá chậm
      - RandomizedSearchCV chọn ngẫu nhiên n_iter=50 tổ hợp → nhanh hơn
        mà vẫn tìm được vùng tham số tốt (Bergstra & Bengio, 2012)
      - Scoring = f1_macro → công bằng với 6 ngành
    """
    print("[tune]  Bắt đầu RandomizedSearchCV (50 iterations, 5-fold CV)...")

    param_dist = {
        "n_estimators":    randint(100, 500),      # 100-500 cây
        "max_depth":       randint(3, 8),           # 3-7 tầng
        "learning_rate":   uniform(0.01, 0.19),     # 0.01-0.20
        "subsample":       uniform(0.6, 0.4),       # 0.6-1.0
        "min_samples_leaf": randint(2, 20),          # 2-19
    }

    base = GradientBoostingClassifier(random_state=42)

    search = RandomizedSearchCV(
        estimator=base,
        param_distributions=param_dist,
        n_iter=50,
        scoring="f1_macro",
        cv=5,
        random_state=42,
        n_jobs=-1,          # song song hóa CV folds
        verbose=1,
        return_train_score=True,
    )

    search.fit(X_train, y_train_enc, sample_weight=w_train)

    print(f"\n[tune]  Best F1-macro (CV): {search.best_score_:.4f}")
    print(f"[tune]  Best params:")
    for k, v in search.best_params_.items():
        print(f"        {k}: {v}")

    return search


# ── 4. Calibration ────────────────────────────────────────────────────────────
def calibrate_model(best_estimator, X_train, y_train_enc, w_train):
    """
    CalibratedClassifierCV — Platt scaling (1999).
    Điều chỉnh xác suất raw → xác suất thực.
    Nếu không calibrate → confidence luôn cao → tiebreaker KBS không hoạt động.
    """
    print("[calib] Calibrating best model (Platt scaling, 5-fold)...")

    calibrated = CalibratedClassifierCV(
        estimator=best_estimator,
        method="sigmoid",
        cv=5,
    )
    calibrated.fit(X_train, y_train_enc, sample_weight=w_train)
    return calibrated


# ── 5. Đánh giá model ────────────────────────────────────────────────────────
def evaluate(model, X_test, y_test_enc, le):
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    acc = accuracy_score(y_test_enc, y_pred)
    f1m = f1_score(y_test_enc, y_pred, average="macro")
    f1w = f1_score(y_test_enc, y_pred, average="weighted")

    print(f"\n[eval]  Kết quả trên test set:")
    print(f"  Accuracy    : {acc:.4f}")
    print(f"  F1-macro    : {f1m:.4f}")
    print(f"  F1-weighted : {f1w:.4f}")
    print(f"\n{classification_report(y_test_enc, y_pred, target_names=le.classes_)}")

    # Top-3 accuracy
    top3_idx = np.argsort(-y_proba, axis=1)[:, :3]
    top3_acc = sum(y_test_enc[i] in top3_idx[i] for i in range(len(y_test_enc))) / len(y_test_enc)
    print(f"  Top-3 accuracy: {top3_acc:.4f}")

    # Confidence distribution — quan trọng cho KBS
    top_proba   = np.sort(y_proba, axis=1)[:, ::-1]
    confidences = top_proba[:, 0] - top_proba[:, 1]
    low_conf_pct = (confidences < 0.10).mean()
    print(f"  Tỷ lệ confidence < 0.10 (KBS tiebreaker kích hoạt): {low_conf_pct:.1%}")
    print(f"  Confidence mean: {confidences.mean():.4f}  median: {np.median(confidences):.4f}")

    return acc, f1m, f1w, top3_acc


# ── MAIN ─────────────────────────────────────────────────────────────────────
def train():
    print("=" * 60)
    print("  HUẤN LUYỆN ML MODEL — GRADIENT BOOSTING + TUNING")
    print("=" * 60)

    X_train, y_train, w_train, X_test, y_test = load_data()
    y_train_enc, y_test_enc, le = encode_labels(y_train, y_test)

    # Tìm tham số tối ưu
    search = tune_hyperparameters(X_train, y_train_enc, w_train)
    best_base = search.best_estimator_

    # Calibrate model tốt nhất
    model = calibrate_model(best_base, X_train, y_train_enc, w_train)

    # Đánh giá
    acc, f1m, f1w, top3_acc = evaluate(model, X_test, y_test_enc, le)

    # ── Lưu artifacts ────────────────────────────────────────────────────────
    joblib.dump(model,        MODEL_DIR / "classifier.joblib")
    joblib.dump(le,           MODEL_DIR / "label_encoder.joblib")
    joblib.dump(FEATURE_COLS, MODEL_DIR / "feature_cols.joblib")

    # Lưu tuning report
    tuning_report = {
        "algorithm": "GradientBoostingClassifier",
        "tuning_method": "RandomizedSearchCV",
        "n_iter": 50,
        "cv_folds": 5,
        "best_params": {k: (int(v) if isinstance(v, (np.integer,)) else
                           float(v) if isinstance(v, (np.floating,)) else v)
                       for k, v in search.best_params_.items()},
        "best_cv_f1_macro": round(float(search.best_score_), 4),
        "test_accuracy": round(acc, 4),
        "test_f1_macro": round(f1m, 4),
        "test_f1_weighted": round(f1w, 4),
        "test_top3_accuracy": round(top3_acc, 4),
    }
    with open(MODEL_DIR / "tuning_report.json", "w", encoding="utf-8") as f:
        json.dump(tuning_report, f, ensure_ascii=False, indent=2)

    print(f"\n[saved] → {MODEL_DIR}/classifier.joblib")
    print(f"        → {MODEL_DIR}/tuning_report.json")
    print(f"\n{'='*60}")
    print(f"  Best params: {search.best_params_}")
    print(f"  CV F1-macro: {search.best_score_:.4f}")
    print(f"  Test F1-macro: {f1m:.4f}")
    print(f"{'='*60}")

    return model, le


if __name__ == "__main__":
    train()