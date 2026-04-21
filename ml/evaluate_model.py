"""
ml/evaluate_model.py
Đánh giá model sau khi train. Chạy: python ml/evaluate_model.py

Kiểm tra:
  1. Accuracy + F1 trên train và test → phát hiện overfitting/underfitting
  2. Per-class metrics → ngành nào model yếu
  3. Confusion matrix → nhầm lẫn ở đâu
  4. Feature importance → feature nào quan trọng nhất
  5. Cross-validation → model ổn định không
  6. Calibration → xác suất có tin cậy không

Lưu kết quả vào models/eval_report.json để dashboard đọc.
"""

import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix
)
from sklearn.model_selection import cross_val_score

# ── Paths ─────────────────────────────────────────────────────────────────────
TRAIN_PATH  = Path("data/processed/train.csv")
TEST_PATH   = Path("data/processed/test.csv")
MODEL_DIR   = Path("models")
REPORT_PATH = MODEL_DIR / "eval_report.json"

MAJORS = ["CNTT", "ATTT", "MKT", "TMDT", "TTDPT", "KETOAN"]
DIMS   = list("RIASEC")
FEAT   = [f"{d}_norm" for d in DIMS]


def load_data():
    train = pd.read_csv(TRAIN_PATH, index_col="sample_id")
    test  = pd.read_csv(TEST_PATH,  index_col="sample_id")
    return train, test


def load_model():
    model = joblib.load(MODEL_DIR / "classifier.joblib")
    le    = joblib.load(MODEL_DIR / "label_encoder.joblib")
    return model, le


def compute_metrics(model, le, X, y_true_str):
    y_true = le.transform(y_true_str)
    y_pred = model.predict(X)
    proba  = model.predict_proba(X)

    acc    = accuracy_score(y_true, y_pred)
    f1m    = f1_score(y_true, y_pred, average="macro")
    f1w    = f1_score(y_true, y_pred, average="weighted")
    report = classification_report(y_true, y_pred,
                                   target_names=le.classes_,
                                   output_dict=True)
    cm     = confusion_matrix(y_true, y_pred).tolist()

    top3_idx = np.argsort(-proba, axis=1)[:, :3]
    top3_acc = sum(y_true[i] in top3_idx[i] for i in range(len(y_true))) / len(y_true)

    return {
        "accuracy":     round(acc, 4),
        "f1_macro":     round(f1m, 4),
        "f1_weighted":  round(f1w, 4),
        "top3_accuracy": round(top3_acc, 4),
        "per_class":    {k: {m: round(v, 4) for m, v in v2.items()}
                         for k, v2 in report.items()
                         if k in le.classes_},
        "confusion_matrix": cm,
    }


def check_overfitting(train_f1, test_f1):
    gap = train_f1 - test_f1
    if gap > 0.15:
        return "OVERFIT", f"Khoảng cách F1 macro train-test = {gap:.3f} > 0.15 → Model học thuộc dữ liệu train."
    elif test_f1 < 0.60:
        return "UNDERFIT", f"F1 macro test = {test_f1:.3f} < 0.60 → Model chưa đủ phức tạp."
    elif gap > 0.08:
        return "MILD_OVERFIT", f"Khoảng cách F1 = {gap:.3f} — nhẹ. Cân nhắc điều chỉnh hyperparameter."
    else:
        return "OK", f"Train-test gap = {gap:.3f} ≤ 0.08. Model ổn định."


def get_feature_importance(model, le):
    """Lấy feature importance từ GradientBoosting base estimator."""
    try:
        # CalibratedClassifierCV wraps estimators
        base = model.estimators_[0].estimator
        fi = base.feature_importances_
        return {FEAT[i]: round(float(fi[i]), 4) for i in range(len(FEAT))}
    except Exception:
        try:
            base = model.estimator
            fi = base.feature_importances_
            return {FEAT[i]: round(float(fi[i]), 4) for i in range(len(FEAT))}
        except Exception:
            return {f: 0.0 for f in FEAT}


def check_calibration(model, le, X, y_true_str):
    y_true = le.transform(y_true_str)
    proba  = model.predict_proba(X)
    top_proba = proba.max(axis=1)
    y_pred    = model.predict(X)
    correct   = (y_pred == y_true).astype(int)

    bins = np.linspace(0, 1, 6)
    cal  = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (top_proba >= lo) & (top_proba < hi)
        if mask.sum() > 0:
            cal.append({
                "bin":      f"{lo:.1f}–{hi:.1f}",
                "mean_conf": round(float(top_proba[mask].mean()), 3),
                "accuracy":  round(float(correct[mask].mean()), 3),
                "count":     int(mask.sum()),
            })
    return cal


def cross_validate(model, X_train, y_train_enc):
    scores = cross_val_score(model, X_train, y_train_enc,
                             cv=5, scoring="f1_macro", n_jobs=-1)
    return {
        "scores": [round(s, 4) for s in scores],
        "mean":   round(scores.mean(), 4),
        "std":    round(scores.std(), 4),
    }


def make_recommendations(status, train_m, test_m, fi):
    recs = []

    if status == "OVERFIT":
        recs.append("Giảm n_estimators hoặc tăng min_samples_leaf để hạn chế overfitting.")
        recs.append("Giảm max_depth (vd: 4→3) cho cây nông hơn.")
        recs.append("Tăng subsample (vd: 0.8→0.7) để thêm regularization.")

    if status == "UNDERFIT":
        recs.append("Tăng n_estimators hoặc giảm learning_rate.")
        recs.append("Tăng max_depth cho phép cây phân nhánh sâu hơn.")

    f1_by_class = {k: v["f1-score"] for k, v in test_m["per_class"].items()}
    worst = min(f1_by_class, key=f1_by_class.get)
    best  = max(f1_by_class, key=f1_by_class.get)
    if f1_by_class[worst] < 0.65:
        recs.append(f"Ngành {worst} có F1={f1_by_class[worst]:.2f} thấp nhất — xem xét thêm mẫu hoặc kiểm tra nhãn.")

    top_feat = max(fi, key=fi.get)
    recs.append(f"Feature quan trọng nhất: {top_feat} (importance={fi[top_feat]:.3f}).")
    recs.append(f"Ngành tốt nhất: {best} (F1={f1_by_class[best]:.2f}).")

    return recs


def evaluate():
    print("=" * 55)
    print("  ĐÁNH GIÁ MODEL — ML/evaluate_model.py")
    print("=" * 55)

    train, test = load_data()
    model, le   = load_model()

    X_train = train[FEAT].values
    y_train = train["assigned_major"].values
    X_test  = test[FEAT].values
    y_test  = test["assigned_major"].values

    y_train_enc = le.transform(y_train)

    print(f"\n[data]  Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    print("[eval]  Tính metrics trên train...")
    train_m = compute_metrics(model, le, X_train, y_train)
    print("[eval]  Tính metrics trên test...")
    test_m  = compute_metrics(model, le, X_test, y_test)

    status, diag = check_overfitting(train_m["f1_macro"], test_m["f1_macro"])
    print(f"[check] Status: {status} — {diag}")

    print("[feat]  Feature importance...")
    fi = get_feature_importance(model, le)

    print("[calib] Calibration check...")
    cal = check_calibration(model, le, X_test, y_test)

    print("[cv]    Cross-validation (5-fold)...")
    cv = cross_validate(model, X_train, y_train_enc)
    print(f"        F1-macro CV: {cv['mean']:.4f} ± {cv['std']:.4f}")

    recs = make_recommendations(status, train_m, test_m, fi)

    # Load tuning report if exists
    tuning_info = {}
    tuning_path = MODEL_DIR / "tuning_report.json"
    if tuning_path.exists():
        tuning_info = json.loads(tuning_path.read_text(encoding="utf-8"))

    print(f"\n{'─'*55}")
    print(f"  TỔNG KẾT")
    print(f"{'─'*55}")
    print(f"  {'Metric':<22} {'Train':>8}  {'Test':>8}")
    for m in ["accuracy","f1_macro","f1_weighted","top3_accuracy"]:
        print(f"  {m:<22} {train_m[m]:>8.4f}  {test_m[m]:>8.4f}")
    print(f"  {'CV F1-macro':<22} {'—':>8}  {cv['mean']:>8.4f} ±{cv['std']:.4f}")

    print(f"\n  Per-class F1 (test):")
    for cls in le.classes_:
        f1 = test_m["per_class"][cls]["f1-score"]
        bar = "█" * int(f1 * 20)
        flag = " ⚠️" if f1 < 0.65 else ""
        print(f"  {cls:<10} {f1:.4f}  {bar}{flag}")

    print(f"\n  Khuyến nghị:")
    for r in recs:
        print(f"  • {r}")

    report = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "algorithm":     tuning_info.get("algorithm", "GradientBoostingClassifier"),
        "best_params":   tuning_info.get("best_params", {}),
        "status":        status,
        "diagnosis":     diag,
        "train":         train_m,
        "test":          test_m,
        "cv":            cv,
        "feature_importance": fi,
        "calibration":   cal,
        "recommendations": recs,
        "class_names":   list(le.classes_),
        "n_train":       len(X_train),
        "n_test":        len(X_test),
    }
    REPORT_PATH.parent.mkdir(exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n[done]  Lưu báo cáo → {REPORT_PATH}")
    return report


if __name__ == "__main__":
    evaluate()