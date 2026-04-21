"""
pipeline/step3_check_quality.py
Bước 3: Kiểm tra chất lượng file train_ready.csv trước khi train model.

Đầu vào : data/processed/train_ready.csv
Đầu ra  : in kết quả ra màn hình (không lưu file)

8 hạng mục kiểm tra:
  1. Schema & cấu trúc
  2. Missing values & duplicates
  3. Phân phối nhãn
  4. Chất lượng nhãn (Pearson r)
  5. Phân phối RIASEC scores
  6. Logic nhất quán (r_best, argmax, confidence)
  7. Ma trận r trung bình theo ngành
  8. Tóm tắt cho báo cáo
"""

import pandas as pd
import numpy as np
from pathlib import Path

PATH   = Path("./data/processed/train_ready.csv")
DIMS   = list("RIASEC")
MAJORS = ["CNTT", "ATTT", "MKT", "TMDT", "TTDPT", "KETOAN"]


def sep(title=""):
    print(f"\n{'─'*58}")
    if title:
        print(f"  {title}")
        print("─" * 58)


def check_schema(df):
    sep("1. SCHEMA & CẤU TRÚC")
    print(f"  Số mẫu  : {len(df):,}")
    print(f"  Số cột  : {df.shape[1]}")
    expected = (
        [f"{d}_score" for d in DIMS] + [f"{d}_norm" for d in DIMS] +
        [f"r_{m}" for m in MAJORS] +
        ["r_best", "r_second", "confidence", "fit_quality", "assigned_major"]
    )
    missing = [c for c in expected if c not in df.columns]
    print(f"  Schema  : {'✅ đủ' if not missing else '⚠️  thiếu: ' + str(missing)}")


def check_basic(df):
    sep("2. MISSING VALUES & DUPLICATES")
    miss = df.isnull().sum()
    miss = miss[miss > 0]
    print(f"  Missing : {'✅ không có' if miss.empty else miss.to_string()}")
    dups = df.duplicated(subset=[f"{d}_norm" for d in DIMS]).sum()
    print(f"  Dup     : {dups}  {'✅' if dups == 0 else '⚠️'}")


def check_labels(df):
    sep("3. PHÂN PHỐI NHÃN")
    dist  = df["assigned_major"].value_counts()
    total = len(df)
    print(f"  {'Ngành':<10} {'N':>6}  {'%':>6}  Bar")
    for m in MAJORS:
        n   = dist.get(m, 0)
        bar = "█" * int(n / total * 40)
        print(f"  {m:<10} {n:>6,}  {n/total:>5.1%}  {bar}")
    ratio = dist.max() / dist.min()
    print(f"\n  max/min = {ratio:.2f}x  {'✅' if ratio < 1.1 else '⚠️'}")


def check_label_quality(df):
    sep("4. CHẤT LƯỢNG NHÃN (Pearson r — Rounds et al., 2021)")
    if "fit_quality" in df.columns:
        print(f"  {'Fit level':<12} {'N':>6}  {'%':>6}")
        for q in ["Best Fit", "Great Fit", "Good Fit"]:
            n = (df["fit_quality"] == q).sum()
            print(f"  {q:<12} {n:>6,}  ({n/len(df):.1%})")

    print(f"\n  {'Ngành':<10} {'N':>6}  {'r_best':>8}  {'conf':>8}  Quality")
    for m in MAJORS:
        sub = df[df["assigned_major"] == m]
        if not len(sub): continue
        r  = sub["r_best"].mean()
        c  = sub["confidence"].mean()
        q  = "Best Fit ✅" if r >= 0.729 else "Great Fit ✅" if r >= 0.608 else "Good Fit ℹ️ "
        print(f"  {m:<10} {len(sub):>6,}  {r:>8.4f}  {c:>8.4f}  {q}")

    low = (df["confidence"] < 0.05).sum()
    print(f"\n  Mẫu confidence < 0.05: {low:,} ({low/len(df):.1%})")
    print(f"  → Dùng cột 'confidence' làm sample_weight khi train")


def check_features(df):
    sep("5. PHÂN PHỐI RIASEC SCORES")
    norm_cols = [f"{d}_norm" for d in DIMS]
    desc = df[norm_cols].describe().T[["mean", "std", "min", "max"]]
    desc.index = DIMS
    print(desc.round(4).to_string())
    out = df[norm_cols].apply(lambda c: (~c.between(0, 1)).sum()).sum()
    print(f"\n  Giá trị ngoài [0,1]: {out}  {'✅' if out == 0 else '⚠️'}")


def check_logic(df):
    sep("6. KIỂM TRA LOGIC NHẤT QUÁN")
    r_cols = [f"r_{m}" for m in MAJORS if f"r_{m}" in df.columns]

    computed_best = df[r_cols].max(axis=1)
    mm1 = (abs(computed_best - df["r_best"]) > 1e-4).sum()
    print(f"  r_best == max(r_*)           : {'✅' if mm1==0 else f'⚠️  {mm1} mismatch'}")

    argmax = df[r_cols].idxmax(axis=1).str.replace("r_", "", regex=False)
    mm2 = (argmax != df["assigned_major"]).sum()
    print(f"  assigned_major == argmax(r_*): {'✅' if mm2==0 else f'⚠️  {mm2} mismatch'}")

    sorted_r = np.sort(df[r_cols].values, axis=1)
    comp_conf = sorted_r[:, -1] - sorted_r[:, -2]
    mm3 = (abs(comp_conf - df["confidence"].values) > 1e-4).sum()
    print(f"  confidence == r_best-r_second: {'✅' if mm3==0 else f'⚠️  {mm3} mismatch'}")


def check_r_matrix(df):
    sep("7. MA TRẬN r TRUNG BÌNH THEO NGÀNH")
    r_cols = [f"r_{m}" for m in MAJORS if f"r_{m}" in df.columns]
    print(f"  (hàng = assigned_major, cột = r_ngành, 【】= ngành được gán)")
    print(f"  {'':10}" + "".join(f"{m:>9}" for m in MAJORS))
    for m in MAJORS:
        sub = df[df["assigned_major"] == m]
        if not len(sub): continue
        vals = ""
        for mm in MAJORS:
            v = sub[f"r_{mm}"].mean() if f"r_{mm}" in sub.columns else 0
            if mm == m:
                vals += f"【{v:>6.3f}】"
            else:
                vals += f" {v:>7.3f} "
        print(f"  {m:<10}{vals}")
    print(f"\n  Kiểm tra: giá trị 【】phải là lớn nhất trong mỗi hàng")


def summary(df):
    sep("8. TÓM TẮT")
    dist = df["assigned_major"].value_counts()
    print(f"""
  File          : train_ready.csv
  Tổng mẫu      : {len(df):,}
  Input features: 12 (6 X_score + 6 X_norm)
  Target        : assigned_major (6 classes)
  Sample weight : confidence (dùng khi train)

  Phương pháp gán nhãn:
    Pearson profile correlation
    Nguồn: McCloy et al. (1999); Rounds et al. (2021)

  Chất lượng nhãn:
    r_best mean      = {df['r_best'].mean():.4f}
    Best Fit (≥.729) = {(df['r_best']>=0.729).mean():.1%}
    Great+ Fit (≥.608) = {(df['r_best']>=0.608).mean():.1%}
    Confidence mean  = {df['confidence'].mean():.4f}

  Phân phối nhãn:""")
    for m in MAJORS:
        n = dist.get(m, 0)
        print(f"    {m:<10} {n:>5,}  ({n/len(df):.1%})")


def run():
    print("=" * 58)
    print("  BƯỚC 3: KIỂM TRA CHẤT LƯỢNG — train_ready.csv")
    print("=" * 58)
    df = pd.read_csv(PATH, index_col="sample_id")
    print(f"\n  Đọc xong: {len(df):,} mẫu  |  {df.shape[1]} cột")

    check_schema(df)
    check_basic(df)
    check_labels(df)
    check_label_quality(df)
    check_features(df)
    check_logic(df)
    check_r_matrix(df)
    summary(df)


if __name__ == "__main__":
    run()