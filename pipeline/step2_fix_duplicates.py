"""
pipeline/step2_fix_duplicates.py
Bước 2: Loại bỏ duplicate samples trong train_final.csv.

Đầu vào : data/processed/train_final.csv
Đầu ra  : data/processed/train_ready.csv   ← file cuối cùng dùng để train

Tiêu chí trùng lặp: cùng 6 giá trị X_norm (profile RIASEC giống hệt nhau).
Xử lý: khi có duplicate, giữ dòng có confidence cao hơn (nhãn rõ hơn).
"""

import pandas as pd
from pathlib import Path

IN_PATH  = Path("./data/processed/train_final.csv")
OUT_PATH = Path("./data/processed/train_ready.csv")

DIMS      = list("RIASEC")
MAJORS    = ["CNTT", "ATTT", "MKT", "TMDT", "TTDPT", "KETOAN"]
NORM_COLS = [f"{d}_norm" for d in DIMS]


def run():
    print("=" * 55)
    print("  BƯỚC 2: LOẠI BỎ DUPLICATE SAMPLES")
    print("=" * 55)

    df = pd.read_csv(IN_PATH, index_col="sample_id")
    print(f"\n[load]   {len(df):,} mẫu từ {IN_PATH}")

    # Thống kê duplicate
    dup_mask = df.duplicated(subset=NORM_COLS, keep=False)
    dup_df   = df[dup_mask]
    print(f"\n[dup]    Tìm thấy {dup_mask.sum()} dòng trùng lặp "
          f"({dup_mask.sum() // 2} cặp):")
    if dup_mask.sum() > 0:
        print("  Phân bố theo ngành:")
        for m, n in dup_df["assigned_major"].value_counts().items():
            print(f"    {m:<10} {n}")

    # Xóa duplicate: sort theo confidence giảm dần → giữ dòng đầu tiên
    df_sorted = df.sort_values("confidence", ascending=False)
    df_clean  = df_sorted.drop_duplicates(subset=NORM_COLS, keep="first")
    print(f"\n[clean]  Sau xóa: {len(df_clean):,} mẫu "
          f"(loại {len(df) - len(df_clean)})")

    # Kiểm tra phân phối nhãn sau khi xóa
    dist = df_clean["assigned_major"].value_counts()
    print(f"\n[dist]   Phân phối nhãn sau fix:")
    for m in MAJORS:
        n   = dist.get(m, 0)
        pct = n / len(df_clean)
        print(f"  {m:<10} {n:>5,}  ({pct:.1%})")

    ratio = dist.max() / dist.min()
    print(f"\n  max/min = {ratio:.2f}x  {'✅' if ratio < 1.1 else '⚠️'}")

    # Xác nhận không còn duplicate
    remaining = df_clean.duplicated(subset=NORM_COLS).sum()
    print(f"  Duplicates còn lại: {remaining}  "
          f"{'✅' if remaining == 0 else '⚠️'}")

    # Lưu
    df_clean.index.name = "sample_id"
    df_clean.to_csv(OUT_PATH)
    print(f"\n[done]   Lưu → {OUT_PATH}")
    print(f"         {len(df_clean):,} mẫu  |  {df_clean.shape[1]} cột")


if __name__ == "__main__":
    run()