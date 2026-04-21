"""
pipeline/step4_train_test_split.py
Tách train_ready.csv thành tập train (70%) và test (30%).

Chiến lược: Stratified split theo assigned_major
  - Đảm bảo tỷ lệ 6 ngành giống nhau ở cả 2 tập
  - random_state=42 để tái lập kết quả

Đầu vào : data/processed/train_ready.csv
Đầu ra  :
  data/processed/train.csv   (20.923 mẫu ~ 70%)
  data/processed/test.csv    ( 8.967 mẫu ~ 30%)
"""

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

IN_PATH    = Path("./data/processed/train_ready.csv")
TRAIN_PATH = Path("./data/processed/train.csv")
TEST_PATH  = Path("./data/processed/test.csv")

MAJORS = ["CNTT", "ATTT", "MKT", "TMDT", "TTDPT", "KETOAN"]


def run():
    print("=" * 55)
    print("  BƯỚC 4: TÁCH TRAIN / TEST  (70% / 30%)")
    print("=" * 55)

    df = pd.read_csv(IN_PATH, index_col="sample_id")
    print(f"\n[load]   {len(df):,} mẫu từ {IN_PATH}")

    # Stratified split — giữ nguyên tỷ lệ từng ngành
    train_df, test_df = train_test_split(
        df,
        test_size=0.30,
        stratify=df["assigned_major"],
        random_state=42,
    )

    # Báo cáo
    total = len(df)
    print(f"\n[split]  Stratified split (random_state=42):")
    print(f"  {'Ngành':<10} {'Tổng':>6}  {'Train':>7}  {'Test':>7}  "
          f"{'Train%':>8}  {'Test%':>7}")

    for m in MAJORS:
        n_all   = (df["assigned_major"]    == m).sum()
        n_train = (train_df["assigned_major"] == m).sum()
        n_test  = (test_df["assigned_major"]  == m).sum()
        print(f"  {m:<10} {n_all:>6,}  {n_train:>7,}  {n_test:>7,}  "
              f"{n_train/n_all:>7.1%}  {n_test/n_all:>6.1%}")

    print(f"\n  {'TỔNG':<10} {total:>6,}  {len(train_df):>7,}  "
          f"{len(test_df):>7,}  "
          f"{len(train_df)/total:>7.1%}  {len(test_df)/total:>6.1%}")

    # Kiểm tra không có data leak
    train_idx = set(train_df.index)
    test_idx  = set(test_df.index)
    overlap   = train_idx & test_idx
    print(f"\n  Data leak check (overlap): {len(overlap)}  "
          f"{'✅' if len(overlap) == 0 else '🔴 CÓ LEAK!'}")

    # Lưu
    train_df.index.name = "sample_id"
    test_df.index.name  = "sample_id"
    train_df.to_csv(TRAIN_PATH)
    test_df.to_csv(TEST_PATH)

    print(f"\n[done]   Lưu → {TRAIN_PATH}  ({len(train_df):,} mẫu)")
    print(f"         Lưu → {TEST_PATH}  ({len(test_df):,} mẫu)")
    print(f"\n  Lưu ý: tập test KHÔNG được dùng trong quá trình train.")
    print(f"  Chỉ dùng để đánh giá model sau khi train xong.")


if __name__ == "__main__":
    run()