"""
run_pipeline.py
Chạy toàn bộ pipeline chuẩn bị dataset theo thứ tự.

Yêu cầu:
  data/raw/data.csv          — tải từ openpsychometrics.org
  data/raw/Interests.xlsx    — tải từ onetcenter.org/database.html

Kết quả:
  data/processed/train_final.csv   — sau bước 1
  data/processed/train_ready.csv   — file cuối dùng để train

Cài đặt thư viện:
  pip install pandas numpy openpyxl scipy
"""

from pipeline.step1_prepare_dataset import run as step1
from pipeline.step2_fix_duplicates  import run as step2
from pipeline.step3_check_quality   import run as step3

if __name__ == "__main__":
    print("\n" + "="*58)
    print("  PIPELINE CHUẨN BỊ DATASET — PTIT MAJOR ADVISOR")
    print("="*58 + "\n")

    step1()
    print("\n")
    step2()
    print("\n")
    step3()

    print("\n" + "="*58)
    print("  PIPELINE HOÀN THÀNH")
    print("  File sử dụng để train: data/processed/train_ready.csv")
    print("="*58)