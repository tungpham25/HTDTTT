import pandas as pd
from pathlib import Path

path = Path("data\\raw\\Interests.xlsx")
df = pd.read_excel(path, engine="openpyxl", nrows=5)

print("Kiểu index cột:", type(df.columns))
print("Tên cột gốc:")
for i, col in enumerate(df.columns):
    print(f"  [{i}] {repr(col)}  (type: {type(col).__name__})")

print("\n5 dòng đầu:")
print(df.to_string())