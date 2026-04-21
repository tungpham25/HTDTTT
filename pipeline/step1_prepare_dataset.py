"""
pipeline/step1_prepare_dataset.py
Bước 1: Lọc chất lượng + tính điểm RIASEC + gán nhãn bằng Pearson r + sampling.

Đầu vào:
  data/raw/data.csv          — dataset RIASEC từ openpsychometrics.org
  data/raw/Interests.xlsx    — O*NET database, tải từ onetcenter.org/database.html

Đầu ra:
  data/processed/train_final.csv

Phương pháp gán nhãn:
  Pearson profile correlation (McCloy et al., 1999; Rounds et al., 2021)
  Nhãn = argmax(r) giữa vector RIASEC người dùng và 6 profile nghề O*NET
"""

import pandas as pd
import numpy as np
from pathlib import Path

RAW_PATH     = Path("./data/raw/data.csv")
ONET_PATH    = Path("./data/raw/Interests.xlsx")
OUT_PATH     = Path("./data/processed/train_final.csv")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

DIMS         = list("RIASEC")
RIASEC_ITEMS = {d: [f"{d}{i}" for i in range(1, 9)] for d in DIMS}
ALL_ITEMS    = [item for items in RIASEC_ITEMS.values() for item in items]
FAKE_WORDS   = ["VCL6", "VCL9", "VCL12"]
MAJORS       = ["CNTT", "ATTT", "MKT", "TMDT", "TTDPT", "KETOAN"]
SAMPLES_PER_CLASS = 5_000

# SOC codes — ánh xạ ngành PTIT → nghề O*NET (nhóm tự quyết định)
SOC_CODES = {
    "CNTT":   "15-1252.00",   # Software Developers
    "ATTT":   "15-1212.00",   # Information Security Analysts
    "MKT":    "11-2021.00",   # Marketing Managers
    "TMDT":   "13-1199.06",   # Online Merchants
    "TTDPT":  "27-1014.00",   # Special Effects Artists & Animators
    "KETOAN": "13-2011.00",   # Accountants and Auditors
}

ELEMENT_MAP = {
    "Realistic": "R", "Investigative": "I", "Artistic": "A",
    "Social": "S", "Enterprising": "E", "Conventional": "C",
}

# Ngưỡng fit quality — Rounds et al. (2021) O*NET Interest Profiler Manual
FIT_BINS   = [-1.0, 0.607, 0.728, 1.0]
FIT_LABELS = ["Good Fit", "Great Fit", "Best Fit"]


# ── 1. Đọc profile nghề từ O*NET Interests.xlsx ───────────────────────────────
def load_onet_profiles() -> dict[str, np.ndarray]:
    df = pd.read_excel(ONET_PATH, engine="openpyxl")
    df.columns = df.columns.str.strip()

    col_map = {}
    for col in df.columns:
        cl = col.lower()
        if "o*net" in cl or "soc" in cl:   col_map[col] = "soc_code"
        elif "element name" in cl:          col_map[col] = "element_name"
        elif "scale id" in cl:              col_map[col] = "scale_id"
        elif "data value" in cl:            col_map[col] = "data_value"
    df = df.rename(columns=col_map)

    df_oi = df[df["scale_id"] == "OI"].copy()
    profiles = {}

    for major, soc in SOC_CODES.items():
        sub = df_oi[df_oi["soc_code"] == soc]
        profile = {}
        for _, row in sub.iterrows():
            dim = ELEMENT_MAP.get(str(row["element_name"]).strip())
            if dim:
                profile[dim] = float(row["data_value"])
        if len(profile) == 6:
            profiles[major] = np.array([profile[d] for d in DIMS])
            scores = "  ".join(f"{d}={profile[d]:.2f}" for d in DIMS)
            print(f"  [onet] {major:<10} ({soc})  {scores}")
        else:
            print(f"  ⚠️  {major} ({soc}): chỉ tìm thấy {len(profile)}/6 chiều")

    return profiles


# ── 2. Load & filter dataset ──────────────────────────────────────────────────
def load_and_filter() -> pd.DataFrame:
    df = pd.read_csv(RAW_PATH, sep="\t", low_memory=False)
    print(f"[load]   {len(df):>8,} dòng")
    n = len(df)

    # Lọc VCL fake words — careless responding
    vcl = [c for c in FAKE_WORDS if c in df.columns]
    if vcl:
        df = df[df[vcl].sum(axis=1) == 0].copy()
        print(f"[vcl]    {len(df):>8,} dòng  (loại {n - len(df):,})")
        n = len(df)

    # Lọc speeders — < 2 giây/item × 48 items = 96 giây
    if "testelapse" in df.columns:
        df = df[df["testelapse"] >= 96].copy()
        print(f"[speed]  {len(df):>8,} dòng  (loại {n - len(df):,})")
        n = len(df)

    # Validate thang Likert [1, 5]
    items = [c for c in ALL_ITEMS if c in df.columns]
    mask  = df[items].apply(lambda col: col.between(1, 5)).all(axis=1)
    df    = df[mask].copy()
    print(f"[valid]  {len(df):>8,} dòng  (loại {n - len(df):,})")

    # Giữ age, gender, education làm metadata — không lọc tuổi
    # (audit cho thấy lọc tuổi loại 61.2% data mà không đổi phân phối nhãn)
    return df.reset_index(drop=True)


# ── 3. Tính RIASEC scores ─────────────────────────────────────────────────────
def compute_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    X_score = Σ(item_i − 1), i=1..8   →  [0, 32]   (Holland, 1997)
    X_norm  = X_score / 32             →  [0, 1]    (dùng làm input ML, không dùng cho Pearson r)
    """
    for d in DIMS:
        items = [c for c in RIASEC_ITEMS[d] if c in df.columns]
        df[f"{d}_score"] = df[items].sub(1).sum(axis=1).astype("float32")
        df[f"{d}_norm"]  = (df[f"{d}_score"] / 32).astype("float32")
    return df


# ── 4. Gán nhãn bằng Pearson correlation ─────────────────────────────────────
def assign_labels(df: pd.DataFrame, profiles: dict) -> pd.DataFrame:
    """
    Pearson r đo hình dạng tương đối của profile RIASEC.
    Nguồn: McCloy et al. (1999); Rounds et al. (2021) IP Manual
    "The comparison is based on the shape or pattern of the whole profiles
     instead of the absolute level of each score."

    Dùng X_score (0–32) làm input — Pearson r tự xử lý thang đo khác nhau
    thông qua bước trừ trung bình, nên không cần normalize trước.
    """
    # Ma trận người dùng: (N, 6)
    person_mat = df[[f"{d}_score" for d in DIMS]].values.astype(float)

    # Ma trận O*NET: (K, 6)
    majors_ok  = [m for m in MAJORS if m in profiles]
    onet_mat   = np.stack([profiles[m] for m in majors_ok])

    # Tính Pearson r vectorized — center về mean rồi tính cosine similarity
    p_c   = person_mat - person_mat.mean(axis=1, keepdims=True)
    o_c   = onet_mat   - onet_mat.mean(axis=1, keepdims=True)
    p_u   = p_c / (np.linalg.norm(p_c, axis=1, keepdims=True) + 1e-10)
    o_u   = o_c / (np.linalg.norm(o_c, axis=1, keepdims=True) + 1e-10)
    r_mat = p_u @ o_u.T   # (N, K)

    # Lưu r từng ngành
    for j, m in enumerate(majors_ok):
        df[f"r_{m}"] = r_mat[:, j].astype("float32")

    # Gán nhãn = argmax(r)
    sorted_r = np.sort(r_mat, axis=1)
    df["assigned_major"] = [majors_ok[i] for i in r_mat.argmax(axis=1)]
    df["r_best"]         = sorted_r[:, -1].astype("float32")
    df["r_second"]       = sorted_r[:, -2].astype("float32")
    df["confidence"]     = (sorted_r[:, -1] - sorted_r[:, -2]).astype("float32")

    # Fit quality theo ngưỡng O*NET (Rounds et al., 2021)
    df["fit_quality"] = pd.cut(df["r_best"], bins=FIT_BINS, labels=FIT_LABELS)
    return df


# ── 5. Báo cáo phân phối ─────────────────────────────────────────────────────
def report(df: pd.DataFrame):
    print(f"\n{'─'*55}")
    print(f"  PHÂN PHỐI NHÃN ({len(df):,} mẫu)")
    print(f"{'─'*55}")
    dist = df["assigned_major"].value_counts()
    for m in MAJORS:
        n   = dist.get(m, 0)
        pct = n / len(df)
        bar = "█" * int(pct * 35)
        print(f"  {m:<10} {n:>7,}  {pct:>5.1%}  {bar}")

    print(f"\n  Fit quality:")
    for q in ["Best Fit", "Great Fit", "Good Fit"]:
        n   = (df["fit_quality"] == q).sum()
        print(f"  {q:<12} {n:>7,}  ({n/len(df):.1%})")

    c = df["confidence"]
    print(f"\n  Confidence: mean={c.mean():.4f}  median={c.median():.4f}  "
          f"min={c.min():.4f}  max={c.max():.4f}")


# ── 6. Per-class top-K sampling ───────────────────────────────────────────────
def sample_classes(df: pd.DataFrame, k: int = SAMPLES_PER_CLASS) -> pd.DataFrame:
    """
    Mỗi ngành giữ k mẫu có r_best cao nhất.
    k=5000 vì ngành ít nhất (ATTT) có ~5900 mẫu — không cần oversampling.
    """
    print(f"\n[sample] Per-class top-{k} (by r_best):")
    parts = []
    for m in MAJORS:
        sub    = df[df["assigned_major"] == m].sort_values("r_best", ascending=False)
        n_take = min(len(sub), k)
        parts.append(sub.head(n_take))
        flag = "  ⚠️  ít hơn k" if len(sub) < k else ""
        print(f"  {m:<10} {len(sub):>7,} → lấy {n_take:>5,}{flag}")

    balanced = pd.concat(parts).sample(frac=1, random_state=42)
    ratio    = balanced["assigned_major"].value_counts()
    print(f"\n  max/min = {ratio.max()/ratio.min():.2f}x  |  tổng {len(balanced):,} mẫu")
    return balanced.reset_index(drop=True)


# ── 7. Chọn cột xuất ─────────────────────────────────────────────────────────
def select_columns(df: pd.DataFrame) -> pd.DataFrame:
    score_cols = [f"{d}_score" for d in DIMS]   # input ML
    norm_cols  = [f"{d}_norm"  for d in DIMS]   # input ML (chuẩn hóa)
    r_cols     = [f"r_{m}"     for m in MAJORS]  # metadata
    meta_cols  = ["r_best", "r_second", "confidence", "fit_quality", "assigned_major"]
    demo_cols  = [c for c in ["age", "gender", "education"] if c in df.columns]
    keep = score_cols + norm_cols + r_cols + meta_cols + demo_cols
    return df[[c for c in keep if c in df.columns]].copy()


# ── MAIN ──────────────────────────────────────────────────────────────────────
def run():
    print("=" * 55)
    print("  BƯỚC 1: CHUẨN BỊ DATASET")
    print("=" * 55)

    print(f"\n[onet]   Đọc profile nghề từ {ONET_PATH}:")
    profiles = load_onet_profiles()
    if len(profiles) < len(MAJORS):
        print(f"  🔴 Chỉ load được {len(profiles)}/{len(MAJORS)} nghề. Dừng lại.")
        return

    df = load_and_filter()
    df = compute_scores(df)

    print(f"\n[label]  Tính Pearson r với {len(profiles)} O*NET profiles...")
    df = assign_labels(df, profiles)

    report(df)
    df = sample_classes(df)

    out = select_columns(df)
    out.index.name = "sample_id"
    out.to_csv(OUT_PATH)
    print(f"\n[done]   Lưu → {OUT_PATH}  ({len(out):,} mẫu | {out.shape[1]} cột)")


if __name__ == "__main__":
    run()