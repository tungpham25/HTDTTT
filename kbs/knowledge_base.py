"""
kbs/knowledge_base.py
Hệ thống tri thức (KBS) cho tư vấn ngành học PTIT — 45 luật.

THAY ĐỔI CHÍNH so với phiên bản cũ:
  - KBS đánh giá TOÀN BỘ 6 ngành với A1–A6, không chỉ top1
  - Nếu ML top1 fail → KBS chọn ngành xác suất cao nhất TRONG SỐ ngành hợp lệ
  - A7/A8 loại ngành khỏi danh sách hợp lệ
  - A9 ưu tiên đặc biệt TTDPT
  - KBS lọc bằng tri thức domain, ML xếp hạng trong số đã lọc

Nguồn tri thức:
  - Holland (1997): lý thuyết RIASEC, Holland Code
  - Rounds et al. (2008): điểm O*NET cho 6 nghề, thang 1–7
  - Rounds et al. (2021): ngưỡng confidence < 0.10

Cấu trúc 45 luật:
  Nhóm A — Xác nhận        (A1–A9)
  Nhóm B — Tiebreaker      (B1–B15)
  Nhóm C — Cảnh báo        (C1–C8)
  Nhóm D — Giải thích      (D1–D9)
  Nhóm E — Gợi ý chuyên sâu(E1–E4)
"""

from dataclasses import dataclass, field
from scipy.stats import rankdata
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# CẤU TRÚC DỮ LIỆU
# ─────────────────────────────────────────────────────────────────────────────

MAJORS = ["CNTT", "ATTT", "MKT", "TMDT", "TTDPT", "KETOAN"]

MAJOR_FULLNAME = {
    "CNTT":   "Công nghệ Thông tin",
    "ATTT":   "An toàn Thông tin",
    "MKT":    "Marketing",
    "TMDT":   "Thương mại Điện tử",
    "TTDPT":  "Truyền thông Đa phương tiện",
    "KETOAN": "Kế toán",
}

@dataclass
class UserProfile:
    R: float; I: float; A: float
    S: float; E: float; C: float

    def as_array(self) -> np.ndarray:
        return np.array([self.R, self.I, self.A, self.S, self.E, self.C])

    def rank(self) -> dict[str, int]:
        """Xếp hạng 6 chiều, rank 1 = cao nhất."""
        ranks = rankdata(-self.as_array(), method="min")
        return dict(zip("RIASEC", ranks))


@dataclass
class MLResult:
    top3: list[str]
    probas: dict[str, float]

    @property
    def top1(self) -> str: return self.top3[0]

    @property
    def top2(self) -> str: return self.top3[1]

    @property
    def confidence(self) -> float:
        return self.probas[self.top1] - self.probas[self.top2]


@dataclass
class KBSVerdict:
    final_top3: list[str]
    confirmed: bool
    warnings: list[str]           = field(default_factory=list)
    tiebreaker_applied: bool      = False
    tiebreaker_rule: str          = ""
    explanation: str              = ""
    detail_explanation: str       = ""
    specialization_hint: str      = ""
    rules_fired: list[str]        = field(default_factory=list)
    eligible_majors: list[str]    = field(default_factory=list)
    rule_details: list[dict]      = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# NGƯỠNG
# ─────────────────────────────────────────────────────────────────────────────

CONFIDENCE_LOW   = 0.10
FLAT_PROFILE_STD = 0.08
DOMINANT         = 0.55
WEAK             = 0.30
VERY_DOMINANT    = 0.75
VERY_WEAK        = 0.20


# ─────────────────────────────────────────────────────────────────────────────
# NHÓM A — XÁC NHẬN (A1–A9)
# ─────────────────────────────────────────────────────────────────────────────

def rule_A1_confirm_CNTT(p: UserProfile) -> tuple[bool, str, str]:
    """A1 — CNTT (IC): rank(I)≤3 AND rank(C)≤3."""
    r = p.rank()
    ok = r["I"] <= 3 and r["C"] <= 3
    name = "A1: Xác nhận CNTT"
    detail = f"Holland Code IC yêu cầu I và C trong top 3. Hiện tại: rank(I)={r['I']}, rank(C)={r['C']}."
    return ok, name, detail

def rule_A2_confirm_ATTT(p: UserProfile) -> tuple[bool, str, str]:
    """A2 — ATTT (CI): rank(C)≤3 AND rank(I)≤3."""
    r = p.rank()
    ok = r["C"] <= 3 and r["I"] <= 3
    name = "A2: Xác nhận ATTT"
    detail = f"Holland Code CI yêu cầu C và I trong top 3. Hiện tại: rank(C)={r['C']}, rank(I)={r['I']}."
    return ok, name, detail

def rule_A3_confirm_MKT(p: UserProfile) -> tuple[bool, str, str]:
    """A3 — MKT (EC): rank(E)≤2 AND rank(C)≤4."""
    r = p.rank()
    ok = r["E"] <= 2 and r["C"] <= 4
    name = "A3: Xác nhận MKT"
    detail = f"Holland Code EC yêu cầu E top 2 (E=7.00 dominant) và C top 4. Hiện tại: rank(E)={r['E']}, rank(C)={r['C']}."
    return ok, name, detail

def rule_A4_confirm_TMDT(p: UserProfile) -> tuple[bool, str, str]:
    """A4 — TMDT (CE): rank(C)≤3 AND rank(E)≤3."""
    r = p.rank()
    ok = r["C"] <= 3 and r["E"] <= 3
    name = "A4: Xác nhận TMDT"
    detail = f"Holland Code CE yêu cầu C và E trong top 3 (C≈E). Hiện tại: rank(C)={r['C']}, rank(E)={r['E']}."
    return ok, name, detail

def rule_A5_confirm_TTDPT(p: UserProfile) -> tuple[bool, str, str]:
    """A5 — TTDPT (AR): rank(A)=1."""
    r = p.rank()
    ok = r["A"] == 1
    name = "A5: Xác nhận TTDPT"
    detail = f"Holland Code AR yêu cầu A đứng số 1 (A=7.00 tối đa duy nhất). Hiện tại: rank(A)={r['A']}."
    return ok, name, detail

def rule_A6_confirm_KETOAN(p: UserProfile) -> tuple[bool, str, str]:
    """A6 — KETOAN (CEI): rank(C)=1 AND rank(E)≤4 AND rank(I)≤4."""
    r = p.rank()
    ok = r["C"] == 1 and r["E"] <= 4 and r["I"] <= 4
    name = "A6: Xác nhận KETOAN"
    detail = f"Holland Code CEI yêu cầu C đứng số 1 (C=7.00). Hiện tại: rank(C)={r['C']}, rank(E)={r['E']}, rank(I)={r['I']}."
    return ok, name, detail

def rule_A7_negative_CNTT(p: UserProfile) -> tuple[bool, str, str]:
    """A7 — Phủ nhận CNTT: rank(E)=1 AND rank(I)≥4."""
    r = p.rank()
    neg = r["E"] == 1 and r["I"] >= 4
    name = "A7: Phủ nhận CNTT"
    detail = f"CNTT có E=1.87 thấp nhất, I=6.05 cao nhất. Profile E=rank{r['E']}, I=rank{r['I']} mâu thuẫn hoàn toàn."
    return neg, name, detail

def rule_A8_negative_MKT(p: UserProfile) -> tuple[bool, str, str]:
    """A8 — Phủ nhận MKT: rank(R)≤2 AND rank(I)≤2 AND rank(E)≥5."""
    r = p.rank()
    neg = r["R"] <= 2 and r["I"] <= 2 and r["E"] >= 5
    name = "A8: Phủ nhận MKT"
    detail = f"MKT có R=1.00 thấp nhất tuyệt đối. Profile R+I cao, E thấp = kỹ sư/nhà khoa học, mâu thuẫn MKT."
    return neg, name, detail

def rule_A9_strong_confirm_TTDPT(p: UserProfile) -> tuple[bool, str, str]:
    """A9 — Xác nhận mạnh TTDPT khi A_norm ≥ 0.75."""
    strong = p.A >= VERY_DOMINANT
    name = "A9: Xác nhận mạnh TTDPT"
    detail = f"A_norm={p.A:.2f} {'≥' if strong else '<'} 0.75. TTDPT là ngành duy nhất có A=7.00 tối đa."
    return strong, name, detail


# ─────────────────────────────────────────────────────────────────────────────
# NHÓM B — TIEBREAKER (B1–B15)
# ─────────────────────────────────────────────────────────────────────────────

def rule_B1_tiebreak_CNTT_vs_ATTT(p: UserProfile) -> tuple[str | None, str]:
    if p.I > p.C:
        return "CNTT", "B1 bậc 1: I > C → CNTT (Holland Code IC: I trội hơn C)"
    elif p.C > p.I:
        return "ATTT", "B1 bậc 1: C > I → ATTT (Holland Code CI: C trội hơn I)"
    else:
        if p.E > 0.35:
            return "ATTT", "B1 bậc 2: I=C nhưng E cao → ATTT (E ATTT=2.85 > CNTT=1.87)"
        elif p.E < 0.25:
            return "CNTT", "B1 bậc 2: I=C và E thấp → CNTT"
        else:
            return None, "B1 bậc 3: I=C, E không phân biệt → KBS nhường ML"

def rule_B2_tiebreak_MKT_vs_TMDT(p: UserProfile) -> tuple[str, str]:
    return ("MKT",  "B2: E−C > 0.10 → MKT (E dominant, Δ=2.30)") if p.E - p.C > 0.10 \
      else ("TMDT", "B2: C và E cân bằng → TMDT (Δ=0.19)")

def rule_B3_tiebreak_ATTT_vs_KETOAN(p: UserProfile) -> tuple[str, str]:
    if p.R > 0.35:
        return "ATTT",   "B3: R > 0.35 → kỹ thuật → ATTT (R: ATTT=3.56 vs KETOAN=1.14)"
    elif p.E >= p.I:
        return "KETOAN", "B3: E≥I → tài chính → KETOAN"
    return "ATTT", "B3: I>E → phân tích bảo mật → ATTT"

def rule_B4_tiebreak_TMDT_vs_KETOAN(p: UserProfile) -> tuple[str, str]:
    return ("KETOAN", "B4: C−E > 0.15 → KETOAN (C dominant, Δ=3.13)") if p.C - p.E > 0.15 \
      else ("TMDT",   "B4: C và E cân bằng → TMDT (Δ=0.19)")

def rule_B5_tiebreak_TTDPT_vs_MKT(p: UserProfile) -> tuple[str, str]:
    return ("TTDPT", "B5: A≥E → sáng tạo → TTDPT (A=7.00 vs E=7.00)") if p.A >= p.E \
      else ("MKT",   "B5: E>A → kinh doanh → MKT")

def rule_B6_tiebreak_CNTT_vs_KETOAN(p: UserProfile) -> tuple[str, str]:
    return ("CNTT",   "B6: I−E > 0.10 → kỹ thuật → CNTT (Δ(I-E)=4.18)") if p.I - p.E > 0.10 \
      else ("KETOAN", "B6: I và E gần nhau → KETOAN (Δ(I-E)=−0.30)")

def rule_B7_tiebreak_ATTT_vs_TMDT(p: UserProfile) -> tuple[str, str]:
    return ("ATTT", "B7: I>E → bảo mật kỹ thuật → ATTT (I: ATTT=5.40 vs TMDT=1.61)") if p.I > p.E \
      else ("TMDT", "B7: E≥I → thương mại → TMDT")

def rule_B8_tiebreak_TTDPT_vs_KETOAN(p: UserProfile) -> tuple[str, str]:
    return ("TTDPT",  "B8: A>C → sáng tạo → TTDPT (A=7 vs C=7, profile đối lập)") if p.A > p.C \
      else ("KETOAN", "B8: C≥A → tổ chức → KETOAN")

def rule_B9_tiebreak_CNTT_vs_MKT(p: UserProfile) -> tuple[str, str]:
    return ("CNTT", "B9: I>E → phân tích → CNTT (I=6.05 vs E=7.00)") if p.I > p.E \
      else ("MKT",  "B9: E≥I → kinh doanh → MKT")

def rule_B10_tiebreak_CNTT_vs_TMDT(p: UserProfile) -> tuple[str, str]:
    return ("CNTT", "B10: I>E → kỹ thuật → CNTT (I=6.05 vs TMDT I=1.61)") if p.I > p.E \
      else ("TMDT", "B10: E≥I → thương mại → TMDT")

def rule_B11_tiebreak_CNTT_vs_TTDPT(p: UserProfile) -> tuple[str, str]:
    return ("TTDPT", "B11: A>I → sáng tạo → TTDPT (A=7.00 vs I=6.05)") if p.A > p.I \
      else ("CNTT",  "B11: I≥A → phân tích → CNTT")

def rule_B12_tiebreak_ATTT_vs_MKT(p: UserProfile) -> tuple[str, str]:
    return ("MKT",  "B12: E≥0.55 → kinh doanh → MKT (E: MKT=7.00)")  if p.E >= DOMINANT \
      else ("ATTT", "B12: E<0.55 → bảo mật → ATTT (E: ATTT=2.85)")

def rule_B13_tiebreak_ATTT_vs_TTDPT(p: UserProfile) -> tuple[str, str]:
    return ("TTDPT", "B13: A>C → sáng tạo → TTDPT (A=7.00 vs ATTT A=1.34)") if p.A > p.C \
      else ("ATTT",  "B13: C≥A → kiểm soát → ATTT")

def rule_B14_tiebreak_MKT_vs_KETOAN(p: UserProfile) -> tuple[str, str]:
    return ("MKT",    "B14: E>C → kinh doanh → MKT (E dominant Δ=2.30)") if p.E > p.C \
      else ("KETOAN", "B14: C≥E → kế toán → KETOAN (C dominant Δ=3.13)")

def rule_B15_tiebreak_TMDT_vs_TTDPT(p: UserProfile) -> tuple[str, str]:
    return ("TTDPT", "B15: A≥0.55 → đa phương tiện → TTDPT (A=7.00)") if p.A >= DOMINANT \
      else ("TMDT",  "B15: A<0.55 → thương mại → TMDT (A=1.88)")


# ─────────────────────────────────────────────────────────────────────────────
# NHÓM C — CẢNH BÁO (C1–C8)
# ─────────────────────────────────────────────────────────────────────────────

def rule_C1_flat_profile(p: UserProfile) -> tuple[bool, str]:
    std = float(np.std(p.as_array()))
    if std < FLAT_PROFILE_STD:
        return True, f"C1: Profile phẳng (std={std:.3f}<0.08) — 6 chiều gần bằng nhau, kết quả ít tin cậy"
    return False, ""

def rule_C2_contradictory_CNTT(p: UserProfile) -> tuple[bool, str]:
    warn = p.E >= DOMINANT and p.I <= WEAK
    return warn, "C2: E cao + I thấp → mâu thuẫn với CNTT (CNTT cần I cao, E thấp)"

def rule_C3_contradictory_TTDPT(p: UserProfile) -> tuple[bool, str]:
    warn = p.A <= WEAK and p.C >= DOMINANT
    return warn, "C3: A thấp + C cao → mâu thuẫn với TTDPT (TTDPT cần A dominant)"

def rule_C4_contradictory_KETOAN(p: UserProfile) -> tuple[bool, str]:
    warn = p.A >= DOMINANT and p.C <= WEAK
    return warn, "C4: A cao + C thấp → mâu thuẫn với KETOAN (KETOAN cần C dominant)"

def rule_C5_extreme_single_dim(p: UserProfile) -> tuple[bool, str]:
    arr = p.as_array()
    max_val = arr.max()
    others  = np.delete(arr, arr.argmax())
    if max_val >= VERY_DOMINANT and (others <= WEAK).all():
        dim = "RIASEC"[arr.argmax()]
        return True, f"C5: {dim}={max_val:.2f} cực cao, các chiều còn lại thấp — có thể response bias"
    return False, ""

def rule_C6_high_S_for_tech(p: UserProfile) -> tuple[bool, str]:
    r = p.rank()
    return r["S"] == 1, "C6: S cao nhất — bất thường với ngành kỹ thuật (CNTT S=1.81, ATTT S=2.11)"

def rule_C7_high_R_for_business(p: UserProfile) -> tuple[bool, str]:
    r = p.rank()
    return r["R"] == 1, "C7: R cao nhất — bất thường với ngành kinh doanh (MKT R=1.00)"

def rule_C8_low_C_for_KETOAN(p: UserProfile) -> tuple[bool, str]:
    return p.C <= VERY_WEAK, f"C8: C={p.C:.2f} quá thấp — KETOAN yêu cầu C dominant (O*NET C=7.00)"


# ─────────────────────────────────────────────────────────────────────────────
# NHÓM D — GIẢI THÍCH (D1–D9) — CẢI THIỆN CHI TIẾT
# ─────────────────────────────────────────────────────────────────────────────

DIM_DESC = {
    "R": ("Thực hành (Realistic)", "thích làm việc với công cụ, máy móc, môi trường thực địa"),
    "I": ("Phân tích (Investigative)", "thích nghiên cứu, tìm hiểu, giải quyết vấn đề bằng logic"),
    "A": ("Sáng tạo (Artistic)", "thích biểu đạt ý tưởng, thiết kế, sáng tạo nội dung"),
    "S": ("Xã hội (Social)", "thích giúp đỡ, tương tác, làm việc với con người"),
    "E": ("Kinh doanh (Enterprising)", "thích quản lý, thuyết phục, đạt mục tiêu kinh doanh"),
    "C": ("Tổ chức (Conventional)", "thích sắp xếp, xử lý số liệu, làm việc theo quy trình"),
}

MAJOR_REASON = {
    "CNTT":   "CNTT (Holland Code IC) yêu cầu I (Phân tích) và C (Tổ chức) cao — phù hợp với lập trình, xây dựng hệ thống, giải quyết vấn đề kỹ thuật.",
    "ATTT":   "ATTT (Holland Code CI) yêu cầu C (Tổ chức) và I (Phân tích) cao — phù hợp với bảo mật, kiểm soát hệ thống, phân tích mối đe dọa.",
    "MKT":    "MKT (Holland Code EC) yêu cầu E (Kinh doanh) dominant và C (Tổ chức) — phù hợp với chiến lược thương hiệu, truyền thông, quản lý chiến dịch.",
    "TMDT":   "TMDT (Holland Code CE) yêu cầu C (Tổ chức) và E (Kinh doanh) cân bằng — phù hợp với vận hành nền tảng số, quản lý giao dịch trực tuyến.",
    "TTDPT":  "TTDPT (Holland Code AR) yêu cầu A (Sáng tạo) dominant — phù hợp với thiết kế đồ họa, sản xuất nội dung đa phương tiện, kỹ xảo.",
    "KETOAN": "KETOAN (Holland Code CEI) yêu cầu C (Tổ chức) dominant — phù hợp với kế toán, kiểm toán, phân tích tài chính.",
}

def generate_explanation(p: UserProfile, final_top1: str, ml_top1: str,
                         eligible: list[str], rule_details: list[dict]) -> str:
    """D1–D6: Sinh giải thích chi tiết dựa trên profile và kết quả."""
    r = p.rank()
    arr = p.as_array()
    dims = list("RIASEC")

    # Tìm 2 chiều cao nhất
    sorted_dims = sorted(zip(dims, arr), key=lambda x: -x[1])
    top2_dims = sorted_dims[:2]

    lines = []

    # Profile analysis
    dim1_code, dim1_val = top2_dims[0]
    dim2_code, dim2_val = top2_dims[1]
    dim1_name, dim1_desc = DIM_DESC[dim1_code]
    dim2_name, dim2_desc = DIM_DESC[dim2_code]

    lines.append(f"Profile RIASEC của bạn nổi bật nhất ở chiều {dim1_name} ({dim1_val:.0%}) "
                 f"và {dim2_name} ({dim2_val:.0%}).")
    lines.append(f"— {dim1_name}: {dim1_desc}.")
    lines.append(f"— {dim2_name}: {dim2_desc}.")

    # Why this major
    lines.append("")
    lines.append(MAJOR_REASON.get(final_top1, f"Ngành {final_top1} phù hợp với profile của bạn."))

    # If adjusted
    if final_top1 != ml_top1:
        lines.append("")
        lines.append(f"Lưu ý: ML dự đoán ban đầu là {MAJOR_FULLNAME.get(ml_top1, ml_top1)}, "
                     f"nhưng profile của bạn không thỏa mãn điều kiện Holland Code của ngành đó. "
                     f"KBS đã đánh giá toàn bộ 6 ngành và chọn {MAJOR_FULLNAME.get(final_top1, final_top1)} "
                     f"— ngành có xác suất ML cao nhất trong số các ngành phù hợp với profile.")

    return "\n".join(lines)


def rule_D7_explain_CNTT_vs_ATTT(p: UserProfile) -> str:
    if p.I > p.C:
        return (f"I={p.I:.0%} cao hơn C={p.C:.0%} → thiên về xây dựng và phân tích hệ thống "
                f"→ CNTT (Holland IC: I trội hơn C) phù hợp hơn ATTT (Holland CI).")
    return (f"C={p.C:.0%} cao hơn I={p.I:.0%} → thiên về kiểm soát, tuân thủ quy trình "
            f"→ ATTT (Holland CI: C trội hơn I) phù hợp hơn CNTT (Holland IC).")

def rule_D8_explain_MKT_vs_TMDT(p: UserProfile) -> str:
    gap = p.E - p.C
    if gap > 0.10:
        return (f"E={p.E:.0%} vượt C={p.C:.0%} (chênh {gap:.0%}) → thiên về chiến lược thương hiệu "
                f"→ MKT (Holland EC: E dominant, Δ=2.30 trong O*NET).")
    return (f"E={p.E:.0%} và C={p.C:.0%} cân bằng (chênh {abs(gap):.0%}) → thiên về vận hành nền tảng "
            f"→ TMDT (Holland CE: C≈E, Δ=0.19 trong O*NET).")

def rule_D9_explain_low_confidence(p: UserProfile, top3: list[str]) -> str:
    return (f"Profile của bạn cân bằng giữa {MAJOR_FULLNAME.get(top3[0], top3[0])} "
            f"và {MAJOR_FULLNAME.get(top3[1], top3[1])}. "
            f"Kết quả là gợi ý định hướng — nên tham khảo thêm chương trình đào tạo từng ngành "
            f"để quyết định.")


# ─────────────────────────────────────────────────────────────────────────────
# NHÓM E — GỢI Ý CHUYÊN SÂU (E1–E4) — CẢI THIỆN CHI TIẾT
# ─────────────────────────────────────────────────────────────────────────────

def rule_E1_CNTT_track(p: UserProfile) -> str:
    r = p.rank()
    if r["A"] <= 2:
        return ("E1: Chiều A (Sáng tạo) nổi bật → gợi ý hướng Frontend / UI/UX Development. "
                "Bạn kết hợp được tư duy phân tích (I) với khả năng thiết kế (A), "
                "phù hợp xây dựng giao diện người dùng.")
    if r["S"] <= 2:
        return ("E1: Chiều S (Xã hội) nổi bật → gợi ý hướng UX Research / Product Management. "
                "Bạn kết hợp kỹ năng kỹ thuật với khả năng thấu hiểu người dùng.")
    if r["E"] <= 2:
        return ("E1: Chiều E (Kinh doanh) nổi bật → gợi ý hướng Tech Entrepreneurship / Team Lead. "
                "Bạn có tố chất lãnh đạo kết hợp nền tảng kỹ thuật.")
    return ("E1: Profile cân bằng trong ngành CNTT → gợi ý hướng Software Engineering / Backend. "
            "Đây là hướng đi chung phù hợp với nền tảng I+C mạnh.")

def rule_E2_ATTT_track(p: UserProfile) -> str:
    r = p.rank()
    if r["R"] <= 2:
        return ("E2: Chiều R (Thực hành) nổi bật → gợi ý hướng Penetration Testing / Ethical Hacking. "
                "Bạn thích thực hành trực tiếp trên hệ thống, phù hợp kiểm thử xâm nhập.")
    if r["E"] <= 2:
        return ("E2: Chiều E (Kinh doanh) nổi bật → gợi ý hướng Security Management / GRC. "
                "Bạn kết hợp kiến thức bảo mật với khả năng quản lý rủi ro.")
    return ("E2: Profile I dominant → gợi ý hướng Security Research / Threat Intelligence. "
            "Bạn thiên về phân tích sâu mối đe dọa và nghiên cứu lỗ hổng.")

def rule_E3_MKT_track(p: UserProfile) -> str:
    r = p.rank()
    if r["I"] <= 2:
        return ("E3: Chiều I (Phân tích) nổi bật → gợi ý hướng Data Marketing / Marketing Analytics. "
                "Bạn kết hợp dữ liệu với chiến lược marketing, đo lường hiệu quả chiến dịch.")
    if r["A"] <= 2:
        return ("E3: Chiều A (Sáng tạo) nổi bật → gợi ý hướng Creative Marketing / Content Strategy. "
                "Bạn thiên về sáng tạo nội dung và xây dựng câu chuyện thương hiệu.")
    if r["S"] <= 2:
        return ("E3: Chiều S (Xã hội) nổi bật → gợi ý hướng Brand Community / PR. "
                "Bạn có khả năng xây dựng quan hệ và gắn kết cộng đồng thương hiệu.")
    return ("E3: Profile E dominant → gợi ý hướng Brand Management / Campaign Strategy. "
            "Đây là hướng quản lý chiến dịch marketing tổng thể.")

def rule_E4_KETOAN_track(p: UserProfile) -> str:
    r = p.rank()
    if r["I"] <= 3 and p.I >= 0.45:
        return ("E4: Chiều I (Phân tích) nổi bật → gợi ý hướng Financial Analysis / Kế toán quản trị dữ liệu. "
                "Bạn kết hợp khả năng phân tích với nền tảng tài chính.")
    if r["E"] <= 2:
        return ("E4: Chiều E (Kinh doanh) nổi bật → gợi ý hướng Management Accounting / Kiểm toán nội bộ. "
                "Bạn có tố chất quản lý trong lĩnh vực tài chính.")
    return ("E4: Profile C dominant → gợi ý hướng Financial Accounting / Kiểm toán độc lập. "
            "Đây là hướng truyền thống phù hợp với khả năng tổ chức và tuân thủ quy trình.")


# ─────────────────────────────────────────────────────────────────────────────
# MAP TRA CỨU
# ─────────────────────────────────────────────────────────────────────────────

CONFIRM_MAP = {
    "CNTT": rule_A1_confirm_CNTT, "ATTT": rule_A2_confirm_ATTT,
    "MKT":  rule_A3_confirm_MKT,  "TMDT": rule_A4_confirm_TMDT,
    "TTDPT": rule_A5_confirm_TTDPT, "KETOAN": rule_A6_confirm_KETOAN,
}

NEGATIVE_MAP = {"CNTT": rule_A7_negative_CNTT, "MKT": rule_A8_negative_MKT}

TIEBREAKER_MAP = {
    frozenset(["CNTT",  "ATTT"]):   rule_B1_tiebreak_CNTT_vs_ATTT,
    frozenset(["MKT",   "TMDT"]):   rule_B2_tiebreak_MKT_vs_TMDT,
    frozenset(["ATTT",  "KETOAN"]): rule_B3_tiebreak_ATTT_vs_KETOAN,
    frozenset(["TMDT",  "KETOAN"]): rule_B4_tiebreak_TMDT_vs_KETOAN,
    frozenset(["TTDPT", "MKT"]):    rule_B5_tiebreak_TTDPT_vs_MKT,
    frozenset(["CNTT",  "KETOAN"]): rule_B6_tiebreak_CNTT_vs_KETOAN,
    frozenset(["ATTT",  "TMDT"]):   rule_B7_tiebreak_ATTT_vs_TMDT,
    frozenset(["TTDPT", "KETOAN"]): rule_B8_tiebreak_TTDPT_vs_KETOAN,
    frozenset(["CNTT",  "MKT"]):    rule_B9_tiebreak_CNTT_vs_MKT,
    frozenset(["CNTT",  "TMDT"]):   rule_B10_tiebreak_CNTT_vs_TMDT,
    frozenset(["CNTT",  "TTDPT"]):  rule_B11_tiebreak_CNTT_vs_TTDPT,
    frozenset(["ATTT",  "MKT"]):    rule_B12_tiebreak_ATTT_vs_MKT,
    frozenset(["ATTT",  "TTDPT"]):  rule_B13_tiebreak_ATTT_vs_TTDPT,
    frozenset(["MKT",   "KETOAN"]): rule_B14_tiebreak_MKT_vs_KETOAN,
    frozenset(["TMDT",  "TTDPT"]):  rule_B15_tiebreak_TMDT_vs_TTDPT,
}

DETAIL_MAP = {
    frozenset(["CNTT", "ATTT"]): rule_D7_explain_CNTT_vs_ATTT,
    frozenset(["MKT",  "TMDT"]): rule_D8_explain_MKT_vs_TMDT,
}

WARNING_TECH = {"CNTT", "ATTT"}
WARNING_BIZ  = {"MKT",  "KETOAN"}

TRACK_MAP = {
    "CNTT": rule_E1_CNTT_track, "ATTT": rule_E2_ATTT_track,
    "MKT":  rule_E3_MKT_track,  "KETOAN": rule_E4_KETOAN_track,
}


# ─────────────────────────────────────────────────────────────────────────────
# ENGINE — Áp dụng toàn bộ luật (LOGIC MỚI)
# ─────────────────────────────────────────────────────────────────────────────

def apply_kbs(profile: UserProfile, ml_result: MLResult) -> KBSVerdict:
    """
    Engine mới — KBS đánh giá toàn bộ 6 ngành, không chỉ top1 của ML.

    Luồng:
      1. C1, C5 → kiểm tra chất lượng phản hồi
      2. A1–A6 → đánh giá TOÀN BỘ 6 ngành → danh sách eligible
      3. A7, A8 → loại ngành bị phủ nhận khỏi eligible
      4. A9     → xác nhận mạnh TTDPT (ưu tiên đặc biệt)
      5. Xây dựng top3 từ eligible (xếp hạng theo ML proba)
      6. B1–B15 → tiebreaker nếu top1 và top2 eligible gần nhau
      7. C2–C8  → cảnh báo mâu thuẫn
      8. D1–D9  → sinh giải thích chi tiết
      9. E1–E4  → gợi ý chuyên sâu (chỉ khi confirmed)
    """
    verdict = KBSVerdict(final_top3=ml_result.top3.copy(), confirmed=False)
    fired = []
    rule_details = []

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 1: Kiểm tra chất lượng phản hồi (C1, C5)
    # ══════════════════════════════════════════════════════════════════════════
    for fn, rid in [(rule_C1_flat_profile, "C1"), (rule_C5_extreme_single_dim, "C5")]:
        w, msg = fn(profile)
        if w:
            verdict.warnings.append(msg)
            fired.append(rid)
            rule_details.append({"rule": rid, "type": "warning", "detail": msg})

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 2: Đánh giá TOÀN BỘ 6 ngành với A1–A6
    # ══════════════════════════════════════════════════════════════════════════
    eligibility = {}  # major -> True/False
    for major in MAJORS:
        if major in CONFIRM_MAP:
            ok, name, detail = CONFIRM_MAP[major](profile)
            eligibility[major] = ok
            fired.append(name.split(":")[0].strip())  # "A1", "A2", etc.
            rule_details.append({
                "rule": name, "type": "confirm" if ok else "reject",
                "major": major, "detail": detail,
                "result": "PASS" if ok else "FAIL"
            })

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 3: Phủ nhận A7, A8 — loại ngành khỏi eligible
    # ══════════════════════════════════════════════════════════════════════════
    for major, fn in NEGATIVE_MAP.items():
        if eligibility.get(major, False):
            neg, name, detail = fn(profile)
            if neg:
                eligibility[major] = False
                verdict.warnings.append(f"{name}: {detail}")
                fired.append(name.split(":")[0].strip())
                rule_details.append({
                    "rule": name, "type": "negate",
                    "major": major, "detail": detail
                })

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 4: A9 xác nhận mạnh TTDPT
    # ══════════════════════════════════════════════════════════════════════════
    strong_ttdpt, a9_name, a9_detail = rule_A9_strong_confirm_TTDPT(profile)
    if strong_ttdpt:
        fired.append("A9")
        rule_details.append({
            "rule": a9_name, "type": "strong_confirm",
            "major": "TTDPT", "detail": a9_detail
        })

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 5: Xây dựng top3 từ eligible
    # ══════════════════════════════════════════════════════════════════════════
    eligible_majors = [m for m in MAJORS if eligibility.get(m, False)]
    verdict.eligible_majors = eligible_majors

    # Sort eligible theo ML proba (cao → thấp)
    eligible_sorted = sorted(eligible_majors,
                              key=lambda m: ml_result.probas.get(m, 0),
                              reverse=True)

    # A9 override: TTDPT lên top1 nếu eligible
    if strong_ttdpt and "TTDPT" in eligible_sorted:
        eligible_sorted.remove("TTDPT")
        eligible_sorted.insert(0, "TTDPT")

    if len(eligible_sorted) == 0:
        # Không ngành nào hợp lệ → giữ ML, không xác nhận
        verdict.final_top3 = ml_result.top3.copy()
        verdict.confirmed = False
        verdict.warnings.append(
            "Không có ngành nào thỏa mãn đầy đủ điều kiện Holland Code với profile của bạn. "
            "Kết quả ML được giữ làm tham khảo."
        )
    elif ml_result.top1 in eligible_sorted:
        # ML top1 hợp lệ → giữ nguyên, xác nhận
        verdict.final_top3 = ml_result.top3.copy()
        verdict.confirmed = True
    else:
        # ML top1 KHÔNG hợp lệ → KBS chọn từ eligible
        final_top3 = eligible_sorted[:3]
        # Pad nếu thiếu
        if len(final_top3) < 3:
            remaining = [m for m in ml_result.top3 if m not in final_top3]
            final_top3.extend(remaining[:3 - len(final_top3)])
        verdict.final_top3 = final_top3[:3]
        verdict.confirmed = True

    current_top1 = verdict.final_top3[0]

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 6: Tiebreaker B1–B15
    # ══════════════════════════════════════════════════════════════════════════
    if len(verdict.final_top3) >= 2:
        top1 = verdict.final_top3[0]
        top2 = verdict.final_top3[1]
        top1_p = ml_result.probas.get(top1, 0)
        top2_p = ml_result.probas.get(top2, 0)

        if abs(top1_p - top2_p) < CONFIDENCE_LOW:
            pair = frozenset([top1, top2])
            if pair in TIEBREAKER_MAP:
                winner, reason = TIEBREAKER_MAP[pair](profile)
                verdict.tiebreaker_applied = True
                verdict.tiebreaker_rule = reason
                rule_id = reason.split(":")[0].strip()
                fired.append(rule_id)
                rule_details.append({
                    "rule": rule_id, "type": "tiebreaker",
                    "detail": reason
                })

                if winner is not None and winner != top1:
                    verdict.final_top3[0], verdict.final_top3[1] = \
                        verdict.final_top3[1], verdict.final_top3[0]

                if pair in DETAIL_MAP:
                    verdict.detail_explanation = DETAIL_MAP[pair](profile)
                    fired.append("D7/D8")
            else:
                verdict.detail_explanation = rule_D9_explain_low_confidence(
                    profile, verdict.final_top3)
                fired.append("D9")

    current_top1 = verdict.final_top3[0]

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 7: Cảnh báo mâu thuẫn C2–C8
    # ══════════════════════════════════════════════════════════════════════════
    warn_checks = []
    if current_top1 == "CNTT":
        warn_checks.append((rule_C2_contradictory_CNTT, "C2"))
    elif current_top1 == "TTDPT":
        warn_checks.append((rule_C3_contradictory_TTDPT, "C3"))
    elif current_top1 == "KETOAN":
        warn_checks.append((rule_C4_contradictory_KETOAN, "C4"))
        warn_checks.append((rule_C8_low_C_for_KETOAN, "C8"))

    if current_top1 in WARNING_TECH:
        warn_checks.append((rule_C6_high_S_for_tech, "C6"))
    if current_top1 in WARNING_BIZ:
        warn_checks.append((rule_C7_high_R_for_business, "C7"))

    for fn, rid in warn_checks:
        w, msg = fn(profile)
        if w:
            verdict.warnings.append(msg)
            fired.append(rid)
            rule_details.append({"rule": rid, "type": "warning", "detail": msg})

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 8: Giải thích D1–D6 (chi tiết)
    # ══════════════════════════════════════════════════════════════════════════
    verdict.explanation = generate_explanation(
        profile, current_top1, ml_result.top1, eligible_majors, rule_details
    )
    fired.append(f"D{MAJORS.index(current_top1)+1}" if current_top1 in MAJORS else "D?")

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 9: Gợi ý chuyên sâu E1–E4 (chỉ khi confirmed)
    # ══════════════════════════════════════════════════════════════════════════
    if verdict.confirmed and current_top1 in TRACK_MAP:
        verdict.specialization_hint = TRACK_MAP[current_top1](profile)
        e_idx = list(TRACK_MAP.keys()).index(current_top1) + 1
        fired.append(f"E{e_idx}")

    verdict.rules_fired = fired
    verdict.rule_details = rule_details
    return verdict