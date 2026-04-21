"""
ml/predict.py
Interface kết hợp ML model + KBS — điểm tích hợp hai thành phần.

Luồng:
  1. Nhận 48 câu trả lời từ chatbot
  2. Tính UserProfile (6 điểm RIASEC normalized)
  3. ML model dự đoán → MLResult (top3 + probas)
  4. KBS áp dụng luật → KBSVerdict (kết quả cuối + giải thích)
  5. Trả về dict kết quả cho chatbot hiển thị

Nguyên lý tích hợp (phiên bản mới):
  - ML dự đoán xác suất 6 ngành
  - KBS đánh giá TOÀN BỘ 6 ngành với Holland Code (A1–A6)
  - Nếu ML top1 hợp lệ → giữ nguyên, KBS xác nhận
  - Nếu ML top1 KHÔNG hợp lệ → KBS chọn ngành xác suất cao nhất
    trong số các ngành thỏa mãn Holland Code
  - KBS lọc bằng tri thức domain, ML xếp hạng trong số đã lọc
"""

import numpy as np
import joblib
from pathlib import Path
from kbs.knowledge_base import (
    UserProfile, MLResult, KBSVerdict, apply_kbs,
    MAJOR_FULLNAME, MAJORS
)

MODEL_DIR = Path("models")
DIMS = list("RIASEC")


class HybridAdvisor:
    """
    Kết hợp ML model và KBS để tư vấn ngành học.

    ML model : dự đoán xác suất 6 ngành từ pattern trong 20.000+ mẫu train.
    KBS      : 45 luật tri thức kiểm tra, lọc, tiebreaker và sinh giải thích.
    """

    def __init__(self):
        self.model     = joblib.load(MODEL_DIR / "classifier.joblib")
        self.le        = joblib.load(MODEL_DIR / "label_encoder.joblib")
        self.feat_cols = joblib.load(MODEL_DIR / "feature_cols.joblib")

    # ── Bước 1: Tính UserProfile từ 48 câu trả lời ───────────────────────────
    @staticmethod
    def _compute_profile(responses: dict[str, int]) -> UserProfile:
        norms = {}
        for d in DIMS:
            items  = [f"{d}{i}" for i in range(1, 9)]
            values = [max(1, min(5, responses.get(k, 3))) for k in items]
            score  = sum(v - 1 for v in values)
            norms[d] = score / 32
        return UserProfile(
            R=norms["R"], I=norms["I"], A=norms["A"],
            S=norms["S"], E=norms["E"], C=norms["C"],
        )

    # ── Bước 2: ML model dự đoán ─────────────────────────────────────────────
    def _ml_predict(self, profile: UserProfile) -> MLResult:
        X = np.array([[profile.R, profile.I, profile.A,
                       profile.S, profile.E, profile.C]])
        proba_arr = self.model.predict_proba(X)[0]
        classes   = self.le.classes_
        probas = {cls: float(p) for cls, p in zip(classes, proba_arr)}
        top3   = sorted(probas, key=probas.get, reverse=True)[:3]
        return MLResult(top3=top3, probas=probas)

    # ── Bước 3: KBS áp dụng luật ─────────────────────────────────────────────
    @staticmethod
    def _apply_kbs(profile: UserProfile, ml_result: MLResult) -> KBSVerdict:
        return apply_kbs(profile, ml_result)

    # ── API chính ─────────────────────────────────────────────────────────────
    def advise(self, responses: dict[str, int]) -> dict:
        profile    = self._compute_profile(responses)
        ml_result  = self._ml_predict(profile)
        verdict    = self._apply_kbs(profile, ml_result)

        return {
            "top3":           verdict.final_top3,
            "probas":         ml_result.probas,
            "ml_top1":        ml_result.top1,
            "kbs_top1":       verdict.final_top3[0],
            "ml_confidence":  round(ml_result.confidence, 4),
            "confirmed":      verdict.confirmed,
            "warnings":       verdict.warnings,
            "explanation":    verdict.explanation,
            "detail":         verdict.detail_explanation,
            "specialization": verdict.specialization_hint,
            "tiebreaker":     verdict.tiebreaker_rule,
            "rules_fired":    verdict.rules_fired,
            "rule_details":   verdict.rule_details,
            "eligible_majors": verdict.eligible_majors,
            "riasec":         {d: round(getattr(profile, d), 4) for d in DIMS},
            "adjusted":       ml_result.top1 != verdict.final_top3[0],
        }

    # ── Format output cho chatbot ─────────────────────────────────────────────
    def format_output(self, result: dict) -> str:
        lines = [
            "=" * 54,
            "  KẾT QUẢ TƯ VẤN CHỌN NGÀNH — PTIT",
            "=" * 54,
            "\n  Top 3 ngành phù hợp:",
        ]

        medals = ["1.", "2.", "3."]
        for i, major in enumerate(result["top3"]):
            prob     = result["probas"].get(major, 0)
            fullname = MAJOR_FULLNAME.get(major, major)
            lines.append(f"  {medals[i]} {fullname} ({major})  —  {prob:.1%}")

        status = "Xác nhận ✓" if result["confirmed"] else "Chưa xác nhận — xem cảnh báo"
        lines.append(f"\n  Trạng thái KBS : {status}")
        lines.append(f"  Confidence ML  : {result['ml_confidence']:.4f}")

        if result["tiebreaker"]:
            lines.append(f"  Tiebreaker     : {result['tiebreaker']}")

        if result["adjusted"]:
            lines.append(
                f"\n  [KBS điều chỉnh] ML dự đoán: {result['ml_top1']} "
                f"→ KBS chọn: {result['kbs_top1']}"
            )
            lines.append(f"  Ngành hợp lệ (Holland Code): {result['eligible_majors']}")

        if result["warnings"]:
            lines.append("\n  Lưu ý:")
            for w in result["warnings"]:
                lines.append(f"    • {w}")

        lines.append(f"\n  Giải thích:\n    {result['explanation']}")
        if result["detail"]:
            lines.append(f"\n    {result['detail']}")

        if result["specialization"]:
            lines.append(f"\n  Hướng chuyên sâu:\n    {result['specialization']}")

        lines.append("\n  Điểm RIASEC (0–1):")
        for d, v in result["riasec"].items():
            bar = "█" * int(v * 20)
            lines.append(f"    {d}  {v:.3f}  {bar}")

        lines.append(f"\n  Luật KBS đã kích hoạt: {result['rules_fired']}")
        lines.append("=" * 54)
        return "\n".join(lines)


# ── Demo ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    advisor = HybridAdvisor()

    test_profiles = {
        "CNTT rõ ràng (I cao, C cao)":
            {f"{d}{i}": v for d, v in zip("RIASEC", [2,5,1,1,1,4]) for i in range(1,9)},
        "CNTT/ATTT 50-50 (I=C)":
            {f"{d}{i}": v for d, v in zip("RIASEC", [3,4,2,2,2,4]) for i in range(1,9)},
        "ML sai — E cao nhưng ML đoán CNTT":
            {f"{d}{i}": v for d, v in zip("RIASEC", [2,2,2,3,5,3]) for i in range(1,9)},
        "Profile phẳng":
            {f"{d}{i}": 3 for d in "RIASEC" for i in range(1,9)},
    }

    for name, responses in test_profiles.items():
        print(f"\n{'='*20} {name} {'='*20}")
        result = advisor.advise(responses)
        print(advisor.format_output(result))