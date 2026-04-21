"""
seed_data.py
Tạo dữ liệu mẫu cho dashboard và test toàn bộ hệ thống.

Chạy: python seed_data.py
Kết quả: tạo bản ghi trong data/responses.db

Kịch bản test bao gồm:
  - Profile rõ ràng cho mỗi ngành
  - Trường hợp 50-50 (CNTT/ATTT)
  - Profile phẳng (C1 cảnh báo)
  - Profile bị KBS phủ nhận (A7, A8)
  - ML sai → KBS điều chỉnh
"""

import sys, os
sys.path.insert(0, os.getcwd())

import json, uuid, sqlite3, random
from datetime import datetime, timedelta
from pathlib import Path

# ── Khởi tạo ──────────────────────────────────────────────────────────────────
DB = Path("data/responses.db")
DB.parent.mkdir(exist_ok=True)

try:
    from ml.predict import HybridAdvisor
    advisor = HybridAdvisor()
    print("✅ Model loaded")
except Exception as e:
    print(f"❌ Không load được model: {e}")
    print("   Chạy: python run_pipeline.py && python ml/train_model.py")
    sys.exit(1)

# ── Profile templates cho 6 ngành ────────────────────────────────────────────
TEMPLATES = {
    "CNTT_clear":   {"R":3,"I":5,"A":1,"S":1,"E":1,"C":4, "label":"CNTT rõ ràng"},
    "ATTT_clear":   {"R":3,"I":4,"A":1,"S":2,"E":2,"C":5, "label":"ATTT rõ ràng"},
    "MKT_clear":    {"R":1,"I":2,"A":3,"S":3,"E":5,"C":3, "label":"MKT rõ ràng"},
    "TMDT_clear":   {"R":2,"I":2,"A":2,"S":2,"E":4,"C":4, "label":"TMDT rõ ràng"},
    "TTDPT_clear":  {"R":3,"I":2,"A":5,"S":2,"E":2,"C":2, "label":"TTDPT rõ ràng"},
    "KETOAN_clear": {"R":1,"I":3,"A":1,"S":2,"E":3,"C":5, "label":"KETOAN rõ ràng"},
    "CNTT_ATTT_50": {"R":3,"I":4,"A":1,"S":1,"E":2,"C":4, "label":"CNTT/ATTT 50-50"},
    "MKT_TMDT_50":  {"R":1,"I":2,"A":2,"S":3,"E":4,"C":4, "label":"MKT/TMDT gần nhau"},
    "FLAT_profile": {"R":3,"I":3,"A":3,"S":3,"E":3,"C":3, "label":"Profile phẳng (C1)"},
    "TTDPT_strong": {"R":3,"I":2,"A":5,"S":2,"E":1,"C":2, "label":"TTDPT A rất cao"},
}

def make_responses(template: dict, noise: int = 0) -> dict:
    responses = {}
    for dim in "RIASEC":
        base = template[dim]
        for i in range(1, 9):
            val = base + random.randint(-noise, noise)
            val = max(1, min(5, val))
            responses[f"{dim}{i}"] = val
    return responses

def save_to_db(result: dict, raw: dict, created_at: str):
    p = result["riasec"]
    with sqlite3.connect(DB) as c:
        c.execute("""CREATE TABLE IF NOT EXISTS responses (
            id TEXT PRIMARY KEY, created_at TEXT,
            r_norm REAL, i_norm REAL, a_norm REAL,
            s_norm REAL, e_norm REAL, c_norm REAL,
            ml_top1 TEXT, kbs_top1 TEXT,
            top3 TEXT, probas TEXT,
            confirmed INTEGER, confidence REAL,
            warnings TEXT, explanation TEXT,
            specialization TEXT, rules_fired TEXT,
            raw_answers TEXT, rule_details TEXT, eligible_majors TEXT
        )""")
        c.execute("""INSERT OR IGNORE INTO responses VALUES
            (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", (
            str(uuid.uuid4()), created_at,
            p["R"], p["I"], p["A"],
            p["S"], p["E"], p["C"],
            result["ml_top1"], result["kbs_top1"],
            json.dumps(result["top3"]),
            json.dumps(result["probas"]),
            1 if result["confirmed"] else 0,
            result["ml_confidence"],
            json.dumps(result["warnings"]),
            result["explanation"],
            result["specialization"],
            json.dumps(result["rules_fired"]),
            json.dumps(raw),
            json.dumps(result.get("rule_details", [])),
            json.dumps(result.get("eligible_majors", [])),
        ))

# ── Sinh dữ liệu ──────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("  SEED DATA — Tạo dữ liệu mẫu cho dashboard")
print("="*55)

base_time = datetime.now() - timedelta(days=7)
total_created = 0

for key, tmpl in TEMPLATES.items():
    reps = 5 if "50" in key or "FLAT" in key or "strong" in key else 10
    label = tmpl["label"]
    print(f"\n[{label}] Tạo {reps} mẫu...")

    for i in range(reps):
        noise = 0 if "FLAT" in key else 1
        raw    = make_responses(tmpl, noise=noise)
        result = advisor.advise(raw)

        offset  = timedelta(hours=random.randint(0, 168))
        created = (base_time + offset).isoformat()

        save_to_db(result, raw, created)
        total_created += 1

        top1 = result["kbs_top1"]
        conf = result["ml_confidence"]
        adj  = f" → KBS:{result['kbs_top1']}" if result["adjusted"] else ""
        warn = f" ⚠️ {len(result['warnings'])} cảnh báo" if result["warnings"] else ""
        eligible = result.get("eligible_majors", [])
        print(f"  [{i+1:02d}] ML:{result['ml_top1']:<8}{adj:<14} conf={conf:.3f}"
              f"  eligible={eligible}{warn}")

print(f"\n{'='*55}")
print(f"  Đã tạo {total_created} bản ghi → data/responses.db")
print(f"  Mở dashboard: http://localhost:5000/dashboard")
print("="*55)

# ── Kịch bản test đặc biệt ────────────────────────────────────────────────────
print("\n\nKỊCH BẢN TEST ĐẶC BIỆT:")
print("─"*55)

special_cases = [
    {
        "name": "Test 1 — CNTT profile hoàn hảo",
        "resp": {f"{d}{i}": v for d,v in zip("RIASEC",[2,5,1,1,1,4]) for i in range(1,9)},
        "expect": "CNTT, confirmed, A1 pass"
    },
    {
        "name": "Test 2 — E cao, I thấp → ML có thể đoán sai CNTT",
        "resp": {f"{d}{i}": v for d,v in zip("RIASEC",[2,2,2,3,5,3]) for i in range(1,9)},
        "expect": "KBS điều chỉnh nếu ML đoán CNTT (A7 phủ nhận)"
    },
    {
        "name": "Test 3 — Profile phẳng → C1 cảnh báo",
        "resp": {f"{d}{i}": 3 for d in "RIASEC" for i in range(1,9)},
        "expect": "C1 cảnh báo"
    },
    {
        "name": "Test 4 — TTDPT A rất cao → A9 xác nhận mạnh",
        "resp": {f"{d}{i}": v for d,v in zip("RIASEC",[3,2,5,2,1,2]) for i in range(1,9)},
        "expect": "TTDPT, A9 kích hoạt"
    },
    {
        "name": "Test 5 — CNTT/ATTT 50-50 → B1 tiebreaker",
        "resp": {f"{d}{i}": v for d,v in zip("RIASEC",[3,4,1,1,2,4]) for i in range(1,9)},
        "expect": "B1 kích hoạt"
    },
]

for tc in special_cases:
    result = advisor.advise(tc["resp"])
    print(f"\n{tc['name']}")
    print(f"  Kỳ vọng     : {tc['expect']}")
    print(f"  ML dự đoán  : {result['ml_top1']}")
    print(f"  KBS kết luận: {result['kbs_top1']}  (confirmed={result['confirmed']}, adjusted={result['adjusted']})")
    print(f"  Eligible    : {result.get('eligible_majors', [])}")
    print(f"  Luật đã kích: {result['rules_fired']}")
    print(f"  Confidence  : {result['ml_confidence']:.4f}")
    if result["warnings"]:
        for w in result["warnings"]:
            print(f"  ⚠️  {w}")
    print(f"  RIASEC: { {d: result['riasec'][d] for d in 'RIASEC'} }")