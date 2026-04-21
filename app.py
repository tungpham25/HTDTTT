"""
app.py — Flask web application
Khởi động: python app.py
Yêu cầu: pip install flask
Model phải được train trước: python run_pipeline.py && python ml/train_model.py
"""

from flask import Flask, render_template, request, redirect, url_for, jsonify
import sqlite3, json, uuid, os
from datetime import datetime
from pathlib import Path

app = Flask(__name__)
DB  = Path("data/responses.db")
DB.parent.mkdir(exist_ok=True)

# ── Load model 1 lần duy nhất khi khởi động ─────────────────────────────────
try:
    from ml.predict import HybridAdvisor
    advisor = HybridAdvisor()
    print("✅ Model loaded OK")
except Exception as e:
    advisor = None
    print(f"⚠️  Chưa có model: {e}\n   Chạy: python run_pipeline.py && python ml/train_model.py")

# ── 48 câu hỏi với giải thích trung lập ─────────────────────────────────────
QUESTIONS = [
    # R — Realistic
    {"id":"R1","text":"Kiểm tra chất lượng linh kiện trước khi xuất xưởng","hint":"Phản ánh mức độ bạn thích làm việc với vật thể và quy trình kiểm soát chất lượng thực tế."},
    {"id":"R2","text":"Lát gạch hoặc ốp tường","hint":"Phản ánh mức độ bạn thích công việc tay chân đòi hỏi sự khéo léo và chính xác."},
    {"id":"R3","text":"Làm việc trên giàn khoan dầu ngoài khơi","hint":"Phản ánh mức độ bạn thích môi trường làm việc thực địa, chấp nhận thách thức thể chất."},
    {"id":"R4","text":"Lắp ráp các linh kiện điện tử","hint":"Phản ánh mức độ bạn thích làm việc với thiết bị, linh kiện và hệ thống vật lý."},
    {"id":"R5","text":"Vận hành máy mài trong nhà máy","hint":"Phản ánh mức độ bạn thích điều khiển và vận hành máy móc công nghiệp."},
    {"id":"R6","text":"Sửa vòi nước bị hỏng","hint":"Phản ánh mức độ bạn thích tự tay khắc phục sự cố và sửa chữa thiết bị."},
    {"id":"R7","text":"Lắp ráp sản phẩm trong dây chuyền sản xuất","hint":"Phản ánh mức độ bạn thích công việc có quy trình rõ ràng và kết quả hữu hình."},
    {"id":"R8","text":"Lắp sàn gỗ trong nhà","hint":"Phản ánh mức độ bạn thích công việc thủ công đòi hỏi kỹ năng thực hành."},
    # I — Investigative
    {"id":"I1","text":"Nghiên cứu cấu trúc cơ thể người","hint":"Phản ánh mức độ bạn thích tìm hiểu sâu về cách mọi thứ hoạt động ở cấp độ nền tảng."},
    {"id":"I2","text":"Nghiên cứu hành vi động vật","hint":"Phản ánh mức độ bạn thích quan sát, phân tích và tìm ra quy luật từ dữ liệu."},
    {"id":"I3","text":"Nghiên cứu thực vật hoặc động vật","hint":"Phản ánh mức độ bạn thích tiến hành điều tra khoa học và đưa ra kết luận có căn cứ."},
    {"id":"I4","text":"Phát triển phương pháp điều trị y tế mới","hint":"Phản ánh mức độ bạn thích giải quyết vấn đề phức tạp bằng tư duy sáng tạo và logic."},
    {"id":"I5","text":"Thực hiện nghiên cứu sinh học","hint":"Phản ánh mức độ bạn thích làm việc với dữ liệu và phương pháp khoa học chính xác."},
    {"id":"I6","text":"Nghiên cứu cá voi và sinh vật biển","hint":"Phản ánh mức độ bạn thích khám phá những điều chưa được biết đến."},
    {"id":"I7","text":"Làm việc trong phòng thí nghiệm sinh học","hint":"Phản ánh mức độ bạn thích môi trường làm việc đòi hỏi tư duy phân tích và sự tỉ mỉ."},
    {"id":"I8","text":"Lập bản đồ đáy đại dương","hint":"Phản ánh mức độ bạn thích công việc điều tra, khám phá và xử lý thông tin phức tạp."},
    # A — Artistic
    {"id":"A1","text":"Chỉ huy một dàn hợp xướng","hint":"Phản ánh mức độ bạn thích điều phối và thể hiện cảm xúc qua nghệ thuật."},
    {"id":"A2","text":"Đạo diễn một vở kịch","hint":"Phản ánh mức độ bạn thích dẫn dắt quá trình sáng tạo và truyền đạt ý tưởng nghệ thuật."},
    {"id":"A3","text":"Thiết kế tranh minh họa cho tạp chí","hint":"Phản ánh mức độ bạn thích biểu đạt thông điệp qua hình ảnh và thẩm mỹ trực quan."},
    {"id":"A4","text":"Sáng tác một bài hát","hint":"Phản ánh mức độ bạn thích tạo ra tác phẩm mang dấu ấn cá nhân và cảm xúc."},
    {"id":"A5","text":"Viết sách hoặc kịch bản","hint":"Phản ánh mức độ bạn thích xây dựng ý tưởng và biểu đạt chúng qua ngôn từ sáng tạo."},
    {"id":"A6","text":"Chơi nhạc cụ","hint":"Phản ánh mức độ bạn thích thể hiện bản thân qua hoạt động nghệ thuật đòi hỏi kỹ năng."},
    {"id":"A7","text":"Thực hiện các cảnh mạo hiểm trong phim","hint":"Phản ánh mức độ bạn thích công việc biểu diễn và sáng tạo nội dung trực quan."},
    {"id":"A8","text":"Thiết kế bối cảnh cho sân khấu","hint":"Phản ánh mức độ bạn thích kết hợp thẩm mỹ và không gian để kể chuyện."},
    # S — Social
    {"id":"S1","text":"Tư vấn hướng nghiệp cho mọi người","hint":"Phản ánh mức độ bạn thích lắng nghe và hỗ trợ người khác định hướng cuộc sống."},
    {"id":"S2","text":"Làm tình nguyện tại tổ chức phi lợi nhuận","hint":"Phản ánh mức độ bạn thích đóng góp cho cộng đồng và làm việc vì lợi ích chung."},
    {"id":"S3","text":"Giúp người nghiện ma túy hoặc rượu bia","hint":"Phản ánh mức độ bạn thích đồng hành và hỗ trợ người khác vượt qua khó khăn."},
    {"id":"S4","text":"Hướng dẫn ai đó các bài tập thể dục","hint":"Phản ánh mức độ bạn thích chia sẻ kiến thức và đào tạo trực tiếp cho người khác."},
    {"id":"S5","text":"Giúp người có vấn đề gia đình","hint":"Phản ánh mức độ bạn thích tham gia giải quyết vấn đề cá nhân và cảm xúc của người khác."},
    {"id":"S6","text":"Giám sát hoạt động của trẻ em tại trại hè","hint":"Phản ánh mức độ bạn thích chăm sóc, hướng dẫn và tạo môi trường an toàn cho người khác."},
    {"id":"S7","text":"Dạy trẻ em học đọc","hint":"Phản ánh mức độ bạn thích truyền đạt kiến thức và nhìn thấy người khác tiến bộ."},
    {"id":"S8","text":"Hỗ trợ người cao tuổi trong sinh hoạt hằng ngày","hint":"Phản ánh mức độ bạn thích chăm sóc và tạo ra sự khác biệt trực tiếp cho cuộc sống người khác."},
    # E — Enterprising
    {"id":"E1","text":"Bán nhượng quyền kinh doanh nhà hàng","hint":"Phản ánh mức độ bạn thích đàm phán, thuyết phục và tạo ra thỏa thuận có giá trị."},
    {"id":"E2","text":"Bán hàng tại trung tâm thương mại","hint":"Phản ánh mức độ bạn thích tương tác để tạo ra giao dịch và đạt chỉ tiêu doanh số."},
    {"id":"E3","text":"Quản lý hoạt động của khách sạn","hint":"Phản ánh mức độ bạn thích điều phối nhiều bộ phận để đạt hiệu quả hoạt động tổng thể."},
    {"id":"E4","text":"Điều hành salon làm đẹp","hint":"Phản ánh mức độ bạn thích khởi nghiệp và vận hành doanh nghiệp của riêng mình."},
    {"id":"E5","text":"Quản lý một phòng ban trong công ty lớn","hint":"Phản ánh mức độ bạn thích lãnh đạo nhóm và đưa ra quyết định có ảnh hưởng lớn."},
    {"id":"E6","text":"Quản lý cửa hàng quần áo","hint":"Phản ánh mức độ bạn thích kết hợp quản lý, bán hàng và tạo trải nghiệm khách hàng."},
    {"id":"E7","text":"Môi giới bất động sản","hint":"Phản ánh mức độ bạn thích tìm kiếm cơ hội, xây dựng quan hệ và chốt thỏa thuận."},
    {"id":"E8","text":"Điều hành cửa hàng đồ chơi","hint":"Phản ánh mức độ bạn thích vận hành kinh doanh và tạo ra trải nghiệm tích cực cho khách hàng."},
    # C — Conventional
    {"id":"C1","text":"Tính bảng lương hằng tháng cho văn phòng","hint":"Phản ánh mức độ bạn thích xử lý số liệu chính xác và đảm bảo mọi thứ đúng theo quy định."},
    {"id":"C2","text":"Kiểm kê kho hàng bằng máy tính cầm tay","hint":"Phản ánh mức độ bạn thích theo dõi và tổ chức thông tin một cách có hệ thống."},
    {"id":"C3","text":"Dùng phần mềm tạo hóa đơn cho khách hàng","hint":"Phản ánh mức độ bạn thích sử dụng công cụ số để xử lý giao dịch chính xác và hiệu quả."},
    {"id":"C4","text":"Lưu trữ hồ sơ nhân viên","hint":"Phản ánh mức độ bạn thích duy trì hệ thống lưu trữ có tổ chức và dễ tra cứu."},
    {"id":"C5","text":"Tính toán và ghi lại số liệu thống kê","hint":"Phản ánh mức độ bạn thích làm việc với con số, phân tích định lượng và báo cáo."},
    {"id":"C6","text":"Vận hành máy tính tiền","hint":"Phản ánh mức độ bạn thích xử lý giao dịch theo quy trình chuẩn và chính xác."},
    {"id":"C7","text":"Xử lý giao dịch ngân hàng cho khách hàng","hint":"Phản ánh mức độ bạn thích môi trường làm việc có quy trình rõ ràng và trách nhiệm cao về độ chính xác."},
    {"id":"C8","text":"Lưu sổ sách nhập xuất hàng hóa","hint":"Phản ánh mức độ bạn thích ghi chép có hệ thống và kiểm soát luồng thông tin."},
]

MAJOR_FULLNAME = {
    "CNTT":   "Công nghệ Thông tin",
    "ATTT":   "An toàn Thông tin",
    "MKT":    "Marketing",
    "TMDT":   "Thương mại Điện tử",
    "TTDPT":  "Truyền thông Đa phương tiện",
    "KETOAN": "Kế toán",
}

# ── Database ─────────────────────────────────────────────────────────────────
def init_db():
    with sqlite3.connect(DB) as c:
        c.execute("""CREATE TABLE IF NOT EXISTS responses (
            id          TEXT PRIMARY KEY,
            created_at  TEXT,
            r_norm REAL, i_norm REAL, a_norm REAL,
            s_norm REAL, e_norm REAL, c_norm REAL,
            ml_top1     TEXT,
            kbs_top1    TEXT,
            top3        TEXT,
            probas      TEXT,
            confirmed   INTEGER,
            confidence  REAL,
            warnings    TEXT,
            explanation TEXT,
            specialization TEXT,
            rules_fired TEXT,
            raw_answers TEXT,
            rule_details TEXT,
            eligible_majors TEXT
        )""")

def save_response(sid, result, raw):
    p = result["riasec"]
    with sqlite3.connect(DB) as c:
        c.execute("""INSERT INTO responses VALUES
            (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", (
            sid, datetime.now().isoformat(),
            p["R"], p["I"], p["A"], p["S"], p["E"], p["C"],
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

def get_response(sid):
    with sqlite3.connect(DB) as c:
        c.row_factory = sqlite3.Row
        return c.execute("SELECT * FROM responses WHERE id=?", (sid,)).fetchone()

def get_all_responses():
    with sqlite3.connect(DB) as c:
        c.row_factory = sqlite3.Row
        return c.execute(
            "SELECT * FROM responses ORDER BY created_at DESC"
        ).fetchall()

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return redirect(url_for("survey"))

@app.route("/survey")
def survey():
    return render_template("survey.html", questions=QUESTIONS)

@app.route("/submit", methods=["POST"])
def submit():
    if advisor is None:
        return "Model chưa được train. Chạy: python run_pipeline.py && python ml/train_model.py", 503

    raw = {}
    for q in QUESTIONS:
        val = request.form.get(q["id"])
        if val is None:
            return f"Thiếu câu trả lời: {q['id']}", 400
        raw[q["id"]] = int(val)

    result = advisor.advise(raw)
    sid    = str(uuid.uuid4())
    save_response(sid, result, raw)
    return redirect(url_for("result_page", sid=sid))

@app.route("/result/<sid>")
def result_page(sid):
    row = get_response(sid)
    if not row:
        return "Không tìm thấy kết quả", 404

    top3   = json.loads(row["top3"])
    probas = json.loads(row["probas"])
    warns  = json.loads(row["warnings"])
    fired  = json.loads(row["rules_fired"])

    # FIX: Dùng row["col"] thay vì getattr(row, "col")
    riasec = {d: round(row[f"{d.lower()}_norm"] * 100) for d in "RIASEC"}

    top3_display = [
        {"code": m, "name": MAJOR_FULLNAME.get(m, m), "pct": round(probas.get(m, 0)*100, 1)}
        for m in top3
    ]

    # Parse rule_details nếu có
    try:
        rule_details = json.loads(row["rule_details"]) if row["rule_details"] else []
    except (json.JSONDecodeError, KeyError):
        rule_details = []

    try:
        eligible_majors = json.loads(row["eligible_majors"]) if row["eligible_majors"] else []
    except (json.JSONDecodeError, KeyError):
        eligible_majors = []

    return render_template("result.html",
        sid=sid,
        top3=top3_display,
        ml_top1=MAJOR_FULLNAME.get(row["ml_top1"], row["ml_top1"]),
        ml_top1_code=row["ml_top1"],
        kbs_top1=MAJOR_FULLNAME.get(row["kbs_top1"], row["kbs_top1"]),
        kbs_top1_code=row["kbs_top1"],
        adjusted=row["ml_top1"] != row["kbs_top1"],
        confirmed=bool(row["confirmed"]),
        confidence=round(row["confidence"] * 100, 1),
        warnings=warns,
        explanation=row["explanation"],
        specialization=row["specialization"],
        rules_fired=fired,
        rule_details=rule_details,
        eligible_majors=[MAJOR_FULLNAME.get(m, m) for m in eligible_majors],
        eligible_codes=eligible_majors,
        riasec=riasec,
        created_at=row["created_at"][:16].replace("T", " "),
    )

@app.route("/dashboard")
def dashboard():
    rows = get_all_responses()
    total = len(rows)

    dist = {}
    for m in MAJOR_FULLNAME: dist[m] = 0
    for r in rows:
        k = r["kbs_top1"]
        if k in dist: dist[k] += 1

    # FIX: Dùng r["col"] thay vì getattr(r, "col")
    avg = {d: 0.0 for d in "RIASEC"}
    if total:
        for r in rows:
            for d in "RIASEC":
                avg[d] += r[f"{d.lower()}_norm"]
        for d in "RIASEC":
            avg[d] = round(avg[d] / total * 100, 1)

    recent = []
    for r in rows[:20]:
        top3 = json.loads(r["top3"])
        recent.append({
            "id":       r["id"][:8],
            "time":     r["created_at"][:16].replace("T", " "),
            "top1":     MAJOR_FULLNAME.get(r["kbs_top1"], r["kbs_top1"]),
            "top2":     MAJOR_FULLNAME.get(top3[1], top3[1]) if len(top3)>1 else "",
            "conf":     round(r["confidence"] * 100, 1),
            "confirmed": bool(r["confirmed"]),
        })

    conf_dist = {"<20":0, "20-40":0, "40-60":0, "60-80":0, ">80":0}
    for r in rows:
        c = r["confidence"] * 100
        if c < 20: conf_dist["<20"] += 1
        elif c < 40: conf_dist["20-40"] += 1
        elif c < 60: conf_dist["40-60"] += 1
        elif c < 80: conf_dist["60-80"] += 1
        else: conf_dist[">80"] += 1

    adjusted = sum(1 for r in rows if r["ml_top1"] != r["kbs_top1"])

    return render_template("dashboard.html",
        total=total,
        dist=dist,
        dist_labels=list(MAJOR_FULLNAME.values()),
        dist_values=[dist.get(k,0) for k in MAJOR_FULLNAME],
        avg_riasec=avg,
        recent=recent,
        conf_dist=conf_dist,
        adjusted=adjusted,
        adjusted_pct=round(adjusted/total*100, 1) if total else 0,
    )

@app.route("/api/responses")
def api_responses():
    rows = get_all_responses()
    return jsonify([dict(r) for r in rows])

@app.route("/model-dashboard")
def model_dashboard():
    import json
    rp = Path("models/eval_report.json")
    report = json.loads(rp.read_text(encoding="utf-8")) if rp.exists() else None
    return render_template("model_dashboard.html", report=report, enumerate=enumerate)

# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    init_db()
    app.run(debug=True, port=5000)