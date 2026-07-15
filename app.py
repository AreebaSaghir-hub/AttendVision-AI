"""
AttendVision AI — Real-Time Attendance System
Uses OpenCV LBPH Face Recognizer (no dlib, no C++ needed)
Author : Talha | TU Ilmenau
Run    : python app.py
Login  : admin / admin123
"""

from flask import (Flask, render_template, Response, jsonify,
                   request, redirect, url_for, session)
import cv2
import sqlite3, os, json, base64, threading
from datetime import datetime, date
from functools import wraps
import numpy as np

app = Flask(__name__)
app.secret_key = "attendvision-secret-2025"

DB_PATH          = os.path.join("database", "attendance.db")
KNOWN_FACES_DIR  = "known_faces"
ADMIN_USER       = "admin"
ADMIN_PASS       = "admin123"
MODEL_PATH       = os.path.join("database", "lbph_model.yml")

# OpenCV detectors
FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ── Demo dataset ──────────────────────────────────────────────────────
DEMO_STUDENTS = [
    ("Ahmed Khan",      "STU001", "ahmed.khan@university.edu",      "Computer Science"),
    ("Sara Malik",      "STU002", "sara.malik@university.edu",       "Computer Science"),
    ("Bilal Hassan",    "STU003", "bilal.hassan@university.edu",     "Electrical Engineering"),
    ("Ayesha Siddiqui","STU004", "ayesha.siddiqui@university.edu",  "Electrical Engineering"),
    ("Omar Farooq",     "STU005", "omar.farooq@university.edu",      "Mechanical Engineering"),
    ("Zara Ahmed",      "STU006", "zara.ahmed@university.edu",       "Business Administration"),
    ("Usman Ali",       "STU007", "usman.ali@university.edu",        "Mathematics"),
    ("Fatima Noor",     "STU008", "fatima.noor@university.edu",      "Computer Science"),
]

DEMO_ATTENDANCE = [
    ("STU001","Ahmed Khan",      "2025-10-20","08:32:15"),
    ("STU002","Sara Malik",      "2025-10-20","08:35:02"),
    ("STU003","Bilal Hassan",    "2025-10-20","08:41:09"),
    ("STU004","Ayesha Siddiqui","2025-10-20","08:50:33"),
    ("STU005","Omar Farooq",     "2025-10-20","09:01:47"),
    ("STU001","Ahmed Khan",      "2025-10-21","08:29:05"),
    ("STU002","Sara Malik",      "2025-10-21","08:33:18"),
    ("STU006","Zara Ahmed",      "2025-10-21","08:45:55"),
    ("STU007","Usman Ali",       "2025-10-21","08:58:22"),
    ("STU001","Ahmed Khan",      "2025-10-22","08:31:40"),
    ("STU003","Bilal Hassan",    "2025-10-22","08:38:17"),
    ("STU004","Ayesha Siddiqui","2025-10-22","08:44:03"),
    ("STU008","Fatima Noor",     "2025-10-22","08:55:29"),
    ("STU002","Sara Malik",      "2025-10-23","08:28:11"),
    ("STU005","Omar Farooq",     "2025-10-23","08:36:44"),
    ("STU006","Zara Ahmed",      "2025-10-23","08:47:38"),
    ("STU007","Usman Ali",       "2025-10-23","09:02:15"),
    ("STU008","Fatima Noor",     "2025-10-23","09:10:07"),
    ("STU001","Ahmed Khan",      "2025-10-24","08:30:22"),
    ("STU002","Sara Malik",      "2025-10-24","08:34:59"),
    ("STU003","Bilal Hassan",    "2025-10-24","08:43:33"),
    ("STU004","Ayesha Siddiqui","2025-10-24","08:52:08"),
    ("STU005","Omar Farooq",     "2025-10-24","08:59:44"),
    ("STU006","Zara Ahmed",      "2025-10-24","09:05:17"),
]

# ── Runtime state ─────────────────────────────────────────────────────
recognizer         = None   # LBPH model
label_to_student   = {}     # int label -> (name, student_id)
recognition_active = False
last_marked        = {}     # name -> datetime


# ─────────────────────────────────────────────────────────────────────
# DATABASE
# ─────────────────────────────────────────────────────────────────────
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    os.makedirs("database",      exist_ok=True)
    os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

    conn = get_db()
    c    = conn.cursor()

    c.execute("""CREATE TABLE IF NOT EXISTS students (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        name          TEXT    NOT NULL,
        student_id    TEXT    UNIQUE NOT NULL,
        email         TEXT,
        department    TEXT,
        label         INTEGER,
        registered_at TEXT,
        photo_path    TEXT
    )""")

    c.execute("""CREATE TABLE IF NOT EXISTS attendance (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id   TEXT NOT NULL,
        student_name TEXT NOT NULL,
        date         TEXT NOT NULL,
        time         TEXT NOT NULL,
        status       TEXT DEFAULT 'Present',
        method       TEXT DEFAULT 'Face Recognition'
    )""")

    c.execute("SELECT COUNT(*) FROM students")
    if c.fetchone()[0] == 0:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for i, (name, sid, email, dept) in enumerate(DEMO_STUDENTS):
            photo = os.path.join(KNOWN_FACES_DIR, f"{sid}.jpg")
            c.execute(
                """INSERT OR IGNORE INTO students
                   (name,student_id,email,department,label,registered_at,photo_path)
                   VALUES(?,?,?,?,?,?,?)""",
                (name, sid, email, dept, i,
                 now, photo if os.path.exists(photo) else None)
            )

    c.execute("SELECT COUNT(*) FROM attendance")
    if c.fetchone()[0] == 0:
        for sid, name, d, t in DEMO_ATTENDANCE:
            c.execute(
                "INSERT INTO attendance (student_id,student_name,date,time) VALUES(?,?,?,?)",
                (sid, name, d, t)
            )

    conn.commit()
    conn.close()
    print("✅ Database ready.")


# ─────────────────────────────────────────────────────────────────────
# LBPH FACE RECOGNIZER
# ─────────────────────────────────────────────────────────────────────
def extract_face_gray(img_bgr, size=(200, 200)):
    """Detect face in image, return resized grayscale ROI or None."""
    gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )
    if len(faces) == 0:
        return None
    x, y, w, h = faces[0]
    roi = gray[y:y+h, x:x+w]
    roi = cv2.resize(roi, size)
    roi = cv2.equalizeHist(roi)   # improve lighting normalisation
    return roi


def train_recognizer():
    """
    Train LBPH model from all registered student photos.
    Saves model to disk and loads into memory.
    """
    global recognizer, label_to_student

    conn     = get_db()
    students = conn.execute(
        "SELECT name, student_id, label, photo_path FROM students WHERE photo_path IS NOT NULL"
    ).fetchall()
    conn.close()

    faces, labels = [], []
    label_to_student = {}

    for s in students:
        photo = s["photo_path"]
        if not (photo and os.path.exists(photo)):
            continue
        img = cv2.imread(photo)
        if img is None:
            continue

        roi = extract_face_gray(img)

        if roi is None:
            # For demo avatars that have no real face —
            # use the whole image resized as fallback
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            roi  = cv2.resize(gray, (200, 200))
            roi  = cv2.equalizeHist(roi)

        # Add multiple augmented versions for better recognition
        for roi_variant in augment(roi):
            faces.append(roi_variant)
            labels.append(s["label"])

        label_to_student[s["label"]] = (s["name"], s["student_id"])
        print(f"  ✓ Trained: {s['name']}")

    if not faces:
        print("⚠️  No faces to train on.")
        return 0

    recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=2, neighbors=16, grid_x=8, grid_y=8
    )
    recognizer.train(faces, np.array(labels))
    recognizer.save(MODEL_PATH)
    print(f"✅ LBPH model trained on {len(set(labels))} students.")
    return len(set(labels))


def augment(roi):
    """Return small augmentations of a face ROI for better training."""
    variants = [roi]
    # Slight brightness variations
    for delta in [-20, 20]:
        bright = np.clip(roi.astype(np.int16) + delta, 0, 255).astype(np.uint8)
        variants.append(bright)
    # Horizontal flip
    variants.append(cv2.flip(roi, 1))
    return variants


def load_model():
    """Load LBPH model from disk into memory."""
    global recognizer, label_to_student

    if not os.path.exists(MODEL_PATH):
        print("ℹ️  No model found — training now...")
        train_recognizer()
        return

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL_PATH)

    conn     = get_db()
    students = conn.execute("SELECT name, student_id, label FROM students").fetchall()
    conn.close()
    label_to_student = {s["label"]: (s["name"], s["student_id"]) for s in students}
    print(f"✅ Model loaded — {len(label_to_student)} students.")


def mark_attendance(name: str, student_id: str) -> bool:
    now = datetime.now()
    if name in last_marked:
        if (now - last_marked[name]).total_seconds() < 60:
            return False
    last_marked[name] = now
    conn = get_db()
    conn.execute(
        "INSERT INTO attendance (student_id,student_name,date,time) VALUES(?,?,?,?)",
        (student_id, name,
         now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S"))
    )
    conn.commit()
    conn.close()
    return True


def generate_frames():
    """MJPEG stream with LBPH face recognition."""
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Confidence threshold — lower = stricter match
    # LBPH: lower confidence value = better match
    CONFIDENCE_THRESHOLD = 80

    frame_n = 0

    while recognition_active:
        ok, frame = cap.read()

        if not ok:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "No Camera Detected", (140, 220),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (80, 200, 80), 2)
            cv2.putText(frame, "Connect a webcam to start", (130, 260),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (140, 140, 140), 1)
        else:
            frame_n += 1
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = FACE_CASCADE.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=6, minSize=(80, 80)
            )

            for (x, y, w, h) in faces:
                label = "Unknown"
                color = (0, 0, 210)

                if recognizer is not None:
                    roi = gray[y:y+h, x:x+w]
                    roi = cv2.resize(roi, (200, 200))
                    roi = cv2.equalizeHist(roi)

                    pred_label, confidence = recognizer.predict(roi)

                    if confidence < CONFIDENCE_THRESHOLD and pred_label in label_to_student:
                        name, sid = label_to_student[pred_label]
                        newly     = mark_attendance(name, sid)
                        label     = f"{name} ({int(confidence)})"
                        color     = (0, 200, 0) if newly else (200, 200, 0)

                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.rectangle(frame, (x, y+h-32), (x+w, y+h), color, cv2.FILLED)
                cv2.putText(frame, label, (x+5, y+h-8),
                            cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 255, 255), 1)

        ts = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
        cv2.putText(frame, ts, (10, frame.shape[0]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
               + buf.tobytes() + b"\r\n")

    cap.release()


# ─────────────────────────────────────────────────────────────────────
# AUTH
# ─────────────────────────────────────────────────────────────────────
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("logged_in"):
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated


# ─────────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return redirect(url_for("dashboard" if session.get("logged_in") else "login"))


@app.route("/login", methods=["GET","POST"])
def login():
    error = None
    if request.method == "POST":
        if (request.form.get("username") == ADMIN_USER and
                request.form.get("password") == ADMIN_PASS):
            session["logged_in"] = True
            return redirect(url_for("dashboard"))
        error = "Invalid credentials. Try admin / admin123"
    return render_template("login.html", error=error)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/dashboard")
@login_required
def dashboard():
    conn  = get_db()
    today = date.today().strftime("%Y-%m-%d")
    stats = {
        "total_students":   conn.execute("SELECT COUNT(*) FROM students").fetchone()[0],
        "today_present":    conn.execute(
            "SELECT COUNT(DISTINCT student_id) FROM attendance WHERE date=?", (today,)
        ).fetchone()[0],
        "total_records":    conn.execute("SELECT COUNT(*) FROM attendance").fetchone()[0],
        "registered_faces": conn.execute(
            "SELECT COUNT(*) FROM students WHERE photo_path IS NOT NULL"
        ).fetchone()[0],
    }
    recent = conn.execute(
        "SELECT * FROM attendance ORDER BY date DESC, time DESC LIMIT 10"
    ).fetchall()
    conn.close()
    return render_template("dashboard.html", stats=stats, recent=recent, today=today)


@app.route("/live")
@login_required
def live():
    return render_template("live.html", recognition_active=recognition_active)


@app.route("/video_feed")
@login_required
def video_feed():
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/toggle_recognition", methods=["POST"])
@login_required
def toggle_recognition():
    global recognition_active
    recognition_active = not recognition_active
    if recognition_active:
        load_model()
        return jsonify({"active": True, "students": len(label_to_student)})
    return jsonify({"active": False})


@app.route("/students")
@login_required
def students():
    conn = get_db()
    rows = conn.execute("SELECT * FROM students ORDER BY registered_at DESC").fetchall()
    conn.close()
    return render_template("students.html", students=rows)


@app.route("/register", methods=["GET","POST"])
@login_required
def register():
    if request.method == "POST":
        name       = request.form.get("name","").strip()
        student_id = request.form.get("student_id","").strip()
        email      = request.form.get("email","").strip()
        department = request.form.get("department","").strip()
        image_data = request.form.get("image_data","")

        if not all([name, student_id, image_data]):
            return jsonify({"success": False,
                            "message": "Name, Student ID and photo are required."})
        try:
            _, encoded = image_data.split(",", 1)
            img_bytes  = base64.b64decode(encoded)
            nparr      = np.frombuffer(img_bytes, np.uint8)
            img        = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            roi = extract_face_gray(img)
            if roi is None:
                return jsonify({"success": False,
                                "message": "No face detected. Ensure good lighting and face the camera directly."})

            # Get next label
            conn  = get_db()
            max_l = conn.execute("SELECT MAX(label) FROM students").fetchone()[0]
            label = (max_l + 1) if max_l is not None else 0

            photo_path = os.path.join(KNOWN_FACES_DIR, f"{student_id}.jpg")
            cv2.imwrite(photo_path, img)

            conn.execute(
                """INSERT OR REPLACE INTO students
                   (name,student_id,email,department,label,registered_at,photo_path)
                   VALUES(?,?,?,?,?,?,?)""",
                (name, student_id, email, department, label,
                 datetime.now().strftime("%Y-%m-%d %H:%M:%S"), photo_path)
            )
            conn.commit()
            conn.close()

            # Retrain model with new student
            train_recognizer()
            load_model()
            return jsonify({"success": True, "message": f"✅ {name} registered and model updated!"})

        except sqlite3.IntegrityError:
            return jsonify({"success": False,
                            "message": f"Student ID {student_id} already exists."})
        except Exception as e:
            return jsonify({"success": False, "message": f"Error: {e}"})

    return render_template("register.html")


@app.route("/attendance")
@login_required
def attendance():
    conn        = get_db()
    filter_date = request.args.get("date", date.today().strftime("%Y-%m-%d"))
    filter_dept = request.args.get("department","")

    query  = """SELECT a.*, s.department
                FROM attendance a
                LEFT JOIN students s ON a.student_id = s.student_id
                WHERE a.date = ?"""
    params = [filter_date]
    if filter_dept:
        query  += " AND s.department = ?"
        params.append(filter_dept)
    query += " ORDER BY a.time DESC"

    records = conn.execute(query, params).fetchall()
    depts   = conn.execute("SELECT DISTINCT department FROM students").fetchall()
    conn.close()
    return render_template("attendance.html", records=records,
                           filter_date=filter_date, filter_dept=filter_dept,
                           departments=depts)


@app.route("/api/attendance_chart")
@login_required
def attendance_chart():
    conn = get_db()
    rows = conn.execute(
        """SELECT date, COUNT(DISTINCT student_id) as cnt
           FROM attendance GROUP BY date
           ORDER BY date DESC LIMIT 14"""
    ).fetchall()
    conn.close()
    return jsonify({"dates":  [r["date"] for r in reversed(rows)],
                    "counts": [r["cnt"]  for r in reversed(rows)]})


@app.route("/api/delete_student/<sid>", methods=["DELETE"])
@login_required
def delete_student(sid):
    conn = get_db()
    conn.execute("DELETE FROM students   WHERE student_id=?", (sid,))
    conn.execute("DELETE FROM attendance WHERE student_id=?", (sid,))
    conn.commit()
    conn.close()
    train_recognizer()
    load_model()
    return jsonify({"success": True})


@app.route("/api/retrain", methods=["POST"])
@login_required
def retrain():
    count = train_recognizer()
    load_model()
    return jsonify({"success": True, "trained": count})


# ─────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    init_db()
    print("\n🧠 Training face recognition model ...")
    train_recognizer()
    load_model()

    print("\n" + "="*50)
    print("  👁️  AttendVision AI is running!")
    print("  URL  : http://127.0.0.1:5000")
    print("  Login: admin  /  admin123")
    print("="*50 + "\n")

    port = int(os.environ.get("PORT", 5000))
app.run(debug=False, host="0.0.0.0", port=port, threaded=True)
