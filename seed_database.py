"""
seed_database.py — AttendVision AI (LBPH OpenCV version)
Run once: python seed_database.py
"""

import sqlite3, csv, os, json
from datetime import datetime
import cv2
import numpy as np

DB_PATH         = os.path.join("database", "attendance.db")
KNOWN_FACES_DIR = "known_faces"
MODEL_PATH      = os.path.join("database", "lbph_model.yml")
STUDENTS_CSV    = os.path.join("dataset", "students.csv")
ATTENDANCE_CSV  = os.path.join("dataset", "attendance_history.csv")

FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

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
    ("STU001","Ahmed Khan","2025-10-20","08:32:15"),
    ("STU002","Sara Malik","2025-10-20","08:35:02"),
    ("STU003","Bilal Hassan","2025-10-20","08:41:09"),
    ("STU001","Ahmed Khan","2025-10-21","08:29:05"),
    ("STU002","Sara Malik","2025-10-21","08:33:18"),
    ("STU006","Zara Ahmed","2025-10-21","08:45:55"),
    ("STU001","Ahmed Khan","2025-10-22","08:31:40"),
    ("STU003","Bilal Hassan","2025-10-22","08:38:17"),
    ("STU004","Ayesha Siddiqui","2025-10-22","08:44:03"),
    ("STU002","Sara Malik","2025-10-23","08:28:11"),
    ("STU005","Omar Farooq","2025-10-23","08:36:44"),
    ("STU006","Zara Ahmed","2025-10-23","08:47:38"),
    ("STU007","Usman Ali","2025-10-23","09:02:15"),
    ("STU008","Fatima Noor","2025-10-23","09:10:07"),
]


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def create_tables(conn):
    conn.execute("""CREATE TABLE IF NOT EXISTS students (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        student_id TEXT UNIQUE NOT NULL,
        email TEXT, department TEXT, label INTEGER,
        registered_at TEXT, photo_path TEXT
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS attendance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id TEXT NOT NULL, student_name TEXT NOT NULL,
        date TEXT NOT NULL, time TEXT NOT NULL,
        status TEXT DEFAULT 'Present',
        method TEXT DEFAULT 'Face Recognition'
    )""")
    conn.commit()
    print("✅ Tables ready.")


def seed_students(conn):
    conn.execute("DELETE FROM students")
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for i, (name, sid, email, dept) in enumerate(DEMO_STUDENTS):
        photo = os.path.join(KNOWN_FACES_DIR, f"{sid}.jpg")
        conn.execute(
            """INSERT OR IGNORE INTO students
               (name,student_id,email,department,label,registered_at,photo_path)
               VALUES(?,?,?,?,?,?,?)""",
            (name, sid, email, dept, i, now,
             photo if os.path.exists(photo) else None)
        )
    conn.commit()
    print(f"✅ Seeded {len(DEMO_STUDENTS)} students.")


def seed_attendance(conn):
    conn.execute("DELETE FROM attendance")
    for sid, name, d, t in DEMO_ATTENDANCE:
        conn.execute(
            "INSERT INTO attendance (student_id,student_name,date,time) VALUES(?,?,?,?)",
            (sid, name, d, t)
        )
    conn.commit()
    print(f"✅ Seeded {len(DEMO_ATTENDANCE)} attendance records.")


def train_lbph(conn):
    students = conn.execute(
        "SELECT name,student_id,label,photo_path FROM students WHERE photo_path IS NOT NULL"
    ).fetchall()

    faces, labels = [], []

    for s in students:
        photo = s["photo_path"]
        if not (photo and os.path.exists(photo)):
            continue
        img  = cv2.imread(photo)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dets = FACE_CASCADE.detectMultiScale(gray, 1.1, 5, minSize=(60,60))

        if len(dets):
            x,y,w,h = dets[0]
            roi = gray[y:y+h, x:x+w]
        else:
            roi = gray   # fallback for demo avatars

        roi = cv2.resize(roi, (200,200))
        roi = cv2.equalizeHist(roi)

        # augment
        for v in [roi,
                  np.clip(roi.astype(np.int16)+25,0,255).astype(np.uint8),
                  np.clip(roi.astype(np.int16)-25,0,255).astype(np.uint8),
                  cv2.flip(roi,1)]:
            faces.append(v)
            labels.append(s["label"])

        print(f"  ✓ {s['name']}")

    if not faces:
        print("⚠️  No photos found to train on.")
        return

    rec = cv2.face.LBPHFaceRecognizer_create(radius=2, neighbors=16, grid_x=8, grid_y=8)
    rec.train(faces, np.array(labels))
    rec.save(MODEL_PATH)
    print(f"✅ LBPH model trained and saved → {MODEL_PATH}")


if __name__ == "__main__":
    print("\n🌱 AttendVision AI — Seeder")
    print("="*40)
    os.makedirs("database",      exist_ok=True)
    os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

    conn = get_db()
    create_tables(conn)
    seed_students(conn)
    seed_attendance(conn)

    print("\n🧠 Training LBPH model ...")
    train_lbph(conn)
    conn.close()

    print("\n✅ Done! Run:  python app.py")
    print("   Login:      admin / admin123\n")
