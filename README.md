# 👁️ AttendVision AI

> Real-Time AI-Powered Attendance System using OpenCV, Flask, and SQLite.

![Python](https://img.shields.io/badge/Python-3.11+-blue?style=flat-square&logo=python)
![Flask](https://img.shields.io/badge/Flask-3.0-green?style=flat-square&logo=flask)
![OpenCV](https://img.shields.io/badge/OpenCV-4.10-orange?style=flat-square&logo=opencv)
![SQLite](https://img.shields.io/badge/SQLite-3-lightblue?style=flat-square&logo=sqlite)

---

## ✨ Features

- 🧠 Real-time face detection via OpenCV Haar Cascade
- 📊 Admin dashboard with Chart.js attendance trend
- 👥 Student registration with webcam photo capture
- 📋 Attendance log with date & department filter
- 🗄️ SQLite database — 8 demo students + 24 history records
- 🔐 Session-based admin login

---

## ⚙️ Install & Run (VS Code)

```bash
# 1. Open terminal in VS Code (Ctrl + `)
python -m venv venv
venv\Scripts\activate

# 2. Install (takes ~30 seconds, no C++ needed)
pip install -r requirements.txt

# 3. Generate demo faces
python generate_faces.py

# 4. Seed database
python seed_database.py

# 5. Run
python app.py
```

Open **http://127.0.0.1:5000** — Login: `admin` / `admin123`

---

## 📤 Push to GitHub

```bash
git init
git add .
git commit -m "feat: AttendVision AI - real-time face detection attendance system"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/AttendVision-AI.git
git push -u origin main
```

---

## 🏗️ Tech Stack

| Layer | Tech |
|---|---|
| Backend | Python, Flask |
| Computer Vision | OpenCV (Haar Cascade + HSV histogram) |
| Database | SQLite3 |
| Frontend | Vanilla JS, Chart.js, dark CSS |

---

## 👨‍💻 Author

Talha — MSc CS Student @ TU Ilmenau
