# Two-Factor Authentication System

## Face Recognition + RF Fingerprinting

A two-phase multi-modal authentication system that verifies both the person (via facial recognition) and their device (via RF fingerprinting) before granting access.

---

## Features

- **Phase 1: Face Recognition** (YOLOv8 + ArcFace)
- **Phase 2: RF Fingerprinting** (Random Forest, 88.59% accuracy)
- **Device-User Binding** via SQLite
- **Sequential Fail-Fast Design** (400ms for impostors)
- **Full-Stack Deployment** (Flask + Dashboards)

---

## Project Structure

Final_Project/
├── config.py
├── setup_database.py
├── server.py
├── enroll.py
├── extract_features.py
├── user_dashboard.html
├── admin_dashboard.html
├── requirements.txt
└── README.md

---

## Installation

```bash
pip install -r requirements.txt
python setup_database.py
python enroll.py
python server.py
```
