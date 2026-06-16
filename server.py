import cv2
import numpy as np
import pickle
import sqlite3
from functools import wraps

from insightface.app import FaceAnalysis
from flask import Flask, request, jsonify, g
from flask_cors import CORS
import base64
import joblib
import os
from datetime import datetime

from config import (
    DB_PATH, RF_MODEL_PATH, LABEL_ENCODER_PATH, ALL_FEATURES_PATH,
    ADMIN_TOKEN, FACE_SIMILARITY_THRESHOLD, FACE_MODEL_NAME,
    FACE_DET_SIZE, DEBUG, PORT
)
from setup_database import setup_database

app = Flask(__name__)
CORS(app)

# --- Startup: ensure DB exists ---
setup_database()

# --- Load models once at startup ---
print("Loading face model...")
face_app = FaceAnalysis(name=FACE_MODEL_NAME)
face_app.prepare(ctx_id=0, det_size=FACE_DET_SIZE)

print("Loading RF model...")
rf_model = joblib.load(RF_MODEL_PATH)
rf_label_encoder = joblib.load(LABEL_ENCODER_PATH)

print("Loading device features...")
with open(ALL_FEATURES_PATH, 'rb') as f:
    device_features_db = pickle.load(f)

print(f"✅ Server ready. {len(device_features_db)} RF devices loaded.")


# ============================================================
# Database helpers (connection-per-request via Flask g)
# ============================================================

def get_db():
    if 'db' not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(error):
    db = g.pop('db', None)
    if db is not None:
        db.close()


def log_attempt(user_name, face_score, device_id, final_decision):
    db = get_db()
    db.execute(
        "INSERT INTO auth_log (user_name, face_score, device_id, final_decision) VALUES (?, ?, ?, ?)",
        (user_name, face_score, device_id, final_decision)
    )
    db.commit()
    print(f"📝 Logged: {user_name} - {final_decision}")


# ============================================================
# Auth guard for admin endpoints
# ============================================================

def require_admin(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('X-Admin-Token', '')
        if token != ADMIN_TOKEN:
            return jsonify({'error': 'Unauthorized'}), 401
        return f(*args, **kwargs)
    return decorated


# ============================================================
# Face helpers
# ============================================================

def get_face_embedding(frame):
    faces = face_app.get(frame)
    if len(faces) > 0:
        return faces[0].embedding
    return None


def verify_face(face_embedding, stored_embedding, threshold=FACE_SIMILARITY_THRESHOLD):
    norm = np.linalg.norm(face_embedding) * np.linalg.norm(stored_embedding)
    if norm == 0:
        return 0.0, False
    similarity = float(np.dot(face_embedding, stored_embedding) / norm)
    return similarity, similarity >= threshold


def decode_image_from_base64(face_base64):
    if ',' in face_base64:
        face_base64 = face_base64.split(',')[1]
    face_bytes = base64.b64decode(face_base64)
    nparr = np.frombuffer(face_bytes, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


# ============================================================
# Public endpoint: authenticate
# ============================================================

@app.route('/authenticate', methods=['POST'])
def authenticate():
    try:
        data = request.json
        face_base64 = data.get('face_image')
        claimed_user = data.get('claimed_user')

        if claimed_user == 'Unknown':
            log_attempt('Unknown', None, None, 'DENIED')
            return jsonify({
                'final_verdict': 'DENIED',
                'phase1': {'status': 'FAILED', 'similarity': 0},
                'phase2': {'status': 'SKIPPED'},
                'message': 'Unknown user - Access denied'
            })

        if not face_base64:
            return jsonify({'error': 'No face image'}), 400

        frame = decode_image_from_base64(face_base64)
        if frame is None:
            return jsonify({'error': 'Failed to decode image'}), 400

        face_emb = get_face_embedding(frame)
        if face_emb is None:
            log_attempt(claimed_user, None, None, 'DENIED')
            return jsonify({'error': 'No face detected'}), 400

        db = get_db()
        user = db.execute(
            "SELECT face_embedding, device_id FROM users WHERE name = ?",
            (claimed_user,)
        ).fetchone()

        if user is None:
            log_attempt(claimed_user, None, None, 'DENIED')
            return jsonify({'error': f'User {claimed_user} not found'}), 404

        stored_emb = np.frombuffer(user['face_embedding'], dtype=np.float32)
        face_similarity, face_match = verify_face(face_emb, stored_emb)

        if not face_match:
            log_attempt(claimed_user, face_similarity, None, 'DENIED')
            return jsonify({
                'final_verdict': 'DENIED',
                'phase1': {'status': 'FAILED', 'similarity': face_similarity},
                'phase2': {'status': 'SKIPPED'},
                'message': f'Face verification failed (similarity: {face_similarity:.3f})'
            })

        # --- Phase 2: RF device fingerprinting ---
        expected_device = user['device_id']

        if expected_device not in device_features_db:
            log_attempt(claimed_user, face_similarity, expected_device, 'DENIED')
            return jsonify({
                'final_verdict': 'DENIED',
                'phase1': {'status': 'PASSED', 'similarity': face_similarity},
                'phase2': {'status': 'FAILED'},
                'message': f'Device {expected_device} not found in RF database'
            })

        rf_features = device_features_db[expected_device].reshape(1, -1)
        predicted_device_idx = rf_model.predict(rf_features)[0]
        predicted_device = rf_label_encoder.inverse_transform([predicted_device_idx])[0]

        print(f"Expected: {expected_device}, Predicted: {predicted_device}")

        if predicted_device != expected_device:
            log_attempt(claimed_user, face_similarity, expected_device, 'DENIED')
            return jsonify({
                'final_verdict': 'DENIED',
                'phase1': {'status': 'PASSED', 'similarity': face_similarity},
                'phase2': {'status': 'FAILED', 'expected': expected_device, 'predicted': predicted_device},
                'message': 'Device mismatch'
            })

        log_attempt(claimed_user, face_similarity, expected_device, 'GRANTED')
        return jsonify({
            'final_verdict': 'GRANTED',
            'phase1': {'status': 'PASSED', 'similarity': face_similarity},
            'phase2': {'status': 'PASSED', 'device_id': expected_device},
            'message': 'Access granted'
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ============================================================
# Public endpoint: list users (names only, for dropdown)
# ============================================================

@app.route('/users', methods=['GET'])
def get_users():
    db = get_db()
    users = db.execute("SELECT name, device_id FROM users").fetchall()
    return jsonify([{'name': u['name'], 'device_id': u['device_id']} for u in users])


# ============================================================
# Face enrollment endpoints (protected)
# ============================================================

@app.route('/enroll_face_temp', methods=['POST'])
@require_admin
def enroll_face_temp():
    try:
        data = request.json
        face_base64 = data.get('face_image')

        frame = decode_image_from_base64(face_base64)
        if frame is None:
            return jsonify({'error': 'Failed to decode image'}), 400

        face_emb = get_face_embedding(frame)
        if face_emb is None:
            return jsonify({'error': 'No face detected'}), 400

        return jsonify({'embedding': face_emb.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/save_face_embedding', methods=['POST'])
@require_admin
def save_face_embedding():
    try:
        data = request.json
        username = data.get('username')
        embedding = data.get('embedding')

        embedding_array = np.array(embedding, dtype=np.float32)
        # Store as raw float32 bytes — portable across numpy versions
        embedding_bytes = sqlite3.Binary(embedding_array.tobytes())

        db = get_db()
        db.execute(
            "UPDATE users SET face_embedding = ? WHERE name = ?",
            (embedding_bytes, username)
        )
        db.commit()
        return jsonify({'status': 'ok'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================
# Admin endpoints (all protected by require_admin)
# ============================================================

@app.route('/admin/stats', methods=['GET'])
@require_admin
def admin_stats():
    db = get_db()
    total_users = db.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    total_attempts = db.execute("SELECT COUNT(*) FROM auth_log").fetchone()[0]
    success_count = db.execute(
        "SELECT COUNT(*) FROM auth_log WHERE final_decision = 'GRANTED'"
    ).fetchone()[0]
    success_rate = (success_count / total_attempts * 100) if total_attempts > 0 else 0

    return jsonify({
        'total_users': total_users,
        'total_attempts': total_attempts,
        'success_rate': round(success_rate, 1),
        'total_rf_devices': len(device_features_db)
    })


@app.route('/admin/users', methods=['GET'])
@require_admin
def admin_users():
    db = get_db()
    users = db.execute(
        "SELECT user_id, name, device_id, enrollment_date, face_embedding FROM users"
    ).fetchall()

    result = []
    for u in users:
        has_face = 'no'
        if u['face_embedding'] is not None:
            try:
                emb = np.frombuffer(u['face_embedding'], dtype=np.float32)
                has_face = 'yes' if len(emb) > 0 else 'no'
            except Exception:
                has_face = 'no'
        result.append({
            'user_id': u['user_id'],
            'name': u['name'],
            'device_id': u['device_id'],
            'enrolled_date': u['enrollment_date'],
            'has_face': has_face
        })
    return jsonify(result)


@app.route('/admin/logs', methods=['GET'])
@require_admin
def admin_logs():
    db = get_db()
    logs = db.execute(
        "SELECT timestamp, user_name, face_score, device_id, final_decision "
        "FROM auth_log ORDER BY timestamp DESC LIMIT 100"
    ).fetchall()
    return jsonify([{
        'timestamp': l['timestamp'],
        'user_name': l['user_name'],
        'face_score': l['face_score'],
        'device_id': l['device_id'],
        'final_decision': l['final_decision']
    } for l in logs])


@app.route('/admin/rf_devices', methods=['GET'])
@require_admin
def admin_rf_devices():
    return jsonify([
        {'mac': mac, 'features_count': len(features)}
        for mac, features in device_features_db.items()
    ])


@app.route('/admin/delete_user/<int:user_id>', methods=['DELETE'])
@require_admin
def admin_delete_user(user_id):
    db = get_db()
    user = db.execute("SELECT name FROM users WHERE user_id = ?", (user_id,)).fetchone()
    if user:
        db.execute("DELETE FROM auth_log WHERE user_name = ?", (user['name'],))
        db.execute("DELETE FROM users WHERE user_id = ?", (user_id,))
        db.commit()
        return jsonify({'status': 'ok'})
    return jsonify({'error': 'User not found'}), 404


@app.route('/admin/clear_logs', methods=['DELETE'])
@require_admin
def admin_clear_logs():
    db = get_db()
    db.execute("DELETE FROM auth_log")
    db.commit()
    return jsonify({'status': 'ok'})


@app.route('/admin/add_user', methods=['POST'])
@require_admin
def admin_add_user():
    data = request.json
    name = data.get('name', '').strip()
    device_id = data.get('device_id', '').strip()

    if not name or not device_id:
        return jsonify({'error': 'name and device_id are required'}), 400

    db = get_db()
    try:
        # Placeholder embedding: zeros as raw float32 bytes
        placeholder = sqlite3.Binary(np.zeros(512, dtype=np.float32).tobytes())
        db.execute(
            "INSERT INTO users (name, device_id, face_embedding) VALUES (?, ?, ?)",
            (name, device_id, placeholder)
        )
        db.commit()
        return jsonify({'status': 'ok'})
    except sqlite3.IntegrityError:
        return jsonify({'error': f'User "{name}" already exists'}), 409
    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT, debug=DEBUG)