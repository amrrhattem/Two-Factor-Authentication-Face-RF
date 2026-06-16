import cv2
import numpy as np
import pickle
import sqlite3
import time
import json
from insightface.app import FaceAnalysis
from datetime import datetime

print("Loading face model...")
face_app = FaceAnalysis(name='buffalo_l')
face_app.prepare(ctx_id=0, det_size=(640, 640))

def get_face_embedding(frame):
    faces = face_app.get(frame)
    return faces[0].embedding if faces else None

def verify_face(face_emb, stored_emb, threshold=0.60):
    norm = np.linalg.norm(face_emb) * np.linalg.norm(stored_emb)
    if norm == 0:
        return 0.0, False
    similarity = float(np.dot(face_emb, stored_emb) / norm)
    return similarity, similarity >= threshold

# Connect to database
conn = sqlite3.connect('auth_system.db')
c = conn.cursor()
c.execute("SELECT name, face_embedding, device_id FROM users WHERE name = 'Amr'")
user = c.fetchone()
conn.close()

if user is None:
    print("Error: User 'Amr' not found. Run enroll.py first.")
    exit()

stored_emb = np.frombuffer(user[1], dtype=np.float32)
print(f"Testing with user: {user[0]}")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("Camera error")
    exit()

# Wait for camera to warm up
time.sleep(1)

results = []

print("\n=== Part 1: Genuine Attempts (Amr looking at camera) ===")
input("Press Enter when ready for 10 genuine attempts...")

for i in range(10):
    ret, frame = cap.read()
    if not ret:
        continue
    
    face_emb = get_face_embedding(frame)
    if face_emb is not None:
        similarity, passed = verify_face(face_emb, stored_emb)
        results.append({
            'type': 'genuine',
            'attempt': i+1,
            'similarity': similarity,
            'passed': passed
        })
        print(f"  Attempt {i+1}: similarity={similarity:.4f}, {'PASSED' if passed else 'FAILED'}")
    else:
        print(f"  Attempt {i+1}: No face detected")
    
    cv2.imshow('Test', frame)
    cv2.waitKey(500)
    time.sleep(0.5)

print("\n=== Part 2: Impostor Attempts (Amr with different conditions) ===")
input("Press Enter when ready for 10 impostor attempts...")
print("Tips: Turn away slightly, change lighting, use hand to partially cover face")

for i in range(10):
    ret, frame = cap.read()
    if not ret:
        continue
    
    face_emb = get_face_embedding(frame)
    if face_emb is not None:
        similarity, passed = verify_face(face_emb, stored_emb)
        results.append({
            'type': 'impostor',
            'attempt': i+1,
            'similarity': similarity,
            'passed': passed
        })
        print(f"  Attempt {i+1}: similarity={similarity:.4f}, {'PASSED' if passed else 'FAILED'}")
    else:
        print(f"  Attempt {i+1}: No face detected")
    
    cv2.imshow('Test', frame)
    cv2.waitKey(500)
    time.sleep(0.5)

cap.release()
cv2.destroyAllWindows()

print("\n" + "="*50)
print("RESULTS SUMMARY")
print("="*50)

genuine_sims = [r['similarity'] for r in results if r['type'] == 'genuine']
impostor_sims = [r['similarity'] for r in results if r['type'] == 'impostor']

print(f"\nGenuine attempts (n={len(genuine_sims)}):")
print(f"  Min similarity: {min(genuine_sims):.4f}")
print(f"  Max similarity: {max(genuine_sims):.4f}")
print(f"  Avg similarity: {np.mean(genuine_sims):.4f}")
print(f"  Std deviation: {np.std(genuine_sims):.4f}")

print(f"\nImpostor attempts (n={len(impostor_sims)}):")
print(f"  Min similarity: {min(impostor_sims):.4f}")
print(f"  Max similarity: {max(impostor_sims):.4f}")
print(f"  Avg similarity: {np.mean(impostor_sims):.4f}")
print(f"  Std deviation: {np.std(impostor_sims):.4f}")

print(f"\nPerformance metrics (at threshold = 0.60):")
true_accepted = sum(1 for r in results if r['type'] == 'genuine' and r['passed'])
false_accepted = sum(1 for r in results if r['type'] == 'impostor' and r['passed'])
true_rejected = sum(1 for r in results if r['type'] == 'impostor' and not r['passed'])
false_rejected = sum(1 for r in results if r['type'] == 'genuine' and not r['passed'])

print(f"  True Accepts: {true_accepted}/{len(genuine_sims)}")
print(f"  False Rejects: {false_rejected}/{len(genuine_sims)}")
print(f"  True Rejects: {true_rejected}/{len(impostor_sims)}")
print(f"  False Accepts: {false_accepted}/{len(impostor_sims)}")

far = false_accepted / len(impostor_sims) if len(impostor_sims) > 0 else 0
frr = false_rejected / len(genuine_sims) if len(genuine_sims) > 0 else 0
accuracy = (true_accepted + true_rejected) / (len(genuine_sims) + len(impostor_sims))

print(f"\n  FAR (False Acceptance Rate): {far:.4f} ({far*100:.2f}%)")
print(f"  FRR (False Rejection Rate): {frr:.4f} ({frr*100:.2f}%)")
print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# Save results to JSON
output = {
    'timestamp': datetime.now().isoformat(),
    'genuine_similarities': genuine_sims,
    'impostor_similarities': impostor_sims,
    'threshold': 0.60,
    'true_accepted': true_accepted,
    'false_accepted': false_accepted,
    'true_rejected': true_rejected,
    'false_rejected': false_rejected,
    'far': far,
    'frr': frr,
    'accuracy': accuracy
}

with open('test_results.json', 'w') as f:
    json.dump(output, f, indent=2)

print("\n✅ Results saved to test_results.json")