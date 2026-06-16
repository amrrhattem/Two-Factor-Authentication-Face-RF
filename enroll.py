import cv2
import numpy as np
import sqlite3
import time
from insightface.app import FaceAnalysis

from config import DB_PATH, FACE_MODEL_NAME, FACE_DET_SIZE
from setup_database import setup_database

# Ensure DB and schema exist
setup_database()

face_app = FaceAnalysis(name=FACE_MODEL_NAME)
face_app.prepare(ctx_id=0, det_size=FACE_DET_SIZE)

cap = cv2.VideoCapture(0)


def get_face_embedding(frame):
    faces = face_app.get(frame)
    return faces[0].embedding if faces else None


print("=== Face Enrollment ===")
user_name = input("Enter user name: ").strip()
device_id = input(f"Enter device MAC address for {user_name}: ").strip()

embeddings = []
print("Look at camera. Capturing 10 images...")

for i in range(10):
    ret, frame = cap.read()
    if not ret:
        print("Camera error")
        break

    embedding = get_face_embedding(frame)
    label_text = f"Captured {i+1}/10" if embedding is not None else "No face detected"
    label_color = (0, 255, 0) if embedding is not None else (0, 0, 255)

    if embedding is not None:
        embeddings.append(embedding)
        print(f"  Image {i+1}/10 captured")
    else:
        print(f"  Image {i+1}/10 failed - no face detected")

    cv2.putText(frame, label_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, label_color, 2)
    cv2.imshow('Enrollment', frame)
    cv2.waitKey(500)
    time.sleep(0.5)

cap.release()
cv2.destroyAllWindows()

if len(embeddings) >= 5:
    avg_embedding = np.mean(embeddings, axis=0).astype(np.float32)
    # Store as portable raw bytes instead of pickle
    embedding_bytes = avg_embedding.tobytes()

    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(
            "INSERT INTO users (name, face_embedding, device_id) VALUES (?, ?, ?)",
            (user_name, sqlite3.Binary(embedding_bytes), device_id)
        )
        conn.commit()
        print(f"\n✅ {user_name} enrolled successfully!")
        print(f"   Face embedding shape: {avg_embedding.shape}")
        print(f"   Device: {device_id}")
    except sqlite3.IntegrityError:
        print(f"\n⚠️  User '{user_name}' already exists. Updating face embedding...")
        conn.execute(
            "UPDATE users SET face_embedding = ?, device_id = ? WHERE name = ?",
            (sqlite3.Binary(embedding_bytes), device_id, user_name)
        )
        conn.commit()
        print(f"✅ {user_name} updated successfully!")
    finally:
        conn.close()
else:
    print(f"\n❌ Enrollment failed. Only {len(embeddings)} valid faces captured (need at least 5).")