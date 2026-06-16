import sqlite3
import numpy as np
from config import DB_PATH  # Use the same path as enrollment

conn = sqlite3.connect(DB_PATH)
c = conn.cursor()

c.execute("SELECT name, device_id, face_embedding FROM users")
users = c.fetchall()

print("=== Users in database ===")
for user in users:
    name = user[0]
    device = user[1]
    embedding = user[2]

    if embedding is None:
        emb_info = "None"
    else:
        try:
            emb = np.frombuffer(embedding, dtype=np.float32)  # Match tobytes() storage
            emb_info = f"{emb.shape[0]} floats, first value: {emb[0]:.4f}"
        except Exception as e:
            emb_info = f"Corrupted ({e})"

    print(f"  Name: '{name}', Device: '{device}', Embedding: {emb_info}")

conn.close()