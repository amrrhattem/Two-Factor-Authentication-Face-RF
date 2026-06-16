import json
import matplotlib.pyplot as plt
import numpy as np

with open('test_results.json', 'r') as f:
    results = json.load(f)

TP = results['true_accepted']
FN = results['false_rejected']
FP = results['false_accepted']
TN = results['true_rejected']

cm = np.array([[TP, FN], [FP, TN]])

print("="*50)
print("FACE RECOGNITION CONFUSION MATRIX")
print("="*50)
print("\n                 Predicted")
print("                 PASS    FAIL")
print(f"Actual Genuine    {TP:3d}     {FN:3d}")
print(f"       Impostor    {FP:3d}     {TN:3d}")

accuracy = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
far = FP / (FP + TN) if (FP + TN) > 0 else 0
frr = FN / (FN + TP) if (FN + TP) > 0 else 0

print(f"\nAccuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"FAR:       {far:.4f} ({far*100:.2f}%)")
print(f"FRR:       {frr:.4f} ({frr*100:.2f}%)")

# Plot
fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
ax.figure.colorbar(im, ax=ax)
ax.set(xticks=np.arange(2), yticks=np.arange(2),
       xticklabels=['PASS', 'FAIL'],
       yticklabels=['Genuine', 'Impostor'],
       title='Confusion Matrix - Face Recognition',
       ylabel='Actual', xlabel='Predicted')

for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                color="white" if cm[i, j] > cm.max()/2 else "black")

plt.tight_layout()
plt.savefig('face_confusion_matrix.png', dpi=150)
plt.show()

print("\n✅ Saved to face_confusion_matrix.png")