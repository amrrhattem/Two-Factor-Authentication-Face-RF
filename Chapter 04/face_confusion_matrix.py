import json
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

# Load your test results
with open('test_results.json', 'r') as f:
    results = json.load(f)

# Create 2x2 confusion matrix
# Rows: Actual (Genuine, Impostor)
# Columns: Predicted (PASS, FAIL)

TP = results['true_accepted']  # Genuine → PASS
FN = results['false_rejected']  # Genuine → FAIL
FP = results['false_accepted']  # Impostor → PASS
TN = results['true_rejected']   # Impostor → FAIL

cm = np.array([[TP, FN], [FP, TN]])

print("="*50)
print("FACE RECOGNITION CONFUSION MATRIX")
print("="*50)
print("\n                 Predicted")
print("                 PASS    FAIL")
print(f"Actual Genuine    {TP:3d}     {FN:3d}")
print(f"       Impostor    {FP:3d}     {TN:3d}")
print("\n" + "="*50)

# Calculate metrics
accuracy = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
far = FP / (FP + TN) if (FP + TN) > 0 else 0
frr = FN / (FN + TP) if (FN + TP) > 0 else 0

print(f"\nAccuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")
print(f"FAR:       {far:.4f} ({far*100:.2f}%)")
print(f"FRR:       {frr:.4f} ({frr*100:.2f}%)")

# Plot confusion matrix
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
ax.figure.colorbar(im, ax=ax)

# Set labels
ax.set(xticks=np.arange(2), yticks=np.arange(2),
       xticklabels=['PASS', 'FAIL'],
       yticklabels=['Genuine', 'Impostor'],
       title='Confusion Matrix - Face Recognition',
       ylabel='Actual', xlabel='Predicted')

# Add text annotations
thresh = cm.max() / 2
for i in range(2):
    for j in range(2):
        ax.text(j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")

plt.tight_layout()
plt.savefig('face_confusion_matrix.png', dpi=150)
plt.show()

print("\n✅ Face confusion matrix saved to face_confusion_matrix.png")

# Also save to JSON for LaTeX
latex_data = {
    'cm': cm.tolist(),
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'f1': f1,
    'far': far,
    'frr': frr
}
with open('face_metrics.json', 'w') as f:
    json.dump(latex_data, f, indent=2)