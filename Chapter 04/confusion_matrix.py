import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import train_test_split
import os

print("Loading RF model...")
rf_model = joblib.load('rf_model_15features.pkl')
rf_label_encoder = joblib.load('label_encoder_15features.pkl')

print("Loading feature data...")
dataset_path = r"C:\Users\Amr Hatem\Desktop\ML - New RFFI\FE -  Dataset"
files = [f for f in os.listdir(dataset_path) if f.endswith('.csv')]

feature_cols = ['CFO', 'short_freq', 'long_freq', 'frac_dimension_1', 'frac_dimension_2',
                'iqi_1', 'iqi_2', 'mag_error_mean_1', 'mag_error_var_1', 
                'mag_error_mean_2', 'mag_error_var_2', 'phase_error_mean_1', 
                'phase_error_var_1', 'phase_error_mean_2', 'phase_error_var_2']

all_features = []
all_labels = []

print("Loading data (this may take a moment)...")
for file in files:
    try:
        df = pd.read_csv(os.path.join(dataset_path, file))
        mac = file.replace('_pre.csv', '').replace('.csv', '')
        available_cols = [col for col in feature_cols if col in df.columns]
        X = df[available_cols].values
        y = [mac] * len(X)
        all_features.append(X)
        all_labels.extend(y)
    except Exception as e:
        print(f"Error in {file}: {e}")

# Combine all data
X = np.vstack(all_features)
y = np.array(all_labels)

print(f"Loaded {X.shape[0]} samples, {X.shape[1]} features")
print(f"Unique devices: {len(np.unique(y))}")

# Encode labels
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split (use smaller sample if needed due to memory)
# Take 50,000 random samples for faster computation
if X.shape[0] > 50000:
    np.random.seed(42)
    indices = np.random.choice(X.shape[0], 50000, replace=False)
    X_sampled = X[indices]
    y_sampled = y_encoded[indices]
    print(f"Using sampled data: {X_sampled.shape[0]} samples")
else:
    X_sampled = X
    y_sampled = y_encoded

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_sampled, y_sampled, test_size=0.2, stratify=y_sampled, random_state=42
)

print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

# Predict
print("Predicting...")
y_pred = rf_model.predict(X_test)

# Get unique classes in test set
unique_classes = np.unique(y_test)
n_classes = len(unique_classes)
print(f"Number of classes in test set: {n_classes}")

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=unique_classes)

# Get class names
class_names = le.inverse_transform(unique_classes)

print("\n" + "="*60)
print("CONFUSION MATRIX SUMMARY")
print("="*60)

# Calculate per-class metrics
print("\nPer-class accuracy (first 10 classes):")
for i, name in enumerate(class_names[:10]):
    tp = cm[i, i]
    total = np.sum(cm[i, :])
    acc = tp / total if total > 0 else 0
    print(f"  {name}: {acc:.4f} ({tp}/{total})")

# Overall accuracy
overall_acc = np.trace(cm) / np.sum(cm)
print(f"\nOverall accuracy: {overall_acc:.4f} ({overall_acc*100:.2f}%)")

# Save confusion matrix as text
with open('confusion_matrix_results.txt', 'w') as f:
    f.write("="*60 + "\n")
    f.write("CONFUSION MATRIX RESULTS\n")
    f.write("="*60 + "\n\n")
    f.write(f"Total test samples: {np.sum(cm)}\n")
    f.write(f"Number of classes: {n_classes}\n")
    f.write(f"Overall accuracy: {overall_acc:.4f} ({overall_acc*100:.2f}%)\n\n")
    f.write("Per-class accuracy (first 20 classes):\n")
    for i, name in enumerate(class_names[:20]):
        tp = cm[i, i]
        total = np.sum(cm[i, :])
        acc = tp / total if total > 0 else 0
        f.write(f"  {name}: {acc:.4f} ({tp}/{total})\n")

# Plot confusion matrix (show first 15 classes for readability)
if n_classes > 15:
    cm_small = cm[:15, :15]
    labels_small = class_names[:15]
else:
    cm_small = cm
    labels_small = class_names

plt.figure(figsize=(12, 10))
disp = ConfusionMatrixDisplay(confusion_matrix=cm_small, display_labels=labels_small)
disp.plot(cmap='Blues', xticks_rotation=90, ax=plt.gca())
plt.title(f'Confusion Matrix - RF Fingerprinting (First {len(labels_small)} Classes)\nOverall Accuracy: {overall_acc*100:.2f}%')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n✅ Confusion matrix saved to confusion_matrix.png")
print("✅ Results saved to confusion_matrix_results.txt")