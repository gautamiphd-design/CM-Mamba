import numpy as np
import matplotlib.pyplot as plt
import pickle

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.preprocessing import label_binarize

from dataset import load_datasets
from model import create_model

# =====================
# Paths
# =====================
train_path = "DATASET_PATH/train"
val_path   = "DATASET_PATH/val"

# =====================
# Load datasets
# =====================
train_ds, val_ds, class_names = load_datasets(train_path, val_path)

# =====================
# Load model
# =====================
model = create_model(input_shape=(224,224,3), num_classes=len(class_names))
model.load_weights("best_medmamba_weights.h5")

# =====================
# Load training history
# =====================
with open("history.pkl", "rb") as f:
    history = pickle.load(f)

# =====================
# Function to get predictions
# =====================
def get_predictions(dataset):
    y_true, y_pred, y_prob = [], [], []

    for images, labels in dataset:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(preds, axis=1))
        y_prob.extend(preds)

    return np.array(y_true), np.array(y_pred), np.array(y_prob)

# =====================
# Get train & val predictions
# =====================
y_train_true, y_train_pred, y_train_prob = get_predictions(train_ds)
y_val_true, y_val_pred, y_val_prob       = get_predictions(val_ds)

# =====================
# Binarize labels
# =====================
y_train_bin = label_binarize(y_train_true, classes=range(len(class_names)))
y_val_bin   = label_binarize(y_val_true, classes=range(len(class_names)))

# =====================
# Create figure
# =====================
fig = plt.figure(figsize=(20, 14))

# ---------------------------------
# (a) Training Confusion Matrix
# ---------------------------------
ax1 = plt.subplot(3,2,1)
cm_train = confusion_matrix(y_train_true, y_train_pred)
ConfusionMatrixDisplay(cm_train, display_labels=class_names).plot(
    cmap='Blues', ax=ax1, colorbar=False
)
ax1.set_title("(a) Training Confusion Matrix")

# ---------------------------------
# (b) Validation Confusion Matrix
# ---------------------------------
ax2 = plt.subplot(3,2,2)
cm_val = confusion_matrix(y_val_true, y_val_pred)
ConfusionMatrixDisplay(cm_val, display_labels=class_names).plot(
    cmap='Blues', ax=ax2, colorbar=False
)
ax2.set_title("(b) Validation Confusion Matrix")

# ---------------------------------
# (c) Training ROC
# ---------------------------------
ax3 = plt.subplot(3,2,3)
for i in range(len(class_names)):
    fpr, tpr, _ = roc_curve(y_train_bin[:, i], y_train_prob[:, i])
    roc_auc = auc(fpr, tpr)
    ax3.plot(fpr, tpr, label=f"{class_names[i]} (AUC={roc_auc:.2f})")

ax3.plot([0,1],[0,1],'k--')
ax3.set_title("(c) Training ROC")
ax3.set_xlabel("FPR")
ax3.set_ylabel("TPR")
ax3.legend()

# ---------------------------------
# (d) Validation ROC
# ---------------------------------
ax4 = plt.subplot(3,2,4)
for i in range(len(class_names)):
    fpr, tpr, _ = roc_curve(y_val_bin[:, i], y_val_prob[:, i])
    roc_auc = auc(fpr, tpr)
    ax4.plot(fpr, tpr, label=f"{class_names[i]} (AUC={roc_auc:.2f})")

ax4.plot([0,1],[0,1],'k--')
ax4.set_title("(d) Validation ROC")
ax4.set_xlabel("FPR")
ax4.set_ylabel("TPR")
ax4.legend()

# ---------------------------------
# (e) Accuracy Curve
# ---------------------------------
ax5 = plt.subplot(3,2,5)
ax5.plot(history['accuracy'], label='Train Accuracy')
ax5.plot(history['val_accuracy'], label='Validation Accuracy')
ax5.set_title("(e) Accuracy Curve")
ax5.set_xlabel("Epochs")
ax5.set_ylabel("Accuracy")
ax5.legend()

# ---------------------------------
# (f) Loss Curve
# ---------------------------------
ax6 = plt.subplot(3,2,6)
ax6.plot(history['loss'], label='Train Loss')
ax6.plot(history['val_loss'], label='Validation Loss')
ax6.set_title("(f) Loss Curve")
ax6.set_xlabel("Epochs")
ax6.set_ylabel("Loss")
ax6.legend()

# =====================
# Final Layout
# =====================
plt.tight_layout()
plt.savefig("full_evaluation_figure.png", dpi=600, bbox_inches='tight')
plt.show()

print("✅ Full evaluation figure generated")
