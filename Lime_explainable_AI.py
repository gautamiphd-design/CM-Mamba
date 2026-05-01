import os
import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory
import numpy as np
import matplotlib.pyplot as plt
import cv2

# -----------------------------
# 1) GET CLASS NAMES SAFELY
# -----------------------------
temp_dataset_for_class_names = image_dataset_from_directory(
    train_path,
    image_size=img_size,
    batch_size=1,
    label_mode='int',
    shuffle=False,
    interpolation='nearest'
)
class_names = temp_dataset_for_class_names.class_names
del temp_dataset_for_class_names

print("Class order from dataset:", class_names)

# -----------------------------
# 2) TRUE LABEL FROM FILE PATH
# -----------------------------
true_class_name = os.path.basename(os.path.dirname(test_image_path))
print("True class (from folder):", true_class_name)

# -----------------------------
# 3) MODEL PREDICTION
# -----------------------------
preds = model.predict(img_batch)[0]   # shape (num_classes,)
pred_class_index = np.argmax(preds)
pred_class_name = class_names[pred_class_index]

print("\nMODEL PREDICTION")
print("====================")
print("Predicted Class:", pred_class_name)
for i, p in enumerate(preds):
    print(f"{class_names[i]} : {p*100:.2f}%")

# -----------------------------
# 4) LIME → SMOOTH HEATMAP UTILS
# -----------------------------
def smooth_mask(mask):
    mask = mask.astype("float32")
    mask = cv2.GaussianBlur(mask, (25,25), 0)
    if mask.max() > 0:
        mask = mask / mask.max()
    return mask

def apply_heatmap(img, mask, cmap="jet", alpha=0.55):
    """Overlay heatmap (e.g. jet) over the original image."""
    heatmap = plt.get_cmap(cmap)(mask)[..., :3]         # (H,W,3), drop alpha
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    blended = (1 - alpha) * img + alpha * heatmap
    return blended

# make sure img is float in [0,1]
img_norm = img.astype("float32")
if img_norm.max() > 1.0:
    img_norm = img_norm / 255.0

# -----------------------------
# 5) GENERATE LIME HEATMAPS (INDEX-BASED)
# -----------------------------
lime_heatmaps = []   # index 0 → class_names[0], etc.

num_classes = len(class_names)

for class_idx in range(num_classes):
    temp, mask = explanation.get_image_and_mask(
        class_idx,
        positive_only=True,
        num_features=10,
        hide_rest=False
    )
    mask = smooth_mask(mask)
    heat_img = apply_heatmap(img_norm, mask, cmap="jet", alpha=0.55)
    lime_heatmaps.append(heat_img)

# -----------------------------
# 6) 1×5 FIGURE: ORIGINAL + 3 LIME + BAR CHART
# -----------------------------
plt.figure(figsize=(28, 6))

# Panel 1 – Original
plt.subplot(1, 5, 1)
plt.imshow(img_norm)
plt.title("Original CXR")
plt.axis("off")

# Panel 2 – Class 0 LIME
plt.subplot(1, 5, 2)
plt.imshow(lime_heatmaps[0])
plt.title(f"{class_names[0]} LIME\n({preds[0]*100:.2f}%)")
plt.axis("off")

# Panel 3 – Class 1 LIME
plt.subplot(1, 5, 3)
plt.imshow(lime_heatmaps[1])
plt.title(f"{class_names[1]} LIME\n({preds[1]*100:.2f}%)")
plt.axis("off")

# Panel 4 – Class 2 LIME
plt.subplot(1, 5, 4)
plt.imshow(lime_heatmaps[2])
plt.title(f"{class_names[2]} LIME\n({preds[2]*100:.2f}%)")
plt.axis("off")

# Panel 5 – Probability Bar Chart
plt.subplot(1, 5, 5)
bars = plt.bar(class_names, preds * 100)
plt.title("Prediction Probabilities (%)")
plt.ylim(0, 100)
plt.ylabel("Probability (%)")

for bar, p in zip(bars, preds * 100):
    h = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., h + 1,
             f"{p:.2f}%", ha='center', va='bottom', fontsize=11)

plt.tight_layout()
plt.show()
