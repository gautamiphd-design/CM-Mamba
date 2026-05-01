import tensorflow as tf
import datetime
import matplotlib.pyplot as plt

from dataset import load_datasets
from model import create_model

# =====================
# Paths (update)
# =====================
train_path = "DATASET_PATH/train"
val_path   = "DATASET_PATH/val"

# =====================
# Load data
# =====================
train_ds, val_ds, class_names = load_datasets(train_path, val_path)

# =====================
# Create model
# =====================
model = create_model(input_shape=(224,224,3), num_classes=len(class_names))

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# =====================
# Callbacks (MATCH PAPER)
# =====================
log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir=log_dir),
    tf.keras.callbacks.EarlyStopping(
        patience=100, monitor='val_loss', restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        "best_medmamba_model_CXR.keras",
        save_best_only=True,
        monitor='val_loss'
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )
]

# =====================
# Train
# =====================
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=100,
    callbacks=callbacks
)

# =====================
# Save weights
# =====================
model.save_weights("best_medmamba_weights.h5")

# =====================
# Plot training curves
# =====================
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Val')
plt.title("Accuracy")
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Val')
plt.title("Loss")
plt.legend()

plt.tight_layout()
plt.savefig("training_curves.png", dpi=300)
plt.show()
