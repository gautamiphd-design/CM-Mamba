import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory

def load_datasets(train_path, val_path, img_size=(224,224), batch_size=32):

    train_ds = image_dataset_from_directory(
        train_path,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='int',
        shuffle=True
    )

    val_ds = image_dataset_from_directory(
        val_path,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='int',
        shuffle=False
    )

    class_names = train_ds.class_names
    print("Classes:", class_names)

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)

    return train_ds, val_ds, class_names
