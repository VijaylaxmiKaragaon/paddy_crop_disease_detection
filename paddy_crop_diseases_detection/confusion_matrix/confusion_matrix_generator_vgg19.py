import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# --------------------------
# LOAD MODEL
# --------------------------
model = tf.keras.models.load_model("rice_model_densenet.keras")

# --------------------------
# LOAD TEST DATASET
# --------------------------
img_height = 224
img_width = 224
batch_size = 32

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "dataset/train_images",   # <-- Your test folder
    image_size=(img_height, img_width),
    batch_size=batch_size,
    shuffle=False
)

class_names = test_ds.class_names
print("Classes:", class_names)

# --------------------------
# GET TRUE LABELS
# --------------------------
y_true = np.concatenate([y for x, y in test_ds], axis=0)

# --------------------------
# GET MODEL PREDICTIONS
# --------------------------
y_pred = model.predict(test_ds)
y_pred_labels = np.argmax(y_pred, axis=1)

# --------------------------
# CONFUSION MATRIX
# --------------------------
cm = confusion_matrix(y_true, y_pred_labels)
print("\nConfusion Matrix:\n")
print(cm)

# --------------------------
# CLASSIFICATION REPORT
# --------------------------
print("\nClassification Report:\n")
print(classification_report(y_true, y_pred_labels, target_names=class_names))
