## importing essential Libraries

import os
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess_input
import numpy as np

# Disable TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# --- Configuration ---
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_PATH, '..', 'dataset', 'train_images')
MODEL_PATH = os.path.join(BASE_PATH, '..', 'model', 'rice_model_densenet.keras')

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

# --- Check for existing model ---
if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 0:
    print(f"✅ Found existing trained model at {MODEL_PATH}. Loading instead of retraining...")
    model_densenet = load_model(MODEL_PATH)
    print("Model loaded successfully!")
else:
    # --- Data Loading and Preprocessing ---
    print("📦 Loading and preprocessing data...")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_PATH,
        labels='inferred',
        label_mode='categorical',
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    validation_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_PATH,
        labels='inferred',
        label_mode='categorical',
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    class_names = train_ds.class_names
    num_classes = len(class_names)
    print(f"Detected classes: {class_names}")

    # --- DenseNet Model Training ---
    print("\n🚀 Training DenseNet121 Model...")
    densenet_train_ds = train_ds.map(lambda x, y: (densenet_preprocess_input(x), y))
    densenet_validation_ds = validation_ds.map(lambda x, y: (densenet_preprocess_input(x), y))

    base_model_densenet = DenseNet121(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    base_model_densenet.trainable = False
    x = base_model_densenet.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model_densenet = Model(inputs=base_model_densenet.input, outputs=predictions)

    model_densenet.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model_densenet.fit(densenet_train_ds, validation_data=densenet_validation_ds, epochs=EPOCHS)

    # Save the trained model
    print(f"💾 Saving DenseNet model to {MODEL_PATH}...")
    model_densenet.save(MODEL_PATH)
    print("✅ Model saved successfully!")

print("\n✅ Ready to use model for predictions.")
