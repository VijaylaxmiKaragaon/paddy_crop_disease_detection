import tensorflow as tf
from tensorflow.keras.models import load_model
import os

# Paths
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
OLD_MODEL_PATH = os.path.join(BASE_PATH, "rice_model_densenet.h5")
NEW_MODEL_PATH = os.path.join(BASE_PATH, "rice_model_densenet.keras")

print("🔄 Loading old .h5 model...")
model = load_model(OLD_MODEL_PATH, compile=False)

print("💾 Saving in new .keras format...")
model.save(NEW_MODEL_PATH)

print(f"✅ Conversion complete! New model saved at: {NEW_MODEL_PATH}")
