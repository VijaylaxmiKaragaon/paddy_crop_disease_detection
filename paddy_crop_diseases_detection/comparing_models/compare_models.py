import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG19, DenseNet121
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset", "train_images")

# --- Parameters ---
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
LR = 0.0001

# --- Data Generator ---
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

train_gen = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True
)

num_classes = train_gen.num_classes


# =============================
# 📌 BUILD MODEL FUNCTION
# =============================
def build_model(base_model):
    base_model.trainable = False  # freeze layers

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.5)(x)
    output = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer=Adam(LR), loss="categorical_crossentropy", metrics=["accuracy"])

    return model


# =============================
# 🚀 Train VGG19
# =============================
print("\n============================")
print("Training VGG19")
print("============================")

vgg_base = VGG19(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
model_vgg = build_model(vgg_base)

history_vgg = model_vgg.fit(train_gen, epochs=EPOCHS, verbose=1)

loss_vgg, acc_vgg = model_vgg.evaluate(train_gen)
print(f"\n📌 VGG19 Accuracy: {acc_vgg*100:.2f}%\n")


# =============================
# 🚀 Train DenseNet121
# =============================
print("\n============================")
print("Training DenseNet121")
print("============================")

dense_base = DenseNet121(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
model_dense = build_model(dense_base)

history_dense = model_dense.fit(train_gen, epochs=EPOCHS, verbose=1)

loss_dense, acc_dense = model_dense.evaluate(train_gen)
print(f"\n📌 DenseNet121 Accuracy: {acc_dense*100:.2f}%\n")


# =============================
# 🏆 Compare Models
# =============================
print("\n============================")
print("🏆 MODEL COMPARISON RESULT")
print("============================")

print(f"🔹 VGG19 Accuracy:       {acc_vgg*100:.2f}%")
print(f"🔹 DenseNet121 Accuracy: {acc_dense*100:.2f}%")

if acc_vgg > acc_dense:
    print("\n✅ BEST MODEL: VGG19")
else:
    print("\n✅ BEST MODEL: DenseNet121")
