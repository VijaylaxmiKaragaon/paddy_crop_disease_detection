from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def compare_models():
    # Correct path to your test images folder
    test_dir = r"C:\Users\VIJAYLAXMI SHANKAR K\Downloads\rice-plant-disease-detection-main\rice-plant-disease-detection-mainn\dataset\test_images"

    datagen = ImageDataGenerator(rescale=1./255)

    test_data = datagen.flow_from_directory(
        test_dir,
        target_size=(224, 224),
        batch_size=16,
        class_mode='categorical'
    )

    vgg = load_model("vgg19_model.h5")
    dense = load_model("densenet_model.h5")

    acc_vgg = vgg.evaluate(test_data)[1]
    acc_dense = dense.evaluate(test_data)[1]

    return acc_vgg, acc_dense
