# 🌾 Paddy Crop Disease Detection using Deep Learning

![Python](https://img.shields.io/badge/Python-3.8-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-DeepLearning-orange)
![Keras](https://img.shields.io/badge/Keras-NeuralNetwork-red)
![Streamlit](https://img.shields.io/badge/Streamlit-WebApp-green)
![Status](https://img.shields.io/badge/Project-Completed-success)

An AI-powered system that detects **Paddy Crop Diseases Detection** using **Deep Learning and Computer Vision**.
The system helps farmers and agricultural experts identify crop diseases early and take preventive actions to reduce yield loss.

---

## 📌 Project Overview

Paddy is one of the most widely cultivated crops in the world. However, various diseases can severely impact its productivity. Traditional disease detection methods require manual inspection by experts, which is time-consuming and sometimes inaccurate.

This project uses **Deep Learning and Computer Vision** to automatically detect diseases from **paddy leaf images**.

The system analyzes the uploaded leaf image and predicts the disease category with high accuracy.

---

## 🎯 Objectives

- Detect paddy crop diseases automatically  
- Assist farmers with early disease diagnosis  
- Reduce manual crop inspection  
- Improve agricultural productivity  
- Promote smart farming techniques  

---


# 🦠 Diseases Detected

The model classifies **10 different classes**:

1. bacterial_leaf_blight  
2. bacterial_leaf_streak  
3. bacterial_panicle_blight  
4. blast  
5. brown_spot  
6. dead_heart  
7. downy_mildew  
8. hispa  
9. normal (healthy leaf)  
10. tungro  

---

# 📊 Dataset

The dataset contains **images of paddy leaves** belonging to different disease categories.

Dataset characteristics:

- Image type: Paddy leaf images  
- Number of classes: **10**
- Format: JPG / PNG images
- Data split:
  - Training set
  - Testing set

Images were preprocessed before training:

- Image resizing
- Normalization
- Data augmentation (rotation, zoom, flipping)

---

## 🧠 Technologies Used

### Programming
- Python

### Deep Learning
- CNN (Convolutional Neural Network)
- VGG19
- DenseNet

### Libraries
- TensorFlow
- Keras
- NumPy
- OpenCV
- Matplotlib

### Web Interface
- Streamlit
- HTML
- CSS

### Development Tools
- VS Code
- Jupyter Notebook

---

## 🏗 System Workflow

```
Leaf Image → Image Preprocessing → Feature Extraction → Deep Learning Model → Disease Prediction
```

### Steps

1. Image Collection  
2. Image Preprocessing  
3. Data Augmentation  
4. Model Training  
5. Disease Classification  
6. Prediction using Web Interface  

---

## 📊 Model Performance

| Model | Accuracy |
|------|---------|
| CNN | >90% |
| VGG19 | ~82% |
| DenseNet | **>94.44% (Best)** |

DenseNet achieved the highest accuracy in detecting paddy leaf diseases.

---

# 📂 Project Structure

```
rice-plant-disease-detection
│
├── dataset
│   ├── train_images
│   └── test_images
│
├── model
│   ├── rice_model_densenet.h5
│   └── rice_model_vgg19.h5
│
├── training_models
│   ├── train_model.py
│   └── train_vgg19_model.py
│
├── frontend
│   └── all_full_features.py
│
├── confusion_matrix
├── comparing_models
│
├── requirements.txt
├── LICENSE
└── README.md
```

---

## ▶️ How to Run the Project

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/yourusername/paddy-crop-disease-detection.git
```

### 2️⃣ Navigate to Project Folder

```bash
cd paddy-crop-disease-detection
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run the Application

```bash
streamlit run app.py
```

### 5️⃣ Upload Paddy Leaf Image

Upload a leaf image and the system will **predict the disease instantly**.

---

## 🌱 Future Enhancements

- Mobile app for farmers  
- Drone-based crop monitoring  
- IoT integration for smart farming  
- Multi-crop disease detection  
- Cloud-based agricultural monitoring  

---

## 📜 License

This project is developed for **educational and research purposes only**.

---

## ⭐ Support

If you like this project:

⭐ Star the repository  
🍴 Fork the repository  
📢 Share with others
