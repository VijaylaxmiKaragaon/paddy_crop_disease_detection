# 🌾 Paddy Crop Disease Detection using Deep Learning

![Python](https://img.shields.io/badge/Python-3.8-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-DeepLearning-orange)
![Keras](https://img.shields.io/badge/Keras-NeuralNetwork-red)
![Streamlit](https://img.shields.io/badge/Streamlit-WebApp-green)
![Status](https://img.shields.io/badge/Project-Completed-success)

An intelligent **Deep Learning-based system** that detects diseases in **paddy (rice) crops** using leaf images.  
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

## 🦠 Diseases Detected

The model can detect the following paddy leaf conditions:

- 🌿 Healthy Leaf  
- 🍂 Brown Spot  
- 🔥 Leaf Blast  
- 🦠 Bacterial Leaf Blight  
- 🌾 Tungro Virus  

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

## 📂 Project Structure

```
paddy-crop-disease-detection
│
├── dataset
│   ├── train
│   └── test
│
├── model
│   └── trained_model.h5
│
├── app.py
├── train_model.py
├── requirements.txt
│
├── static
├── templates
│
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
