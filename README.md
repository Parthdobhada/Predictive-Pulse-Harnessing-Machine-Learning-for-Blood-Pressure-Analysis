# Predictive-Pulse-Harnessing-Machine-Learning-for-Blood-Pressure-Analysis
using machine learning models prediction of heart disease
# CNN Image Classification Project

## 📌 Overview
This project implements a Convolutional Neural Network (CNN) to classify images into predefined categories. It uses Python and TensorFlow/Keras for model training and evaluation. The model is designed to work directly with raw images without extensive preprocessing, allowing flexibility in real-world applications.

## 📂 Dataset
- The dataset contains labeled images stored in directories (one folder per class).
- Images are loaded directly and resized to a consistent size before being fed to the CNN.
- Supports `.jpg`, `.png`, and `.jpeg` formats.

## 🧠 Model Architecture
The CNN architecture includes:
1. **Convolutional Layers** – for feature extraction  
2. **MaxPooling Layers** – for downsampling  
3. **Dropout Layers** – for reducing overfitting  
4. **Dense Layers** – for classification  

Example model:
```python
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])
