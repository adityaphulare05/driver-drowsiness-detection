# 🚗 Driver Drowsiness Detection System

A real-time computer vision and deep learning based Driver Drowsiness Detection System designed to improve road safety by monitoring a driver's facial and eye conditions.

## 📌 Overview

Driver fatigue and drowsiness are major causes of road accidents. This project uses computer vision, facial landmark detection, and a deep learning based eye-state classification model to monitor the driver in real time.

The system analyzes the driver's eyes, facial landmarks, yawning behavior, gaze direction, and other visual indicators to identify potentially unsafe driving conditions and provide alerts.

## ✨ Features

- 👁️ Real-time eye state detection
- 😴 Drowsiness detection using Eye Aspect Ratio (EAR)
- 🥱 Yawning detection
- 📱 Phone-use detection
- 👀 Gaze direction detection
- 🚨 Audio alerts for detected unsafe conditions
- 🙂 Facial landmark detection using Dlib
- 🧠 CNN-based eye-state classification
- 📷 Real-time camera-based monitoring
- 📊 Eye dataset collection and model training

## 🛠️ Technologies Used

- Python
- OpenCV
- Dlib
- TensorFlow / Keras
- NumPy
- SciPy
- Scikit-learn
- Imutils
- Pygame
- Matplotlib

## 🧠 Machine Learning Model

The project uses a Convolutional Neural Network (CNN) to classify the driver's eye state into:

- Open
- Closed

The eye images are resized to `24 × 24` pixels and normalized before being passed to the CNN model.

The model is trained using data augmentation techniques such as:

- Rotation
- Width shifting
- Height shifting
- Zoom
- Horizontal flipping

## 📂 Project Structure

```text
driver-drowsiness-detection/
│
├── data/
│   └── eyes/
│       ├── open/
│       └── closed/
│
├── models/
│   └── eye_state_model.h5
│
├── .gitignore
├── .gitattributes
├── alert.wav
├── collect_eye_data.py
├── main.py
├── requirements.txt
├── shape_predictor_68_face_landmarks.dat
└── train_eye_model.py
