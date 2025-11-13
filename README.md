👇

😷 Face Mask Detection using CNN

A Deep Learning project that detects whether a person is wearing a face mask or not in real-time using a Convolutional Neural Network (CNN) built with TensorFlow and Keras.

🚀 Overview

This project was created to help promote safety during the COVID-19 pandemic by automatically detecting if individuals are wearing masks in images or live camera feeds.
It uses OpenCV for real-time face detection and a CNN model trained on a dataset of masked and unmasked faces for classification.

🧠 Features

✅ Real-time face mask detection using webcam
✅ High accuracy CNN-based model
✅ Works on both images and live video streams
✅ Built with TensorFlow, Keras, and OpenCV
✅ Lightweight and easy to deploy

🧩 Tech Stack

Languages & Frameworks:

Python

TensorFlow / Keras

OpenCV

NumPy

Matplotlib

📂 Project Structure
Face-Mask-Detection/
│
├── dataset/
│   ├── with_mask/
│   └── without_mask/
│
├── face_mask_detector.model   # Trained CNN model file
├── detect_mask_video.py       # Real-time detection script
├── train_mask_detector.py     # Model training script
├── requirements.txt           # Dependencies
├── README.md                  # Project documentation
└── utils/                     # Helper functions (optional)

⚙️ Installation
1️⃣ Clone the repository
git clone https://github.com/Moses-beniyel/Face-Mask-Detection.git
cd Face-Mask-Detection

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Train the model 
python train_mask_detector.py

4️⃣ Run real-time detection
python detect_mask_video.py

🧪 How It Works

Face Detection:
OpenCV’s pre-trained Haar Cascade or DNN model locates faces in the frame.

Preprocessing:
The detected face is resized and normalized to match the CNN’s input shape.

Prediction:
The trained CNN model classifies the face as “Mask” or “No Mask.”

Visualization:
A bounding box and label (Mask / No Mask) are displayed in real-time.


