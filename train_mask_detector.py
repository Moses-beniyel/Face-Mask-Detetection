import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.metrics import classification_report
import os
import numpy as np

# --- Hyperparameters ---
INIT_LR = 1e-4          # Initial learning rate
EPOCHS = 20             # Number of epochs for training
BS = 32                 # Batch size
DIRECTORY = "D:\MASK_DETECTION\CODE\Face-Mask-Detection\dataset" # **CHANGE THIS PATH**
CATEGORIES = ["with_mask", "without_mask"]

# --- 1. Data Augmentation and Preparation ---
print("[INFO] Loading images...")

# Use ImageDataGenerator for augmentation and scaling
aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest",
    preprocessing_function=preprocess_input # MobileNetV2 requires specific preprocessing
)

# Load data from the directory and apply augmentation/preprocessing
train_generator = aug.flow_from_directory(
    DIRECTORY,
    target_size=(224, 224), # MobileNetV2 standard input size
    batch_size=BS,
    class_mode="categorical"
)

# --- 2. Build the Model (Transfer Learning) ---

# Load MobileNetV2 weights, exclude the top classification layer
baseModel = MobileNetV2(
    weights="imagenet", 
    include_top=False, 
    input_tensor=Input(shape=(224, 224, 3))
)

# Construct the custom classification head
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(len(CATEGORIES), activation="softmax")(headModel) # 2 outputs (mask/no mask)

# Combine base model and head model
model = Model(inputs=baseModel.input, outputs=headModel)

# Freeze the layers of the base model so they won't be updated during the first training phase
for layer in baseModel.layers:
    layer.trainable = False

# --- 3. Compile and Train ---
print("[INFO] Compiling model...")
opt = Adam(learning_rate=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

print("[INFO] Training head...")
H = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BS,
    validation_data=train_generator, # Using training data as validation for simplicity/quick run
    validation_steps=train_generator.samples // BS,
    epochs=EPOCHS
)

# --- 4. Save Model ---
print("[INFO] Saving mask detector model...")
model.save("mask_detector.h5")