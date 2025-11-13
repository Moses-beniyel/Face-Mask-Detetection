import tensorflow as tf
from keras.layers import TFSMLayer
import keras

try:
    # Try loading as TFSMLayer for Keras 3
    model = TFSMLayer("mask_detector.model", call_endpoint='serving_default')
    
    # Create a functional model wrapper
    inputs = keras.Input(shape=(224, 224, 3))  # Adjust shape based on your model
    outputs = model(inputs)
    functional_model = keras.Model(inputs=inputs, outputs=outputs)
    
    # Save as .h5 format
    functional_model.save("mask_detector.model")
    print("Model successfully converted to H5 format!")
    
except Exception as e:
    print(f"Error: {e}")