# Load the TFLite model
import tensorflow as tf
import numpy as np
import cv2

interpreter = tf.lite.Interpreter(model_path="yolov8n_optimized.tflite")
interpreter.allocate_tensors()

# Get input details
input_details = interpreter.get_input_details()
print("Model Input Shape:", input_details[0]['shape'])

# transpose the image dimensions from (1, 3, 640, 640) to (1, 640, 640, 3) in test1.py
