import tensorflow as tf
import numpy as np
import cv2

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="yolov8n_optimized.tflite")
interpreter.allocate_tensors()

# Get input & output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Check expected input shape
input_shape = input_details[0]['shape']
print("Expected input shape:", input_shape)

# Load and preprocess image
image = cv2.imread("testt.jpg")
image = cv2.resize(image, (640, 640))  # Resize to match YOLOv8 input size
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
image = image.astype(np.float32) / 255.0  # Normalize to [0,1]

# Adjust dimensions based on model expectation
if tuple(input_shape) == (1, 3, 640, 640):  # Some models use (1, C, H, W)
    image = np.transpose(image, (2, 0, 1))  # Convert (H, W, C) -> (C, H, W)

image = np.expand_dims(image, axis=0)  # Add batch dimension

# Run inference
interpreter.set_tensor(input_details[0]['index'], image)
interpreter.invoke()

# Get results
output_data = interpreter.get_tensor(output_details[0]['index'])
print("Detection Results:", output_data)

#Post-processing o/p is done in test2.py
