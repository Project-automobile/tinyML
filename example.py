# from ultralytics import YOLO
# model = YOLO("yolov8n.pt")  # Load a pre-trained YOLOv8 model
# print(model)

from ultralytics import YOLO
import cv2

# Load pre-trained YOLO model
model = YOLO("yolov8n.pt")

# Run inference on an image
# results = model("testt.jpg", show=True)

# # Save the output
# cv2.imwrite("output.jpg", results[0].plot())
model.export(format = 'onnx', opset=13)