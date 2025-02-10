# import tensorflow as tf
# import numpy as np
# import cv2
import test1

# print(type(test1.image))
# print(test1.image.shape)
# if not isinstance(test1.image, np.ndarray):
#     image = np.array(test1.image, dtype=np.uint8)
# else:
#     image = test1.image

# # Remove batch dimension if present
# if len(image.shape) == 4:  # (1, H, W, C)
#     image = image[0]

# # Ensure dtype is uint8
# image = image.astype(np.uint8)

# def process_output(output_data, threshold=0.5):
#     """Extract valid detections from YOLOv8 output."""
#     boxes, scores, class_ids = [], [], []
    
#     for detection in output_data[0]:
#         confidence = detection[4]  # Object confidence score
#         if confidence > threshold:
#             class_id = np.argmax(detection[5:])  # Get class ID
#             x, y, w, h = detection[:4]  # Bounding box coordinates
#             boxes.append([x, y, w, h])
#             scores.append(confidence)
#             class_ids.append(class_id)
    
#     return boxes, scores, class_ids

# boxes, scores, class_ids = process_output(test1.output_data)

# # Draw boxes on image
# for (box, score, class_id) in zip(boxes, scores, class_ids):
#     x, y, w, h = [int(v) for v in box]
#     cv2.rectangle(test1.image[0], (x, y), (x + w, y + h), (0, 255, 0), 2)
#     cv2.putText(test1.image[0], f"Class {class_id}: {score:.2f}", (x, y - 10),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# cv2.imshow("Detections", test1.image[0])
# cv2.waitKey(0)
# cv2.destroyAllWindows()

import numpy as np
import cv2

# Convert to NumPy if needed
if not isinstance(test1.image, np.ndarray):
    image = np.array(test1.image, dtype=np.uint8)
else:
    image = test1.image

# Remove batch dimension if present
if len(image.shape) == 4:  # (1, H, W, C)
    image = image[0]

# Ensure dtype is uint8
image = image.astype(np.uint8)

# Draw rectangle
x, y, w, h = 100, 150, 200, 250
cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Show image
cv2.imshow("Result", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
