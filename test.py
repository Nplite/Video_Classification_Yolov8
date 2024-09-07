import cv2
import numpy as np
from ultralytics import YOLO

# Categories
format = ['Cricket', 'javelin',  'Throw']

# Load the model
model = YOLO("runs/classify/train/weights/best.pt")

# Function to classify video frames
# def video_classifier(frame):
#     result = model.predict(source=frame)
#     # Assuming you need to process the result to extract classification
#     # Modify this based on how the YOLO result is structured
#     # probs = result[0].probs
#     # max_tensor = max(probs)
#     # tensor_pos = (probs == max_tensor).nonzero(as_tuple=True)[0]
#     # return format.get(int(tensor_pos))
#     return result



def video_classifier(frame):
    result = model.predict(source=frame)
    # Assuming you need to process the result to extract classification
    # Modify this based on how the YOLO result is structured
    probs = result[0].probs
    max_tensor = max(probs)
    tensor_pos = (probs == max_tensor).nonzero(as_tuple=True)[0]
    return format.get(int(tensor_pos), "Unknown")


# Initialize webcam
cap = cv2.VideoCapture(1)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break


    classification = video_classifier(frame)
    if classification is not None:
        classification = str(classification)
        cv2.putText(frame, classification, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)


    cv2.putText(frame, classification, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    
    cv2.imshow('Webcam Feed', frame)
    
    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
