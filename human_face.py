# Required imports
from collections import deque
import numpy as np
import cv2
from mtcnn import MTCNN

# Parameters class
class Parameters:
    def __init__(self):
        self.CLASSES = open(
            "C:\\Users\\sanjanasuresh\\OneDrive\\Desktop\\human-activity-recognition-master\\human-activity-recognition-master\\model\\action_recognition_kinetics.txt"
        ).read().strip().split("\n")
        self.ACTION_RESNET = "C:\\Users\\sanjanasuresh\\OneDrive\\Desktop\\human-activity-recognition-master\\human-activity-recognition-master\\model\\resnet-34_kinetics.onnx"
        self.VIDEO_PATH = "test/cctv_footage.mp4"
        self.SAMPLE_DURATION = 16
        self.SAMPLE_SIZE = 112

# Initialize Parameters and deque
param = Parameters()
captures = deque(maxlen=param.SAMPLE_DURATION)

# Initialize MTCNN face detector
face_detector = MTCNN()

# Load human activity recognition model
print("[INFO] loading human activity recognition model...")
net = cv2.dnn.readNet(model=param.ACTION_RESNET)

print("[INFO] accessing video stream...")
vs = cv2.VideoCapture(param.VIDEO_PATH if param.VIDEO_PATH else 0)

while True:
    # Read frame
    (grabbed, capture) = vs.read()
    if not grabbed:
        print("[INFO] no capture read from stream - exiting")
        break

    # Resize for consistent processing
    capture = cv2.resize(capture, dsize=(550, 400))
    captures.append(capture)

    # Face Detection using MTCNN
    rgb_frame = cv2.cvtColor(capture, cv2.COLOR_BGR2RGB)
    faces = face_detector.detect_faces(rgb_frame)
    for face in faces:
        x, y, w, h = face['box']
        cv2.rectangle(capture, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(capture, "Face", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 2)

    # Process only when enough frames are collected
    if len(captures) < param.SAMPLE_DURATION:
        cv2.imshow("Human Activity Recognition + Face Detection", capture)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        continue

    # Prepare input blob
    imageBlob = cv2.dnn.blobFromImages(
        captures, 1.0,
        (param.SAMPLE_SIZE, param.SAMPLE_SIZE),
        (114.7748, 107.7354, 99.4750),
        swapRB=True, crop=True
    )
    imageBlob = np.transpose(imageBlob, (1, 0, 2, 3))
    imageBlob = np.expand_dims(imageBlob, axis=0)

    # Activity Recognition Prediction
    net.setInput(imageBlob)
    outputs = net.forward()
    label = param.CLASSES[np.argmax(outputs)]

    # Display Prediction
    cv2.rectangle(capture, (0, 0), (300, 40), (255, 255, 255), -1)
    cv2.putText(capture, f"Activity: {label}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    # Show the result
    cv2.imshow("Human Activity Recognition + Face Detection", capture)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Clean up
vs.release()
cv2.destroyAllWindows()
