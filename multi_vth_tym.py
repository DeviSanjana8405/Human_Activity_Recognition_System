import cv2
import numpy as np
from collections import deque

# Paths
CLASSES = open(r"C:\Users\sanjanasuresh\OneDrive\Desktop\human-activity-recognition-master\human-activity-recognition-master\model\action_recognition_kinetics.txt").read().strip().split("\n")
MODEL_PATH = r"C:\Users\sanjanasuresh\OneDrive\Desktop\human-activity-recognition-master\human-activity-recognition-master\model\resnet-34_kinetics (1).onnx"
VIDEO_PATH = "test/test3.mp4"
SAMPLE_DURATION = 16
SAMPLE_SIZE = 112

# Load model
print("[INFO] Loading Action Recognition Model...")
net = cv2.dnn.readNet(MODEL_PATH)

# Open video
print("[INFO] Opening video...")
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
frames = deque(maxlen=SAMPLE_DURATION)

frame_counter = 0
second_counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (550, 400))
    resized = cv2.resize(frame, (SAMPLE_SIZE, SAMPLE_SIZE))
    frames.append(resized)
    frame_counter += 1

    # Every 1 second (based on FPS), run prediction
    if frame_counter % int(fps) == 0 and len(frames) == SAMPLE_DURATION:
        # Prepare blob
        blob = cv2.dnn.blobFromImages(frames, 1.0, (SAMPLE_SIZE, SAMPLE_SIZE),
                                      (114.7748, 107.7354, 99.4750),
                                      swapRB=True, crop=True)
        blob = np.transpose(blob, (1, 0, 2, 3))
        blob = np.expand_dims(blob, axis=0)

        net.setInput(blob)
        output = net.forward()
        label = CLASSES[np.argmax(output)]

        mins = int(second_counter // 60)
        secs = int(second_counter % 60)
        print(f"{label} at {mins:02d}:{secs:02d}")
        second_counter += 1

        # Draw label
        cv2.rectangle(frame, (0, 0), (350, 40), (255, 255, 255), -1)
        cv2.putText(frame, f"{label} ({mins:02d}:{secs:02d})", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    cv2.imshow("Action Recognition (Every Second)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
