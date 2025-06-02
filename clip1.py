from flask import Flask, render_template, request, jsonify, send_from_directory
import cv2
import numpy as np
from collections import deque
import os

app = Flask(__name__)

# Parameters class include important paths and constants
class Parameters:
    def __init__(self):
        self.CLASSES = open("C:/Users/sanjanasuresh/OneDrive/Desktop/human-activity-recognition-master/human-activity-recognition-master/model/action_recognition_kinetics.txt"
                            ).read().strip().split("\n")
        self.ACTION_RESNET = "C:/Users/sanjanasuresh/OneDrive/Desktop/human-activity-recognition-master/human-activity-recognition-master/model/resnet-34_kinetics.onnx"
        self.SAMPLE_DURATION = 16
        self.SAMPLE_SIZE = 112

param = Parameters()

captures = deque(maxlen=param.SAMPLE_DURATION)
activities = []  # To store activities with their start and end times

# Load the human activity recognition model
print("[INFO] loading human activity recognition model...")
net = cv2.dnn.readNet(model=param.ACTION_RESNET)

@app.route('/')
def index():
    return render_template('index1.html', activities=activities)

@app.route('/process', methods=['POST'])
def process_video():
    global activities
    activities.clear()
    video = request.files['video']
    upload_dir = "C:/Users/sanjanasuresh/OneDrive/Desktop/human-activity-recognition-master/human-activity-recognition-master/uploads"
    os.makedirs(upload_dir, exist_ok=True)
    video_path = os.path.join(upload_dir, video.filename)
    video.save(video_path)

    vs = cv2.VideoCapture(video_path)
    activity_start_time = None
    activity_end_time = None
    current_activity = None

    while True:
        grabbed, capture = vs.read()
        if not grabbed:
            break
        capture = cv2.resize(capture, dsize=(550, 400))
        captures.append(capture)

        if len(captures) < param.SAMPLE_DURATION:
            continue

        imageBlob = cv2.dnn.blobFromImages(captures, 1.0,
                                           (param.SAMPLE_SIZE, param.SAMPLE_SIZE),
                                           (114.7748, 107.7354, 99.4750),
                                           swapRB=True, crop=True)
        imageBlob = np.transpose(imageBlob, (1, 0, 2, 3))
        imageBlob = np.expand_dims(imageBlob, axis=0)

        net.setInput(imageBlob)
        outputs = net.forward()
        label = param.CLASSES[np.argmax(outputs)]

        if label != current_activity:
            if current_activity is not None:
                activities.append((current_activity, activity_start_time, activity_end_time))
            current_activity = label
            activity_start_time = vs.get(cv2.CAP_PROP_POS_MSEC)

        activity_end_time = vs.get(cv2.CAP_PROP_POS_MSEC)

        # Optional: Show video with label during processing
        cv2.rectangle(capture, (0, 0), (300, 40), (255, 255, 255), -1)
        cv2.putText(capture, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 0), 2)
        cv2.imshow("Human Activity Recognition", capture)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    if current_activity is not None:
        activities.append((current_activity, activity_start_time, activity_end_time))
    vs.release()
    cv2.destroyAllWindows()

    return render_template('index1.html', activities=activities)


@app.route('/search_activities', methods=['POST'])
def search_activities():
    data = request.get_json()
    start_time = data.get('start_time')
    end_time = data.get('end_time')

    matched = []
    for activity, s_time, e_time in activities:
        # Check overlap
        if s_time <= end_time and e_time >= start_time:
            matched.append({
                'activity': activity,
                'start_time': s_time,
                'end_time': e_time
            })

    return jsonify({'activities': matched})


@app.route('/uploads/<path:filename>')
def download_clip(filename):
    upload_dir = "C:/Users/sanjanasuresh/OneDrive/Desktop/human-activity-recognition-master/human-activity-recognition-master/uploads"
    return send_from_directory(upload_dir, filename)


if __name__ == '__main__':
    app.run(debug=True)
