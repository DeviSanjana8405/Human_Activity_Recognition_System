from flask import Flask, render_template, request, jsonify
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
    return render_template('index.html', activities=activities)

@app.route('/process', methods=['POST'])
def process_video():
    global activities
    video = request.files['video']
    video_path = os.path.join("C:/Users/sanjanasuresh/OneDrive/Desktop/human-activity-recognition-master/human-activity-recognition-master/uploads", video.filename)
    video.save(video_path)

    # Video processing
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

        # Store activities with start and end time
        if label != current_activity:
            if current_activity is not None:
                activities.append((current_activity, activity_start_time, activity_end_time))
            current_activity = label
            activity_start_time = vs.get(cv2.CAP_PROP_POS_MSEC)

        activity_end_time = vs.get(cv2.CAP_PROP_POS_MSEC)


        cv2.rectangle(capture, (0, 0), (300, 40), (255, 255, 255), -1)
        cv2.putText(capture, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 0), 2)

        cv2.imshow("Human Activity Recognition", capture)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    activities.append((current_activity, activity_start_time, activity_end_time))  # Add last activity
    vs.release()

    return render_template('index.html', activities=activities)


@app.route('/chatbot', methods=['POST'])
def chatbot():
    query = request.form.get('query')
    for activity, start_time, end_time in activities:
        if query.lower() in activity.lower():
            return jsonify({'response': f'{activity} started at {start_time} ms and ended at {end_time} ms'})
    return jsonify({'response': 'Activity not found'})


if __name__ == '__main__':
    app.run(debug=True)