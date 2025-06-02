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
activities = []  # Store (label, start_time, end_time)
video_path = None  # Store uploaded video path
net = cv2.dnn.readNet(model=param.ACTION_RESNET)


@app.route('/')
def index():
    return render_template('index2.html', activities=activities)


@app.route('/process', methods=['POST'])
def process_video():
    global activities, video_path
    activities.clear()  # Reset old activities

    video = request.files['video']
    video_path = os.path.join("uploads", video.filename)
    video.save(video_path)

    vs = cv2.VideoCapture(video_path)
    activity_start_time = None
    activity_end_time = None
    current_activity = None

    while True:
        grabbed, capture = vs.read()
        if not grabbed:
            break

        capture = cv2.resize(capture, (550, 400))
        captures.append(capture)

        if len(captures) < param.SAMPLE_DURATION:
            continue

        blob = cv2.dnn.blobFromImages(captures, 1.0, (param.SAMPLE_SIZE, param.SAMPLE_SIZE),
                                      (114.7748, 107.7354, 99.4750), swapRB=True, crop=True)
        blob = np.transpose(blob, (1, 0, 2, 3))
        blob = np.expand_dims(blob, axis=0)

        net.setInput(blob)
        outputs = net.forward()
        label = param.CLASSES[np.argmax(outputs)]

        if label != current_activity:
            if current_activity is not None:
                activities.append((current_activity, activity_start_time, activity_end_time))
            current_activity = label
            activity_start_time = vs.get(cv2.CAP_PROP_POS_MSEC)

        activity_end_time = vs.get(cv2.CAP_PROP_POS_MSEC)

        # Optional: Display window (can comment out if not needed)
        cv2.rectangle(capture, (0, 0), (300, 40), (255, 255, 255), -1)
        cv2.putText(capture, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 0), 2)
        cv2.imshow("Human Activity Recognition", capture)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    if current_activity is not None:
        activities.append((current_activity, activity_start_time, activity_end_time))

    vs.release()
    cv2.destroyAllWindows()
    return render_template('index2.html', activities=activities)


@app.route('/chatbot', methods=['POST'])
def chatbot():
    query = request.form.get('query')
    global video_path
    for activity, start_time, end_time in activities:
        if query.lower() in activity.lower():
            start_sec = start_time / 1000.0  # milliseconds to seconds
            # Return the relative URL path for the uploaded video
            video_url = '/uploads/' + os.path.basename(video_path)
            return jsonify({
                'response': f'{activity} started at {int(start_time)} ms and ended at {int(end_time)} ms',
                'play_url': video_url,
                'start_time_sec': start_sec
            })
    return jsonify({'response': 'Activity not found'})


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('uploads', filename)


if __name__ == '__main__':
    if not os.path.exists("uploads"):
        os.makedirs("uploads")
    app.run(debug=True)
