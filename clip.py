from flask import Flask, render_template, request, jsonify, send_from_directory
import cv2
import numpy as np
from collections import deque
import os

app = Flask(__name__)

# Parameters class include important paths and constants
class Parameters:
    def __init__(self):
        self.CLASSES = open(
            "C:/Users/sanjanasuresh/OneDrive/Desktop/human-activity-recognition-master/human-activity-recognition-master/model/action_recognition_kinetics.txt"
        ).read().strip().split("\n")
        self.ACTION_RESNET = "C:/Users/sanjanasuresh/OneDrive/Desktop/human-activity-recognition-master/human-activity-recognition-master/model/resnet-34_kinetics.onnx"
        self.SAMPLE_DURATION = 16
        self.SAMPLE_SIZE = 112

param = Parameters()
captures = deque(maxlen=param.SAMPLE_DURATION)
activities = []

# Load the human activity recognition model
print("[INFO] loading human activity recognition model...")
net = cv2.dnn.readNet(model=param.ACTION_RESNET)

@app.route('/')
def index():
    return render_template('index.html', activities=activities)

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

        imageBlob = cv2.dnn.blobFromImages(
            captures, 1.0,
            (param.SAMPLE_SIZE, param.SAMPLE_SIZE),
            (114.7748, 107.7354, 99.4750),
            swapRB=True, crop=True
        )
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
    return render_template('index.html', activities=activities)

@app.route('/chatbot', methods=['POST'])
def chatbot():
    query = request.form.get('query')
    upload_dir = "C:/Users/sanjanasuresh/OneDrive/Desktop/human-activity-recognition-master/human-activity-recognition-master/uploads"
    latest_video = os.path.join(upload_dir, sorted(os.listdir(upload_dir))[-1])

    for i, (activity, start_time, end_time) in enumerate(activities):
        if query.lower() in activity.lower():
            os.makedirs(upload_dir, exist_ok=True)
            clip_filename = f'{activity.replace(" ", "_")}_{i}.mp4'
            clip_path = os.path.join(upload_dir, clip_filename)

            cap = cv2.VideoCapture(latest_video)
            fps = cap.get(cv2.CAP_PROP_FPS)
            if not fps or fps <= 1e-2:
                fps = 30  # fallback if FPS can't be detected

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            start_frame = int((start_time / 1000.0) * fps)
            end_frame = int((end_time / 1000.0) * fps)

            # Clamp to bounds
            start_frame = max(0, min(start_frame, total_frames - 1))
            end_frame = max(start_frame + 1, min(end_frame, total_frames - 1))

            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            ret, frame = cap.read()
            if not ret:
                cap.release()
                return jsonify({'response': 'Error reading the original video.', 'clip': ''})

            height, width = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(clip_path, fourcc, fps, (width, height))

            out.write(frame)
            current_frame = start_frame + 1
            frames_written = 1

            while current_frame <= end_frame:
                ret, frame = cap.read()
                if not ret:
                    break
                out.write(frame)
                current_frame += 1
                frames_written += 1

            cap.release()
            out.release()

            if not os.path.exists(clip_path) or os.path.getsize(clip_path) == 0:
                return jsonify({'response': 'Failed to generate video clip.', 'clip': ''})

            print(f"[INFO] Clip saved: {clip_path} | Frames: {frames_written} | Size: {os.path.getsize(clip_path)} bytes")

            return jsonify({
                'response': f'{activity} happened between {start_time:.0f} ms and {end_time:.0f} ms',
                'clip': f'/uploads/{clip_filename}'
            })

    return jsonify({'response': 'Activity not found', 'clip': ''})


@app.route('/uploads/<path:filename>')
def download_clip(filename):
    upload_dir = "C:/Users/sanjanasuresh/OneDrive/Desktop/human-activity-recognition-master/human-activity-recognition-master/uploads"
    return send_from_directory(upload_dir, filename)

if __name__ == '__main__':
    app.run(debug=True)
