import cv2
from collections import deque, Counter
import numpy as np
from twilio.rest import Client

class Parameters:
    def __init__(self, video_path=None):
        # Load action recognition classes and model
        self.CLASSES = open("C:\\Users\\sanjanasuresh\\OneDrive\\Desktop\\human-activity-recognition-master\\human-activity-recognition-master\\model\\action_recognition_kinetics.txt"
                            ).read().strip().split("\n")
        self.ACTION_RESNET = "C:\\Users\\sanjanasuresh\\OneDrive\\Desktop\\human-activity-recognition-master\\human-activity-recognition-master\\model\\resnet-34_kinetics (1).onnx"
        # Accept a path to the video file (or a default test video path)
        self.VIDEO_PATH = video_path if video_path else "test/test3.mp4"
        self.SAMPLE_DURATION = 16
        self.SAMPLE_SIZE = 112

    def get_video_stream(self):
        # Open a local video file
        return cv2.VideoCapture(self.VIDEO_PATH)

# Initialize instance of Parameters class with a video file path
video_path = "test/realtime.mp4"  # Replace with your video path
param = Parameters(video_path=video_path)

# A double-ended queue to store frames captured
captures = deque(maxlen=param.SAMPLE_DURATION)

# Load the human activity recognition model
print("[INFO] loading human activity recognition model...")
net = cv2.dnn.readNet(model=param.ACTION_RESNET)

# SMS configuration
TWILIO_SID = 'AC1bcab29d13b717060d342fecbc80cd1f'
TWILIO_AUTH_TOKEN = '6aa6f6e8c1594e1e217ac3d3c408499a'
TWILIO_PHONE_NUMBER ='+13412087068'
TO_PHONE_NUMBER = '+918125376258'
client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)

# Function to send SMS
def send_sms(message):
    client.messages.create(
        body=message,
        from_=TWILIO_PHONE_NUMBER,
        to=TO_PHONE_NUMBER
    )
    print("[INFO] SMS sent: " + message)

# Access the video stream
print("[INFO] accessing video stream...")
vs = param.get_video_stream()

# List of fighting activities to monitor
fighting_activities = [
    "arm wrestling", "boxing", "drop kicking", "hitting baseball", "kickboxing", 
    "wrestling", "punching bag/person", "punching person (boxing)", "throwing axe", 
   "eating"
]

# List to keep track of activities over time
activity_list = []

while True:
    # Read a frame from the video stream
    grabbed, capture = vs.read()

    # Break the loop if no frame is grabbed (end of video or stream issue)
    if not grabbed:
        print("[INFO] no capture read from stream - exiting")
        break

    # Resize frame and append it to the deque
    capture = cv2.resize(capture, dsize=(550, 400))
    captures.append(capture)

    # Process further only when the deque is filled
    if len(captures) < param.SAMPLE_DURATION:
        continue

    # Construct image blob for model input
    imageBlob = cv2.dnn.blobFromImages(captures, 1.0, (param.SAMPLE_SIZE, param.SAMPLE_SIZE), (114.7748, 107.7354, 99.4750), swapRB=True, crop=True)

    # Manipulate the image blob for input into the model
    imageBlob = np.transpose(imageBlob, (1, 0, 2, 3))
    imageBlob = np.expand_dims(imageBlob, axis=0)

    # Forward pass through model to make prediction
    net.setInput(imageBlob)
    outputs = net.forward()

    # Get predicted label
    label = param.CLASSES[np.argmax(outputs)]

    # Append the predicted label to activity list
    activity_list.append(label)

    # Show the predicted activity on the frame
    cv2.rectangle(capture, (0, 0), (300, 40), (255, 255, 255), -1)
    cv2.putText(capture, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    # Display the frame with predicted activity
    cv2.imshow("Human Activity Recognition", capture)

    # Break the loop on pressing 'q'
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# After video finishes, determine the top 3 most frequent activities
if activity_list:
    activity_counts = Counter(activity_list)  # Count frequency of activities
    top_3_activities = activity_counts.most_common(3)  # Get top 3 most frequent activities

    print("[INFO] Top 3 most frequent activities detected:")
    for i, (activity, count) in enumerate(top_3_activities, 1):
        print(f"{i}. {activity} - {count} times")

        # Send an SMS if the activity is a fighting activity
        if activity.lower() in fighting_activities:
            alert_message = f"Alert: Detected fighting activity - {activity}"
            send_sms(alert_message)

# Release the video stream and close all OpenCV windows
vs.release()
cv2.destroyAllWindows()
