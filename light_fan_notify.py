import cv2
import numpy as np
from twilio.rest import Client

# Twilio credentials (replace with your own)
TWILIO_SID = 'AC1bcab29d13b717060d342fecbc80cd1f'
TWILIO_TOKEN = '6aa6f6e8c1594e1e217ac3d3c408499a'
FROM_PHONE = '+13412087068'
TO_PHONE = '+918125376258'

def send_sms(msg):
    client = Client(TWILIO_SID, TWILIO_TOKEN)
    client.messages.create(body=msg, from_=FROM_PHONE, to=TO_PHONE)
    print("[INFO] SMS SENT:", msg)

# Load YOLO model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = open("coco.names").read().strip().split("\n")
layer_names = net.getLayerNames()
output_layers = [layer_names[int(i) - 1] for i in net.getUnconnectedOutLayers()]

cap = cv2.VideoCapture("test/light_fan_off.mp4")
prev_frame = None

empty_frame_count = 0
alert_sent = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    ### --- Light brightness detection region (Tube lights only) --- ###
    # Focus on a narrow center region just under the ceiling
    light_region = gray[30:60, 100:540]  # Excludes top and window sides
    brightness = np.mean(light_region)

    ### --- Fan motion detection region (adjust if needed) --- ###
    fan_region = gray[50:200, :]  # Adjust these values if fan region differs
    motion_score = 0
    if prev_frame is not None:
        prev_fan_region = prev_frame[50:200, :]
        diff = cv2.absdiff(prev_fan_region, fan_region)
        motion_score = np.mean(diff)

    prev_frame = gray.copy()

    print(f"[DEBUG] Brightness (light zone): {brightness:.2f}, Fan motion score: {motion_score:.2f}")

    ### --- Person detection using YOLO --- ###
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416,416), swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward(output_layers)

    person_found = False
    for output in detections:
        for obj in output:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if class_id == 0 and confidence > 0.5:
                person_found = True
                break

    ### --- Lighting & Fan status logic --- ###
    lights_on = brightness > 80  # Updated threshold for artificial lighting
    fans_on = motion_score > 1.5   # Adjust if fan detection is too sensitive

    ### --- Alert logic --- ###
    if not person_found and (lights_on or fans_on):
        empty_frame_count += 1
    else:
        empty_frame_count = 0
        alert_sent = False

    if empty_frame_count > 10 and not alert_sent:
        if lights_on and fans_on:
            alert_msg = "⚠️ Alert: Room is empty but lights and fans are ON!"
        elif lights_on:
            alert_msg = "⚠️ Alert: Room is empty but lights are ON!"
        elif fans_on:
            alert_msg = "⚠️ Alert: Room is empty but fans are ON!"
        else:
            alert_msg = None

        if alert_msg:
            send_sms(alert_msg)
            alert_sent = True
            break

    ### --- Display status on frame --- ###
    status = f"Lights ON: {'Yes' if lights_on else 'No'}, Fans ON: {'Yes' if fans_on else 'No'}"
    cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    cv2.imshow("Room Monitor", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
