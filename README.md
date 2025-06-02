# Human Activity Recognition (HAR) System

This project implements a Human Activity Recognition (HAR) system using pretrained deep learning models and various supporting tools. It processes video inputs to detect and recognize human actions, with additional features like suspicious activity notifications and face identification.

## Project Overview

The HAR system leverages the Kinetics-400 dataset and a pretrained ResNet-34 model in ONNX format to identify 400 types of human activities. It also integrates YOLOv3 for object detection and uses Twilio APIs for sending SMS notifications upon suspicious actions.

---

## Features

- **Activity Recognition**: Detects and displays human activities from video input.
- **Multiple Action Detection**: Recognizes multiple actions occurring simultaneously.
- **Suspicious Activity Alerts**: Sends notifications to a phone using Twilio if suspicious actions are detected.
- **Action Timestamping**: Extracts time ranges for specific activities.
- **Clip Extraction**: Saves video clips of recognized actions.
- **Light and Fan Monitoring**: Detects when lights/fans are on in an empty room and sends alerts.
- **Face Identification**: Uses MTCNN for face detection alongside activity recognition.
- **Attention Recognition in Classrooms**: Detects attentive and inattentive students during lectures.
- **Web Interface**: Provides a user interface to input videos, view recognized actions with timestamps, and query actions via chatbot.

---

## Dependencies

- Python 3.x
- OpenCV
- ONNX Runtime
- Flask
- Twilio Python SDK
- MTCNN
- Other standard libraries (`numpy`, `requests`, etc.)

---

## Usage

1. Clone the repository.
2. Install the required dependencies (e.g., `pip install -r requirements.txt`).
3. Run `app.py` to start the Flask web app.
4. Open `index.html` in a browser to upload videos and interact with the chatbot for activity queries.
5. For direct testing, run `recognise_human_activity.py` with a test video to see recognized activities displayed.

---

## Credits

- Pretrained models and datasets from the Kinetics-400 dataset and YOLO official releases.
- Twilio for notification services.
- VLC for open-source video playback and repair tools.

---
