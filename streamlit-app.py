import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import time

# Initialize mediapipe face detection.
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Setting up the webcam feed.
st.title("Real-Time Face Detection")
run = st.checkbox('Run')
FRAME_WINDOW = st.image([])

# Load face detection model.
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

cap = cv2.VideoCapture(-1)  # Use 0 for webcam

while run:
    ret, frame = cap.read()
    if not ret:
        continue

    # Convert the BGR image to RGB.
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform face detection.
    start_time = time.time()
    results = face_detection.process(image)
    elapsed_time = time.time() - start_time

    # Draw face detections.
    if results.detections:
        for detection in results.detections:
            mp_drawing.draw_detection(image, detection)

            # Extract information for the UI.
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = image.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (255, 0, 0), 2)

            # Calculate and display detection confidence
            confidence = detection.score[0]
            label = f'{confidence * 100:.2f}%'
            cv2.putText(image, label, (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Convert RGB to BGR to display correctly in Streamlit.
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Display the image.
    FRAME_WINDOW.image(image)

    # Show inference time and other details.
    st.sidebar.text(f'Inference Time: {elapsed_time * 1000:.2f} ms')
    st.sidebar.text(f'Face Count: {len(results.detections) if results.detections else 0}')

else:
    st.write('Stopped')

# Release the webcam and close all OpenCV windows
cap.release()
