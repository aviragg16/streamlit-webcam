import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import time

# Initialize MediaPipe face detection.
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Create a function to manage face detection
def detect_faces(image, model_selection):
    with mp_face_detection.FaceDetection(
        model_selection=model_selection, min_detection_confidence=0.5) as face_detection:
        
        # Convert the BGR image to RGB and process it.
        results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Draw face detections.
        if results.detections:
            for detection in results.detections:
                mp_drawing.draw_detection(image, detection)
    return image, results

# Streamlit UI
st.title("Real-Time Face Detection App")
model_type = st.sidebar.selectbox("Select the Face Detection Model", ("Short-Range", "Standard]"))
run = st.checkbox("Run")

# Webcam setup
cap = cv2.VideoCapture(0)

# Streamlit frame setting
FRAME_WINDOW = st.image([])

while run:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Choose model based on dropdown: 0 for Short-Range, 1 for Full-Range
    model_selection = 0 if model_type == "Short-Range" else 1
    
    start_time = time.time()
    annotated_image, results = detect_faces(frame, model_selection)
    elapsed_time = time.time() - start_time
    
    # Convert annotated image from BGR to RGB for displaying
    annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    
    FRAME_WINDOW.image(annotated_image)
    st.sidebar.write(f"Inference Time: {elapsed_time * 1000:.2f} ms")
    st.sidebar.write(f"Detected Faces: {len(results.detections) if results.detections else 0}")
else:
    st.write("Stopped")

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
