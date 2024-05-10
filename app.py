import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import time


mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils


# Create a function to manage face detection
def detect_faces(image, model_selection):
    with mp_face_detection.FaceDetection(
        model_selection=model_selection, min_detection_confidence=0.5
    ) as face_detection:
        start_time = time.time()  # Start time for inference measurement
        results = face_detection.process(image)
        inference_time = time.time() - start_time  # End time and calculate duration

        # Draw face detections.
        if results.detections:
            for detection in results.detections:
                mp_drawing.draw_detection(image, detection)
        return image, results, inference_time


st.title("Real-Time Face Detection App with FPS")
model_type = st.sidebar.selectbox("Select the Face Detection Model", ("Short-Range", "Standard"))


cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

# Variables for FPS calculation
prev_frame_time = 0
fps = 0


while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame (stream end?). Exiting...")
        break

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert for mediapipe
    annotated_image, results, inference_time = detect_faces(frame, 0 if model_type == "Short-Range" else 1)
    annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)  # Convert back for OpenCV

    # Calculate FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_frame_time)
    prev_frame_time = current_time

    # Display the resulting frame, FPS, and inference time
    st.image(annotated_image)
    st.sidebar.write(f"FPS: {fps:.1f}")
    st.sidebar.write(f"Inference Time: {inference_time * 1000:.2f} ms")

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
