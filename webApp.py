import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np

# Initialize YOLO model
model = YOLO('C:/Users/UNS_CT/Desktop/CPE Kmutt Term1/Term1_1-67/CPE101/Final project/Helmet Detection_YOLOv8.v3-with-and-without-helmet-dataset.yolov8/runs/detect/train/weights/best.pt')

# Streamlit app title
st.title("YOLOv8 Live Inference with Streamlit")

# Start and stop the camera
start = st.button("Start Camera")
stop = st.button("Stop Camera")

# Display area for the video feed
frame_placeholder = st.empty()

# Initialize the video capture object
cap = None
if start:
    cap = cv2.VideoCapture(0)  # Default camera
    if not cap.isOpened():
        st.error("Error: Cannot open camera")
        cap = None

if cap:
    while not stop:
        ret, frame = cap.read()
        if not ret:
            st.error("Error: Can't receive frame")
            break
        
        # Perform YOLO inference
        results = model(frame)
        annotated_frame = results[0].plot()

        # Convert the frame to RGB format for Streamlit
        rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

        # Display the frame in Streamlit
        frame_placeholder.image(rgb_frame, channels="RGB", use_column_width=True)

        # Break the loop if the "Stop Camera" button is pressed
        if stop:
            break

# Release resources
if cap:
    cap.release()
cv2.destroyAllWindows()
