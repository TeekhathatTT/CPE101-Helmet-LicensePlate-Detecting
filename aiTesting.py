import cv2
import streamlit as st
from ultralytics import YOLO
import numpy as np
import iapp_ai
import tempfile
import os

# Initialize the API and load models
api_key = '5ysBx4ft4yeZ9PYiU3kvK93EmI6awXKr'
api = iapp_ai.api(api_key)
person_model = YOLO('motor.pt')
helmet_model = YOLO('helmet.pt')

# Define target classes and their colors
target_classes_person = {'person': (0, 255, 0), 'motorcycle': (255, 0, 0)}
target_classes_helmet = {'With Helmet': (0, 255, 0), 'Without Helmet': (0, 0, 255)}

# Streamlit app
st.title("Video Upload and Object Detection")

# Upload video file
video_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

if video_file is not None:
    # Save the uploaded video to a temporary file
    temp_video_path = tempfile.mktemp(suffix=".mp4")
    with open(temp_video_path, "wb") as f:
        f.write(video_file.read())

    # Load the video
    cap = cv2.VideoCapture(temp_video_path)
    if not cap.isOpened():
        st.error("Error: Cannot open video source")
        st.stop()

    k = 0
    frame_placeholder = st.empty()
    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Error: Can't receive frame")
            break

        # Define the region of interest (ROI) for person detection
        # roi_x1, roi_y1, roi_x2, roi_y2 = 500, 0, 700, 500
        # frame_use = frame[roi_y1:roi_y2, roi_x1:roi_x2]
        frame_use = frame

        # Person detection
        results_person = person_model(frame_use)
        class_names_person = person_model.names

        for result in results_person:
            for box in result.boxes:
                class_id = int(box.cls)
                class_name = class_names_person[class_id]

                if class_name == 'motorcycle':
                    # Map box coordinates back to the original frame
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    # x1 += roi_x1
                    # x2 += roi_x1
                    # y1 += roi_y1
                    # y2 += roi_y1

                    motorcycle_roi = frame[y1 - 100:y2, x1:x2]
                    motor_roi = frame[y1:y2 + 50, x1:x2]

                    helmet_results = helmet_model(motorcycle_roi)
                    for helmet_result in helmet_results:
                        for helmet_box in helmet_result.boxes:
                            hx1, hy1, hx2, hy2 = map(int, helmet_box.xyxy[0])
                            confidence = helmet_box.conf[0]
                            helmet_class_id = int(helmet_box.cls)
                            helmet_class_name = helmet_model.names[helmet_class_id]

                            if helmet_class_name in target_classes_helmet:
                                color = target_classes_helmet[helmet_class_name]
                                label = f"{helmet_class_name} {confidence:.2f}"

                                if helmet_class_name == 'Without Helmet':
                                    roi_path = "motor_roi.jpg"
                                    success = cv2.imwrite(roi_path, motor_roi)
                                    if not success:
                                        st.error("Error: Could not write the image to disk.")
                                        continue
                                    if k == 0:
                                        # response = api.license_plate_ocr(roi_path).json()
                                        # st.write(response)

                                        k += 1

                                absolute_hx1 = x1 + hx1
                                absolute_hy1 = (y1 - 100) + hy1
                                absolute_hx2 = x1 + hx2
                                absolute_hy2 = (y1 - 100) + hy2

                                cv2.rectangle(
                                    frame,
                                    (absolute_hx1, absolute_hy1),
                                    (absolute_hx2, absolute_hy2),
                                    color,
                                    2
                                )
                                cv2.putText(
                                    frame,
                                    label,
                                    (absolute_hx1, absolute_hy1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    color,
                                    2,
                                )

                    # Draw bounding box for the motorcycle
                    color = target_classes_person['motorcycle']
                    label = f"{class_name} {box.conf[0]:.2f}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Convert frame to RGB for display in Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Display the frame
        frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

    # Release resources
    cap.release()
    os.remove(temp_video_path)
else:
    st.info("Please upload a video file to start.")
