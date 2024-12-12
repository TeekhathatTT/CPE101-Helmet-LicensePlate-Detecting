import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np

# Initialize YOLO models
person_model = YOLO('motor.pt')
helmet_model = YOLO('helmet.pt')

target_classes_person = {'person': (0, 255, 0), 'motorcycle': (255, 0, 0)}
target_classes_helmet = {'With Helmet': (0, 255, 0), 'Without Helmet': (0, 0, 255)}

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
    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Error: Can't receive frame")
            break

        # Perform YOLO inference
        results_person = person_model(frame)
        class_names_person = person_model.names

        for result in results_person:
            for box in result.boxes:
                class_id = int(box.cls)
                class_name = class_names_person[class_id]

                if class_name == 'person':
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    person_roi = frame[y1:y2, x1:x2]

                    # Helmet detection
                    helmet_results = helmet_model(person_roi)
                    for helmet_result in helmet_results:
                        for helmet_box in helmet_result.boxes:
                            hx1, hy1, hx2, hy2 = map(int, helmet_box.xyxy[0])
                            confidence = helmet_box.conf[0]
                            helmet_class_id = int(helmet_box.cls)
                            helmet_class_name = helmet_model.names[helmet_class_id]

                            if helmet_class_name in target_classes_helmet:
                                color = target_classes_helmet[helmet_class_name]
                                label = f"{helmet_class_name} {confidence:.2f}"

                                cv2.rectangle(frame, (x1 + hx1, y1 + hy1), (x1 + hx2, y1 + hy2), color, 2)
                                cv2.putText(
                                    frame,
                                    label,
                                    (x1 + hx1, y1 + hy1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    color,
                                    2,
                                )

                    # Draw bounding box for the person
                    color = target_classes_person['person']
                    label = f"{class_name} {box.conf[0]:.2f}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Convert the frame to RGB format for Streamlit
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Display the frame in Streamlit
        frame_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)

        # Break the loop if the "Stop Camera" button is pressed
        if stop:
            break

# Release resources
if cap:
    cap.release()
cv2.destroyAllWindows()
