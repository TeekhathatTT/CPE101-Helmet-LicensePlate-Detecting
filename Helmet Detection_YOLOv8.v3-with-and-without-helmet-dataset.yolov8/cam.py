from ultralytics import YOLO
import cv2

# Load the trained YOLOv8 model
model = YOLO('D:/Helmet Detection_YOLOv8.v3-with-and-without-helmet-dataset.yolov8/runs/detect/train/weights/best.pt')  # Path to your trained model

# Initialize the camera (use 0 for the default camera, or change it for external ones)
cap = cv2.VideoCapture(0)

# Check if the camera is opened successfully
if not cap.isOpened():
    print("Error: Cannot open camera")
    exit()

while True:
    # Read a frame from the camera
    ret, frame = cap.read()
    if not ret:
        print("Error: Can't receive frame")
        break

    # Run YOLOv8 inference on the frame
    results = model(frame)

    # Visualize results directly on the frame
    annotated_frame = results[0].plot()

    # Display the annotated frame
    cv2.imshow("YOLOv8 Live Inference", annotated_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
