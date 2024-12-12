from ultralytics import YOLO

# Load the YOLOv8 model (you can choose 'yolov8n.pt' for nano, 'yolov8s.pt' for small, etc.)
model = YOLO('yolov8n.pt')  # Replace 'yolov8n.pt' with other versions as needed

# Train the model
model.train(data='data.yaml', epochs=50, imgsz=640)
