import cv2
import numpy as np
import torch
from ultralytics import YOLO
from sort import Sort  # SORT tracking algorithm
import csv
import os
from utils import calculate_speed_acceleration

# Load YOLO model
yolo_model = YOLO("models/yolov8n.pt")
tracker = Sort()

# Load video
video_path = "input/InputVideo.mp4"
capture = cv2.VideoCapture(video_path)
frame_rate = 20  # 20 Hz
frame_interval = int(capture.get(cv2.CAP_PROP_FPS) / frame_rate)

output_video_path = "output/result.mp4"
frame_width = int(capture.get(3))
frame_height = int(capture.get(4))
out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 20, (frame_width, frame_height))

# Prepare CSV file
csv_path = "output/trajectory.csv"
os.makedirs("output", exist_ok=True)
with open(csv_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Track ID", "Vehicle Type", "x [m]", "y [m]", "Speed [km/h]", "Tangential Acceleration [ms-2]", "Lateral Acceleration [ms-2]", "Time [s]"])

def process_frame(frame, time_stamp):
    results = yolo_model(frame)
    detections = []
    for r in results:
        for box in r.boxes.xyxy:  # Bounding box format (x1, y1, x2, y2)
            x1, y1, x2, y2 = box[:4].tolist()
            detections.append([x1, y1, x2, y2, 1.0])  # 1.0 as confidence score
    
    # Update tracker
    tracked_objects = tracker.update(np.array(detections))
    extracted_data = []
    for obj in tracked_objects:
        x1, y1, x2, y2, track_id = obj
        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2
        speed, tangential_accel, lateral_accel = calculate_speed_acceleration(track_id, x_center, y_center, time_stamp)
        extracted_data.append([track_id, "Car", x_center, y_center, speed, tangential_accel, lateral_accel, time_stamp])

        # Draw bounding box
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {int(track_id)}", (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return frame, extracted_data

time_stamp = 0
frame_count = 0
while capture.isOpened():
    ret, frame = capture.read()
    if not ret:
        break
    
    if frame_count % frame_interval == 0:
        processed_frame, data = process_frame(frame, time_stamp)
        with open(csv_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(data)
        out.write(processed_frame)
        time_stamp += 1/frame_rate
    
    frame_count += 1

capture.release()
out.release()
cv2.destroyAllWindows()
print("Processing complete. Data saved in output/trajectory.csv and output/result.mp4")
