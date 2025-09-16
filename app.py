import torch
import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLO model
model = YOLO("runs/detect/train11/weights/best.pt")

# Load MiDaS model
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas.to("cpu").eval()
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO Detection
    results = model(frame)
    boxes = results[0].boxes

    # MiDaS Depth Map
    input_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_tensor = midas_transforms(input_image).to("cpu")

    with torch.no_grad():
        prediction = midas(input_tensor)
        depth_map = prediction.squeeze().cpu().numpy()
        depth_map_resized = cv2.resize(depth_map, (frame.shape[1], frame.shape[0]))  # ✅ resize depth map

    # Loop over detected objects
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        # Get approximate depth value from resized depth map
        z = depth_map_resized[cy, cx]

        # Annotate frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
        cv2.putText(frame, f"({cx},{cy},{z:.2f})", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        print(f"📱 3D Position: X={cx}, Y={cy}, Z={z:.2f}")

    # Show result
    cv2.imshow("3D Mobile Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
