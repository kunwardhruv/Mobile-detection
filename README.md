# Mobile Phone Detection

## Project Description
**Mobile Phone Detection** is a computer vision project that detects mobile phones in images or real-time video streams. This project uses object detection techniques to accurately identify and highlight mobile phones, making it useful for surveillance, classroom monitoring, or automated systems.

---

## Features
- Real-time mobile phone detection via webcam.
- Detects multiple mobile phones in the frame.
- Can process individual images or a batch of images.
- Highlights detected phones with bounding boxes.
- Optional: Saves detection results for further analysis.

---

## Tech Stack
- **Language:** Python  
- **Libraries:** OpenCV, YOLOv8 (PyTorch)  
- **Model:** Pretrained YOLOv8 (yolov8n.pt)

---

## Installation
1. Clone the repository:
```bash
git clone https://github.com/kunwardhruv/Mobile-detection.git
cd Mobile-detection


Mobile-detection/
├─ valid/labels/          # Dataset labels
├─ yolov8n.pt             # Pretrained YOLOv8 model
├─ detect.py              # Detection script
├─ README.md              # Project documentation
├─ requirements.txt       # Python dependencies
