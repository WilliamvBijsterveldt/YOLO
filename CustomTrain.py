import os
from ultralytics import YOLO

def train_model():
    model = YOLO("yolov8n.pt")
    model.train(data="C:\\Users\Daan Hovens\OneDrive - Office 365 Fontys\Mechatronica Fontys\Minor AR\CHAL2\YOLO\datasets\data.yaml",
                epochs=50, imgsz=640, batch=16, device="cuda")

if __name__ == '__main__':
    train_model()
