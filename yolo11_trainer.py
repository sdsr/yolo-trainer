from ultralytics import YOLO
import torch

torch.cuda.empty_cache()

if __name__ == '__main__':
    # COCO 사전 학습된 YOLOv11n 모델 로드
    model = YOLO("C:/Users/DONGIN/PycharmProjects/"
                 "yolov11-training/runs/train/train3/weights/best.pt")

    # 모델 학습
    results = model.train(data="yoloV11/data.yaml", epochs=100, imgsz=1024,
                          batch=-1, project="runs/train", cache="disk", device=0)