from ultralytics import YOLO
import torch

torch.cuda.empty_cache()  # GPU 메모리 캐시 비우기

if __name__ == '__main__':
    # 사전 학습된 YOLO12n 모델 로드 (기존 학습된 best.pt 사용 가능)
    model = YOLO("C:/Users/DONGIN/PycharmProjects/yolov12-training/runs/train/train3/weights/best.pt")

    # 모델 학습
    results = model.train(
        data="yoloV12/data.yaml",  # 데이터셋 설정 파일
        epochs=100,  # 학습 에포크
        imgsz=640,  # 이미지 크기 (1024 → 640으로 수정하여 안정적인 학습 가능)
        batch=4,  # 자동 설정 대신 명확한 배치 사이즈 지정
        project="runs/train",  # 프로젝트 경로
        cache="disk",  # 데이터 캐시 옵션
        device=0  # GPU 0번 사용
    )
