import os
import cv2
import numpy as np
from ultralytics import YOLO
from segment_anything import SamPredictor, sam_model_registry

# 1. 경로 설정
input_folder = "selenium_images/_banana"  # 입력 이미지 폴더
output_folder = "yolo_sam_result"  # 결과 저장 폴더

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 2. YOLOv8 모델 로드
yolo_model = YOLO("yolov8n.pt")

# 3. SAM 모델 로드
sam_checkpoint = "sam_vit_b_01ec64.pth"  # SAM 가중치 경로
model_type = "vit_b"  # 모델 타입 (vit_b, vit_l, vit_h 중 선택)
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
predictor = SamPredictor(sam)

# 4. 이미지 파일 목록 가져오기
image_files = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]

# 5. 각 이미지 처리
for image_file in image_files:
    image_path = os.path.join(input_folder, image_file)

    # 이미지 읽기
    img = cv2.imread(image_path)
    if img is None:
        print(f"이미지를 불러올 수 없습니다: {image_file}")
        continue

    # YOLOv8으로 객체 탐지
    results = yolo_model(img)

    # SAM에 이미지 설정
    predictor.set_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # 결과 이미지 복사본
    output_img = img.copy()

    # 바나나 클래스(46번)만 처리
    for box in results[0].boxes:
        if int(box.cls) == 46:  # "banana" 클래스
            # 바운딩 박스 좌표 추출
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            input_box = np.array([x1, y1, x2, y2])

            # SAM으로 세그먼테이션
            masks, scores, _ = predictor.predict(
                box=input_box,
                multimask_output=False  # 단일 마스크만 반환
            )

            # 가장 높은 점수의 마스크 선택
            mask = masks[0]

            # 마스크를 이미지에 적용 (예: 빨간색으로 윤곽선 그리기)
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(output_img, contours, -1, (0, 0, 255), 2)

    # 결과 저장
    output_path = os.path.join(output_folder, f"sam_{image_file}")
    cv2.imwrite(output_path, output_img)
    print(f"후처리 완료: {output_path}")

print("모든 이미지 처리가 완료되었습니다!")