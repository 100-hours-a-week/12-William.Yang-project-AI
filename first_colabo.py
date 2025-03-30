import os
import cv2
import numpy as np
from ultralytics import SAM, YOLO


# 2. wget으로 이미지 다운로드
# file_name = "2025-ford-mustang-60th-anniversary-exterior.jpg"
file_name = "1190_tomato_with_Noise.jpg"


# 3. 이미지 읽기 및 크기 조정 (imgsz=640)
img = cv2.imread(file_name)
if img is None:
    print("이미지를 읽어오지 못했습니다.")
else:
    h, w = img.shape[:2]
    if h > w:
        new_h = 640
        new_w = int(w * (640 / h))
    else:
        new_w = 640
        new_h = int(h * (640 / w))
    resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    resized_file_path = os.path.join("./", "resized_image.jpg")
    cv2.imwrite(resized_file_path, resized_img)
    print("리사이즈된 이미지 크기:", resized_img.shape)

    # 4. YOLO로 bbox 추출
    yolo_model = YOLO("yolo12n.pt")
    yolo_results = yolo_model(resized_file_path)
    if yolo_results[0].boxes is not None:
        bbox = yolo_results[0].boxes.xyxy[0].cpu().numpy().tolist()
        print("YOLO로 추출된 bbox:", bbox)
    else:
        print("YOLO로 객체를 탐지하지 못했습니다. 기본 bbox 사용.")
        bbox = [new_w // 4, new_h // 4, 3 * new_w // 4, 3 * new_h // 4]

    # 5. SAM 모델 로드 및 추론
    model = SAM("sam2_b.pt")  # 더 큰 모델 사용
    print("SAM2 모델이 로드되었습니다.")
    model.info()

    # 포인트 추가
    points = [[new_w // 2, new_h // 2]]
    labels = [1]
    results = model(resized_file_path, bboxes=[bbox], points=points, labels=labels)
    print("추론이 완료되었습니다.")

    # 6. 결과 처리
    if results and results[0].masks is not None:
        mask = results[0].masks.data[0].cpu().numpy()
        mask = cv2.resize(mask, (new_w, new_h))

        # 마스크 후처리
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        mask = cv2.erode(mask, kernel, iterations=1)

        # 오버레이
        overlay = resized_img.copy()
        overlay[mask > 0] = [0, 255, 0]

        # 결과 저장
        result_path = os.path.join("./", "result_image2.jpg")
        cv2.imwrite(result_path, overlay)
        print(f"결과 이미지가 '{result_path}'에 저장되었습니다.")
    else:
        print("세그멘테이션 결과가 없습니다.")