# from ultralytics import YOLO
# import os
# # Load a COCO-pretrained YOLO12n model
# model = YOLO("yolo11m-cls.pt")
#
# # # Train the model on the COCO8 example dataset for 100 epochs
# # results = model.train(
# #     data="imagenet",
# #     epochs=100,
# #     imgsz=224,
# #     device='cuda' if torch.cuda.is_available() else 'cpu'
# # )
#
# # Run inference with the YOLO12n model on the 'bus.jpg' image
# results = model("kritikseth_fruit-and-vegetable-image-recognitionbanana_to_kritikseth_fruit-and-vegetable-image-recognitionjalepeno_object only banana_2.jpg")
# print("추론이 완료되었습니다.")
#
# # 6. 결과 확인 (옵션)
# # 결과는 results 객체에 저장됨; 필요에 따라 시각화하거나 저장 가능
# results[0].show()  # 첫 번째 결과 이미지 표시 (Colab에서 실행 시)
# # 또는 결과를 저장하려면:
# results[0].save(os.path.join("./", "result_image_banana.jpg"))
# print(results)
# print("결과 이미지가 저장되었습니다.")

import os
from ultralytics import YOLO
import cv2

# 1. 경로 설정
input_folder = "selenium_images/_banana"  # 입력 이미지 폴더
output_folder = "yolo_banana_result"  # 결과 저장 폴더

# 출력 폴더가 없으면 생성
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 2. YOLOv8 모델 로드 (Nano 버전 추천)
model = YOLO("yolov8n.pt")  # 사전 학습된 YOLOv8 Nano 모델 사용

# 3. 이미지 파일 목록 가져오기
image_files = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]

# 4. 각 이미지에 대해 추론 수행
for image_file in image_files:
    # 이미지 경로
    image_path = os.path.join(input_folder, image_file)

    # 이미지 읽기
    img = cv2.imread(image_path)
    if img is None:
        print(f"이미지를 불러올 수 없습니다: {image_file}")
        continue

    # YOLO 모델로 추론
    results = model(img)

    # 결과 이미지에 바운딩 박스 그리기
    annotated_img = results[0].plot()  # 결과에 바운딩 박스와 라벨을 그린 이미지

    # 출력 파일 경로
    output_path = os.path.join(output_folder, f"result_{image_file}")

    # 결과 이미지 저장
    cv2.imwrite(output_path, annotated_img)
    print(f"추론 완료: {output_path}")

print("모든 이미지 처리가 완료되었습니다!")