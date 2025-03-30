import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import os
import zipfile
import argparse
import time

from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
# ---------------------------
# [ 속도 측정용 데코레이터 ]
# ---------------------------
def timeit(label):
    def decorator(func):
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            print(f"{label}: {end - start:.4f} seconds")
            return result
        return wrapper
    return decorator

# -------------------------------------------------------
# [ 학습(Train) 모드와 추론(Test) 모드 분리를 위한 파서 ]
# -------------------------------------------------------
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "test"], required=True,
                        help="Choose between 'train' or 'test' mode.")
    parser.add_argument("--model-path", type=str, default="resnet50_fruits.pth",
                        help="Path to save or load the model weights.")
    args = parser.parse_args()
    return args

# ----------------------
# [ CUDA 장치 설정 ]
# ----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------------
# [ ZIP 해제 함수 예시 ]
# ---------------------
@timeit("ZIP Extraction")
def extract_dataset():
    fruit_and_vegetable_zip_path = "resnet_convnext/datasets/kritikseth_fruit-and-vegetable-image-recognition/fruit-and-vegetable-image-recognition.zip"
    fruit_and_vegetable_extract_path = "resnet_convnext/datasets/use/kritikseth_fruit-and-vegetable-image-recognition"
    if not os.path.exists(fruit_and_vegetable_extract_path) or not os.path.isdir(fruit_and_vegetable_extract_path):
        with zipfile.ZipFile(fruit_and_vegetable_zip_path, 'r') as zip_ref:
            zip_ref.extractall(fruit_and_vegetable_extract_path)
        print("Dataset extracted!")
    else:
        print(f"Directory {fruit_and_vegetable_extract_path} already exists, skipping extraction.")
    return fruit_and_vegetable_extract_path

# ----------------------------------------------------------
# [ Palette 이미지(투명 채널) 처리 함수 : Lambda 변환에서 사용 ]
# ----------------------------------------------------------
def convert_to_rgba(image):
    if image.mode == 'P' and 'transparency' in image.info:
        return image.convert('RGBA')
    return image

# ---------------
# [ 메인 함수 ]
# ---------------
def main():
    args = parse_arguments()

    # -------------------------
    # [ train / test 분기 ]
    # -------------------------
    if args.mode == "train":
        print(">> Training mode <<")
        # ---------------------
        # [ 데이터 경로 세팅 ]
        # ---------------------
        extract_path = extract_dataset()  # (ZIP 해제) 데이터셋 경로
        train_dir = os.path.join(extract_path, 'train')
        val_dir   = os.path.join(extract_path, 'validation')

        # -----------------------
        # [ 공통 전처리 객체 ]
        # -----------------------
        train_transforms = transforms.Compose([
            transforms.Lambda(convert_to_rgba),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        val_transforms = transforms.Compose([
            transforms.Lambda(convert_to_rgba),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

        # ------------------------------
        # [ 공통 하이퍼파라미터 설정 ]
        # ------------------------------
        num_classes = 36
        learning_rate = 1e-4
        num_epochs = 10
        batch_size = 64
        num_workers = 4

        # -------------------------------
        # [ 모델 준비: ResNet50 예시 ]
        # -------------------------------
        model = convnext_tiny(pretrained=True)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
        model = model.to(device)

        # -------------------------------
        # [ 손실 함수 & 옵티마이저 설정 ]
        # -------------------------------
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # -------------------------
        # [ DataLoader 준비 ]
        # -------------------------
        train_dataset = ImageFolder(train_dir, transform=train_transforms)
        val_dataset   = ImageFolder(val_dir,   transform=val_transforms)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=True)
        val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False,
                                  num_workers=num_workers, pin_memory=True)

        # -------------------------
        # [ 학습 루프 함수 ]
        # -------------------------
        @timeit("Training")
        def train_model():
            for epoch in range(num_epochs):
                model.train()
                running_loss = 0.0
                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()

                avg_loss = running_loss / len(train_loader)
                print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}")

        # -------------------------
        # [ 검증 루프 함수 ]
        # -------------------------
        @timeit("Validation")
        def validate_model():
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            accuracy = 100 * correct / total
            print(f"Validation Accuracy: {accuracy:.2f}%")

        # -------------------------
        # [ 학습 & 검증 실행 ]
        # -------------------------
        train_model()
        validate_model()

        # -------------------------
        # [ 모델 저장 ]
        # -------------------------
        print("Saving model to:", args.model_path)
        torch.save(model.state_dict(), args.model_path)

    else:
        print(">> Test (Inference) mode <<")

        # 추론만 할 때는 굳이 학습용 데이터셋/로더를 만들 필요가 없습니다.
        # 만약 검증셋 같은 걸 쓰고 싶다면, 아래에서 추가하시면 됩니다.

        # -------------------------
        # [ 모델 로드 & 평가모드 ]
        # -------------------------
        if not os.path.exists(args.model_path):
            raise FileNotFoundError(f"Model file not found: {args.model_path}")

        model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
        model.eval()

        # -------------------------
        # [ 간단한 테스트 이미지 추론 예시 ]
        # -------------------------
        # (사용자가 테스트할 이미지를 지정)
        test_image_path = "test_image.jpg"  # 예시
        if not os.path.exists(test_image_path):
            raise FileNotFoundError(f"{test_image_path} not found!")

        # 추론 전처리 (val_transforms 등 사용 가능)
        test_transforms = transforms.Compose([
            transforms.Lambda(convert_to_rgba),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

        # -------------------------
        # [ 테스트 이미지 로드 ]
        # -------------------------
        @timeit("Test Image Preprocessing")
        def load_test_image(path):
            image = Image.open(path).convert("RGB")
            test_input = test_transforms(image).unsqueeze(0)  # shape: [1, 3, 224, 224]
            test_input = test_input.to(device)
            return test_input

        # -------------------------
        # [ 추론 함수 ]
        # -------------------------
        @timeit("PyTorch Inference")
        def pytorch_inference(model, test_input):
            with torch.no_grad():
                outputs = model(test_input)
                predicted_class = outputs.argmax(dim=1).item()
            return predicted_class

        # -------------------------
        # [ 추론 실행 ]
        # -------------------------
        test_input = load_test_image(test_image_path)
        pred_idx = pytorch_inference(model, test_input)

        # ----------------------------------------------------
        #  만약 인덱스를 폴더명(클래스명)으로 매핑하려면?
        #  - ImageFolder 생성 후에 dataset.classes를 가져와야 함.
        #  - test만 따로 한다면, 아래와 같이 처리할 수 있음:
        # ----------------------------------------------------
        # 굳이 train_dir, val_dir 중 아무 폴더나 사용해
        # 실제 클래스 목록만 가져오면 됩니다.
        dummy_dataset = ImageFolder(train_dir)  # transform은 없어도 됨
        class_names = dummy_dataset.classes     # 알파벳 순 정렬
        predicted_class_name = class_names[pred_idx]

        print(f"Test Image Path : {test_image_path}")
        print(f"Predicted Index : {pred_idx}")
        print(f"Predicted Class : {predicted_class_name}")


# -------------------------------
# [ 스크립트 실행 진입점 ]
# -------------------------------
if __name__ == "__main__":
    main()
