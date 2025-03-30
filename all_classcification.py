import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import os
import zipfile
import argparse
import time

from PIL import Image, UnidentifiedImageError
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, ConcatDataset, Dataset
from torchvision.models import resnet50, convnext_tiny


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
# [ 인자 파서 ]
# -------------------------------------------------------
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["resnet50", "convnext_tiny"], default="resnet50",
                        help="사용할 모델을 선택합니다.")
    parser.add_argument("--mode", choices=["train", "test"], default="train",
                        help="학습(train) 또는 테스트(test) 모드를 선택합니다.")
    parser.add_argument("--model-path", type=str, default="combined_model.pth",
                        help="모델 파라미터를 저장하거나 불러올 경로입니다.")
    args = parser.parse_args()
    return args


# ----------------------
# [ CUDA 장치 설정 ]
# ----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ----------------------------------------------------------
# [ RemappedDataset: 데이터셋의 라벨에 오프셋(offset) 적용 ]
# ----------------------------------------------------------
class RemappedDataset(Dataset):
    def __init__(self, dataset, label_offset=0):
        self.dataset = dataset
        self.label_offset = label_offset

    def __getitem__(self, index):
        img, label = self.dataset[index]
        return img, label + self.label_offset

    def __len__(self):
        return len(self.dataset)


# ----------------------------------------------------------
# [ Palette 이미지(투명 채널) 처리 함수 ]
# ----------------------------------------------------------
def convert_to_rgba(image):
    if image.mode == 'P' and 'transparency' in image.info:
        return image.convert('RGBA')
    return image


# ----------------------------------------------------------
# [ 데이터 전처리 정의 ]
# ----------------------------------------------------------
def get_transforms(train=True):
    if train:
        transforms.Compose([
            transforms.Lambda(convert_to_rgba),
            transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),  # 색상 증강
            transforms.RandomRotation(15),  # 회전 증강
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        return transforms.Compose([
            transforms.Lambda(convert_to_rgba),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Lambda(convert_to_rgba),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

# ----------------------------------------------------------
# [ get_dataset_configs: datasets 폴더 내의 하위 폴더(단, use 폴더 제외)에서 zip 파일 추출 및 구성 ]
# ----------------------------------------------------------
@timeit("Dataset Configurations")
def get_dataset_configs():
    base_dir = "./datasets"
    use_dir = os.path.join(base_dir, "use")
    # use 폴더가 없으면 생성
    if not os.path.exists(use_dir):
        os.makedirs(use_dir)

    dataset_configs = []
    for folder_name in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder_name)
        if os.path.isdir(folder_path) and folder_name != "use":
            # 각 폴더 내의 zip 파일 검색 (하나만 있다고 가정)
            zip_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".zip")]
            if len(zip_files) != 1:
                print(f"Warning: {folder_path}에 zip 파일이 1개가 아님 (찾은 갯수: {len(zip_files)})")
                continue
            zip_file_path = os.path.join(folder_path, zip_files[0])
            # 추출 폴더는 use 폴더 내에 동일한 이름으로 설정
            extract_path = os.path.join(use_dir, folder_name)
            if not os.path.exists(extract_path) or not os.path.isdir(extract_path):
                print(f"Extracting {zip_file_path} to {extract_path}")
                with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_path)
            else:
                print(f"Directory {extract_path} already exists, skipping extraction.")
            # train과 validation 폴더가 추출된 디렉토리 내에 있다고 가정
            dataset_configs.append({
                "name": folder_name,
                "train_dir": os.path.join(extract_path, "train"),
                "val_dir": os.path.join(extract_path, "validation")
            })
    return dataset_configs


# ---------------
# [ 메인 함수 ]
# ---------------
def main():
    args = parse_arguments()

    # 하이퍼파라미터 설정
    learning_rate = 1e-4
    num_epochs = 20
    batch_size = 128
    num_workers = 16
    weight_decay = 0.01

    train_transforms = get_transforms(train=True)
    val_transforms = get_transforms(train=False)

    # ----------------------------
    # [ 데이터셋 구성: datasets 폴더 내의 모든 zip 파일 추출 ]
    # ----------------------------
    dataset_configs = get_dataset_configs()
    if not dataset_configs:
        raise RuntimeError("사용할 데이터셋 구성이 없습니다.")

    # ----------------------------------------------------------
    # [ 각 데이터셋 로드 및 라벨 오프셋 적용 ]
    # ----------------------------------------------------------
    train_datasets = []
    val_datasets = []
    combined_class_names = []
    label_offset = 0

    def check_images(folder):
        bad_files = []
        for root, dirs, files in os.walk(folder):
            for f in files:
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                    path = os.path.join(root, f)
                    try:
                        with Image.open(path) as img:
                            img.verify()  # 이미지 무결성 검사
                    except (OSError, UnidentifiedImageError):
                        bad_files.append(path)
        return bad_files

    for config in dataset_configs:
        # ImageFolder로 각 데이터셋 로드
        # 각 데이터셋의 train 및 validation 경로 설정 시 validation 폴더 존재 여부 체크
        train_dir = config["train_dir"]
        val_dir = config["val_dir"]
        if not os.path.exists(val_dir):
            print(f"Warning: {val_dir} not found. Using training folder as validation for {config['name']}.")
            val_dir = train_dir  # validation 폴더가 없으면 train 폴더로 대체

        train_ds = ImageFolder(train_dir, transform=train_transforms)
        val_ds = ImageFolder(val_dir, transform=val_transforms)
        # print(check_images(train_dir))
        # print(check_images(val_dir))
        classes = train_ds.classes  # 각 데이터셋의 클래스 (알파벳 순 정렬)
        classes = [str(config['name'])+str(c) for c in classes]
        print(f"{config['name']} 클래스:", classes)
        combined_class_names.extend(classes)

        # 라벨 오프셋 적용
        train_ds_remap = RemappedDataset(train_ds, label_offset=label_offset)
        val_ds_remap = RemappedDataset(val_ds, label_offset=label_offset)
        train_datasets.append(train_ds_remap)
        val_datasets.append(val_ds_remap)
        label_offset += len(classes)

    num_classes = len(combined_class_names)
    print("전체 클래스 수:", num_classes)
    print("통합 클래스 목록:", combined_class_names)

    # ----------------------------------------------------------
    # [ 통합 데이터셋 구성 ]
    # ----------------------------------------------------------
    combined_train_dataset = ConcatDataset(train_datasets)
    combined_val_dataset = ConcatDataset(val_datasets)

    train_loader = DataLoader(combined_train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(combined_val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    # ----------------------------------------------------------
    # [ 모델 준비: 선택한 아키텍처에 맞춰 마지막 분류층 수정 ]
    # ----------------------------------------------------------
    if args.model == "resnet50":
        model = resnet50(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:  # convnext_tiny
        model = convnext_tiny(pretrained=True)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
    model = model.to(device)

    # ----------------------------------------------------------
    # [ 학습 모드 ]
    # ----------------------------------------------------------
    # Mixup 함수
    def mixup(images, labels, alpha=1.0):
        lambda_ = torch.distributions.beta.Beta(alpha, alpha).sample().to(images.device)
        batch_size = images.size(0)
        index = torch.randperm(batch_size).to(images.device)
        mixed_images = lambda_ * images + (1 - lambda_) * images[index]
        return mixed_images, (labels, labels[index], lambda_)

    if args.mode == "train":
        print(">> Training mode <<")
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        @timeit("Training")
        def train_model():
            for epoch in range(num_epochs):
                model.train()
                running_loss = 0.0
                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    mixed_inputs, (labels_a, labels_b, lambda_) = mixup(inputs, labels, alpha=1.0)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                    #print("Training:",inputs, labels, loss)
                avg_loss = running_loss / len(train_loader)
                print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")
                print("Saving model to after Epoch:", "_".join((str(epoch + 1),args.model_path)))
                torch.save(model.state_dict(), "_".join((str(epoch + 1),args.model_path)))

        @timeit("Validation")
        def validate_model():
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (preds == labels).sum().item()
            acc = 100 * correct / total
            print(f"Validation Accuracy: {acc:.2f}%")

        train_model()

        print("Saving model to after training:", args.model_path)
        torch.save(model.state_dict(), args.model_path)
        validate_model()

        print("Saving model to after validate:", args.model_path)
        torch.save(model.state_dict(), args.model_path)

    # ----------------------------------------------------------
    # [ 테스트 모드 ]
    # ----------------------------------------------------------
    else:
        print(">> Test (Inference) mode <<")
        if not os.path.exists(args.model_path):
            raise FileNotFoundError(f"Model file not found: {args.model_path}")
        model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
        model.eval()

        test_transforms = get_transforms(train=False)

        @timeit("Test Image Preprocessing")
        def load_test_image(path):
            if not os.path.exists(path):
                raise FileNotFoundError(f"{path} not found!")
            image = Image.open(path).convert("RGB")
            test_input = test_transforms(image).unsqueeze(0)
            return test_input.to(device)

        @timeit("PyTorch Inference")
        def pytorch_inference(model, test_input):
            with torch.no_grad():
                outputs = model(test_input)
                pred_idx = outputs.argmax(dim=1).item()
            return pred_idx

        test_image_path = "test_image.jpg"  # 실제 테스트 이미지 경로로 수정
        test_input = load_test_image(test_image_path)
        pred_idx = pytorch_inference(model, test_input)
        predicted_class = combined_class_names[pred_idx]
        print(f"Test Image: {test_image_path}")
        print(f"Predicted Index: {pred_idx}")
        print(f"Predicted Class: {predicted_class}")


if __name__ == "__main__":
    main()
