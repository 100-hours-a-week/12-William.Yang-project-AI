import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets, models
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def timer_decorator(func):
    """함수 실행 시간을 측정하는 데코레이터"""

    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} 실행 시간: {end_time - start_time:.2f}초")
        return result

    return wrapper


def plot_training_metrics(train_losses, val_accuracies, dataset_name, prefix=''):
    """학습 손실 및 검증 정확도 그래프 저장"""
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.title(f'{dataset_name} - Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Validation Accuracy', color='green')
    plt.title(f'{dataset_name} - Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    os.makedirs('./results', exist_ok=True)
    plt.savefig(f'./results/{prefix}{dataset_name}_training_metrics.png')
    plt.close()


def plot_confusion_matrix(true_labels, pred_labels, classes, dataset_name, prefix=''):
    """혼동 행렬 시각화 및 저장"""
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title(f'{dataset_name} - Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    os.makedirs('./results', exist_ok=True)
    plt.tight_layout()
    plt.savefig(f'./results/{prefix}{dataset_name}_confusion_matrix.png')
    plt.close()

# ----------------------------------------------------------
# [ Palette 이미지(투명 채널) 처리 함수 ]
# ----------------------------------------------------------
def convert_to_rgba(image):
    if image.mode == 'P' and 'transparency' in image.info:
        return image.convert('RGBA')
    return image

def get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.Lambda(convert_to_rgba),
            transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Lambda(convert_to_rgba),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


@timer_decorator
def train_dataset_classifier(device):
    """데이터셋 단위 분류기 학습"""
    # 하이퍼파라미터 설정
    learning_rate = 1e-4
    num_epochs = 20
    batch_size = 128
    num_workers = 16
    weight_decay = 0.01

    # 데이터셋 리스트 (수동으로 입력 필요)
    datasets_list = ["cookiefinder_tomato-disease-multiple-sources",
                     "hafiznouman786_potato-plant-diseases-data",
                     "imsparsh_flowers-dataset",
                     "jay7080dev_rice-plant-diseases-dataset",
                     "jonathansilva2020_orange-diseases-dataset",
                     "kritikseth_fruit-and-vegetable-image-recognition",
                     "mahyeks_almond-varieties",
                     "misrakahmed_vegetable-image-dataset",
                     "muratkokludataset_pistachio-image-dataset",
                     "muratkokludataset_rice-image-dataset"]
    dataset_models = {}

    for dataset_name in datasets_list:
        print(f"\n{dataset_name} 데이터셋 분류기 학습 시작")

        # 데이터셋 로드
        dataset_path = f'./datasets/use/{dataset_name}/train'
        dataset_classes = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
        num_dataset_classes = len(dataset_classes)

        train_dataset = datasets.ImageFolder(
            root=dataset_path,
            transform=get_transforms(train=True)
        )

        # 학습/검증 데이터셋 분할
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        # 모델 설정
        model = models.resnet50(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_dataset_classes)
        model = model.to(device)

        # 손실 함수 및 옵티마이저
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # 학습률 스케줄러
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

        # 학습 모니터링 변수
        train_losses = []
        val_accuracies = []
        true_labels = []
        pred_labels = []

        # 학습 루프
        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0.0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            # 검증
            model.eval()
            correct = 0
            total = 0
            val_loss = 0.0

            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                    # 혼동 행렬용 라벨 수집
                    true_labels.extend(labels.cpu().numpy())
                    pred_labels.extend(predicted.cpu().numpy())

            # 메트릭 계산
            val_accuracy = 100 * correct / total
            avg_train_loss = epoch_loss / len(train_loader)

            train_losses.append(avg_train_loss)
            val_accuracies.append(val_accuracy)

            print(f'Epoch [{epoch + 1}/{num_epochs}], '
                  f'Train Loss: {avg_train_loss:.4f}, '
                  f'Val Accuracy: {val_accuracy:.2f}%')

            scheduler.step()

        # 그래프 저장
        plot_training_metrics(train_losses, val_accuracies, dataset_name, prefix='dataset_')
        plot_confusion_matrix(
            true_labels,
            pred_labels,
            dataset_classes,
            dataset_name,
            prefix='dataset_'
        )

        # 모델 저장
        os.makedirs('./weights', exist_ok=True)
        torch.save(model.state_dict(), f'./weights/{dataset_name}_classifier.pth')

        # 데이터셋별 모델 저장
        dataset_models[dataset_name] = model

    return dataset_models


@timer_decorator
def train_class_classifiers(dataset_models, device):
    """데이터셋 내 클래스 분류기 학습"""
    # 하이퍼파라미터 설정
    learning_rate = 1e-4
    num_epochs = 20
    batch_size = 128
    num_workers = 16
    weight_decay = 0.01

    for dataset_name, dataset_model in dataset_models.items():
        print(f"\n{dataset_name} 내 클래스 분류기 학습 시작")

        dataset_path = f'./datasets/use/{dataset_name}/train'
        class_folders = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]

        for class_name in class_folders:
            print(f"  {class_name} 클래스 학습 중")

            # 해당 클래스 데이터셋 로드
            class_dataset_path = os.path.join(dataset_path, class_name)
            train_dataset = datasets.ImageFolder(
                root=dataset_path,
                transform=get_transforms(train=True)
            )

            # 학습/검증 데이터셋 분할
            train_size = int(0.8 * len(train_dataset))
            val_size = len(train_dataset) - train_size
            train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

            train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
            val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

            # 모델 복제 및 최종 레이어 재설정
            model = models.resnet50(pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, len(class_folders))
            model.load_state_dict(dataset_models[dataset_name].state_dict())
            model = model.to(device)

            # 손실 함수 및 옵티마이저
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.AdamW(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )

            # 학습률 스케줄러
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

            # 학습 모니터링 변수
            train_losses = []
            val_accuracies = []
            true_labels = []
            pred_labels = []

            # 학습 루프
            for epoch in range(num_epochs):
                model.train()
                epoch_loss = 0.0

                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(device), labels.to(device)

                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()

                # 검증
                model.eval()
                correct = 0
                total = 0
                val_loss = 0.0

                with torch.no_grad():
                    for inputs, labels in val_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        val_loss += loss.item()

                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()

                        # 혼동 행렬용 라벨 수집
                        true_labels.extend(labels.cpu().numpy())
                        pred_labels.extend(predicted.cpu().numpy())

                # 메트릭 계산
                val_accuracy = 100 * correct / total
                avg_train_loss = epoch_loss / len(train_loader)

                train_losses.append(avg_train_loss)
                val_accuracies.append(val_accuracy)

                print(f'    Epoch [{epoch + 1}/{num_epochs}], '
                      f'Train Loss: {avg_train_loss:.4f}, '
                      f'Val Accuracy: {val_accuracy:.2f}%')

                scheduler.step()

            # 그래프 저장
            plot_training_metrics(train_losses, val_accuracies, f'{dataset_name}_{class_name}', prefix='class_')
            plot_confusion_matrix(
                true_labels,
                pred_labels,
                class_folders,
                f'{dataset_name}_{class_name}',
                prefix='class_'
            )

            # 클래스별 모델 저장
            os.makedirs('./weights/class_models', exist_ok=True)
            torch.save(
                model.state_dict(),
                f'./weights/class_models/{dataset_name}_{class_name}_classifier.pth'
            )


def main():
    # CUDA 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"학습에 사용될 장치: {device}")

    # 데이터셋 단위 분류기 학습
    dataset_models = train_dataset_classifier(device)

    # 클래스별 분류기 학습
    train_class_classifiers(dataset_models, device)


if __name__ == "__main__":
    main()
