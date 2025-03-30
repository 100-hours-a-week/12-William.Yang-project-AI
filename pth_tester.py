import warnings
import os
import zipfile
import argparse
import time
import json
import csv
from collections import defaultdict
import numpy as np
from datetime import datetime
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, ConcatDataset, Dataset
from torchvision.models import resnet50, convnext_tiny

warnings.filterwarnings("ignore", category=UserWarning)


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
    parser.add_argument("--model-path", type=str, default=None,
                        help="모델 파라미터를 불러올 경로입니다. 지정하지 않으면 동일 폴더의 {model}_model.pth 파일을 사용합니다.")
    parser.add_argument("--test-dir", type=str, default="selenium_images",
                        help="테스트 이미지가 있는 폴더 경로입니다.")
    parser.add_argument("--output", type=str, default="test_results.json",
                        help="테스트 결과를 저장할 JSON 파일 경로입니다.")
    parser.add_argument("--visualize", action="store_true",
                        help="오분류된 이미지의 시각화 결과를 저장합니다.")
    parser.add_argument("--top-k", type=int, default=5,
                        help="몇 개의 상위 예측을 보여줄지 지정합니다.")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="배치 크기를 지정합니다. 메모리 부족시 줄이세요.")
    parser.add_argument("--confidence-threshold", type=float, default=0.7,
                        help="낮은 신뢰도 예측 감지를 위한 임계값 설정")
    parser.add_argument("--report-dir", type=str, default="test_report",
                        help="테스트 보고서를 저장할 디렉토리")
    args = parser.parse_args()

    # 모델 경로가 지정되지 않으면 기본값 설정
    if args.model_path is None:
        args.model_path = f"{args.model}_model.pth"

    # 보고서 디렉토리 생성
    if not os.path.exists(args.report_dir):
        os.makedirs(args.report_dir)

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
def get_transforms(train=False):
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
    base_dir = "resnet_convnext/datasets"
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


# ----------------------------------------------------------
# [ 이미지 시각화 함수 ]
# ----------------------------------------------------------
def visualize_prediction(image_path, true_class, pred_class, confidence, top_k_classes, top_k_probs, save_path):
    try:
        # 이미지 로드
        img = Image.open(image_path).convert('RGB')
        draw = ImageDraw.Draw(img)

        # 폰트 설정 (기본 폰트 사용)
        try:
            font = ImageFont.truetype("arial.ttf", 20)
            small_font = ImageFont.truetype("arial.ttf", 14)
        except:
            font = ImageFont.load_default()
            small_font = ImageFont.load_default()

        # 텍스트 색상 설정
        text_color = (255, 255, 255)
        box_color = (0, 0, 0, 180)
        correct_color = (0, 255, 0)
        incorrect_color = (255, 0, 0)

        # 결과 표시 박스 준비
        box_width = img.width
        box_height = 30 + len(top_k_classes) * 20
        box_img = Image.new('RGBA', (box_width, box_height), box_color)

        # 결과 이미지 준비
        result_img = Image.new('RGB', (img.width, img.height + box_height), (0, 0, 0))
        result_img.paste(img, (0, 0))
        result_img.paste(box_img, (0, img.height), box_img)

        draw = ImageDraw.Draw(result_img)

        # 예측 결과 정보 표시
        status_color = correct_color if true_class == pred_class else incorrect_color
        draw.text((10, img.height + 5), f"실제: {true_class}", fill=text_color, font=font)
        draw.text((10, img.height + 30), f"예측: {pred_class} ({confidence:.2%})", fill=status_color, font=font)

        # Top-K 예측 결과 표시
        y_offset = img.height + 60
        for i, (cls, prob) in enumerate(zip(top_k_classes, top_k_probs)):
            if i == 0:  # 이미 위에 표시했으므로 생략
                continue
            draw.text((10, y_offset), f"{i + 1}. {cls} ({prob:.2%})", fill=text_color, font=small_font)
            y_offset += 20

        # 결과 저장
        result_img.save(save_path)
        return True
    except Exception as e:
        print(f"시각화 에러: {e}")
        return False


# ----------------------------------------------------------
# [ 테스트 함수 정의 ]
# ----------------------------------------------------------
@timeit("Test Image Loading")
def load_test_images_batch(image_paths, transform):
    images = []
    errors = []
    valid_paths = []

    for path in image_paths:
        try:
            image = Image.open(path).convert("RGB")
            tensor = transform(image)
            images.append(tensor)
            valid_paths.append(path)
        except Exception as e:
            errors.append((path, str(e)))

    if images:
        return torch.stack(images).to(device), valid_paths, errors
    else:
        return None, [], errors


@timeit("PyTorch Batch Inference")
def pytorch_batch_inference(model, batch_inputs, top_k=5):
    with torch.no_grad():
        outputs = model(batch_inputs)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)

        # Top-K 값과 인덱스 가져오기
        top_k_probs, top_k_indices = torch.topk(probabilities, k=top_k, dim=1)

    return top_k_indices.cpu().numpy(), top_k_probs.cpu().numpy()


# ----------------------------------------------------------
# [ 혼동 행렬(Confusion Matrix) 시각화 함수 ]
# ----------------------------------------------------------
def plot_confusion_matrix(y_true, y_pred, class_names, output_path):
    plt.figure(figsize=(12, 10))
    cm = confusion_matrix(y_true, y_pred)

    # 클래스 수가 많으면 이미지 크기 조정
    if len(class_names) > 20:
        plt.figure(figsize=(20, 18))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


# ----------------------------------------------------------
# [ 클래스별 정확도 시각화 함수 ]
# ----------------------------------------------------------
def plot_class_accuracies(class_accuracies, output_path):
    classes = list(class_accuracies.keys())
    accuracies = list(class_accuracies.values())

    # 정확도에 따라 정렬
    sorted_indices = np.argsort(accuracies)
    classes = [classes[i] for i in sorted_indices]
    accuracies = [accuracies[i] for i in sorted_indices]

    plt.figure(figsize=(12, 8))
    colors = plt.cm.RdYlGn(np.array(accuracies))

    # 클래스 수가 많으면 이미지 크기 조정
    if len(classes) > 20:
        plt.figure(figsize=(14, 10))

    plt.barh(classes, accuracies, color=colors)
    plt.xlabel('Accuracy')
    plt.title('Class Accuracies')
    plt.xlim(0, 1.0)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


# ----------------------------------------------------------
# [ 신뢰도 분포 시각화 함수 ]
# ----------------------------------------------------------
def plot_confidence_distribution(confidences, correct_predictions, output_path):
    plt.figure(figsize=(10, 6))

    # 정확/부정확 예측에 따라 신뢰도 분리
    correct_conf = [conf for conf, correct in zip(confidences, correct_predictions) if correct]
    incorrect_conf = [conf for conf, correct in zip(confidences, correct_predictions) if not correct]

    # 히스토그램 생성
    plt.hist([correct_conf, incorrect_conf], bins=20,
             range=(0, 1), alpha=0.7, label=['Correct', 'Incorrect'])

    plt.xlabel('Confidence')
    plt.ylabel('Count')
    plt.title('Confidence Distribution for Correct vs Incorrect Predictions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


# ----------------------------------------------------------
# [ 통합 테스트 함수 ]
# ----------------------------------------------------------
def test_images(model, test_dir, class_names, transform, args):
    results = {
        "summary": {
            "model": args.model,
            "model_path": args.model_path,
            "test_directory": args.test_dir,
            "test_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_images": 0,
            "processed_images": 0,
            "failed_images": 0,
            "correct_predictions": 0,
            "accuracy": 0.0,
            "avg_inference_time": 0.0,
            "avg_confidence": 0.0,
            "avg_confidence_correct": 0.0,
            "avg_confidence_incorrect": 0.0,
            "class_accuracies": {},
            "top_k_accuracy": {},
            "low_confidence_count": 0  # 낮은 신뢰도 예측 수
        },
        "class_results": defaultdict(lambda: {"correct": 0, "total": 0, "images": []}),
        "errors": [],
        "confusion_matrix": {},
        "misclassified_analysis": {},
        "low_confidence_predictions": []  # 낮은 신뢰도 예측 목록
    }

    # 혼동 행렬 및 분류 보고서용 데이터
    y_true = []
    y_pred = []
    all_confidences = []
    all_correct_flags = []
    inference_times = []
    top_k_correct = {k: 0 for k in range(1, args.top_k + 1)}

    # 테스트 디렉토리가 없으면 에러
    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    # 오분류 저장 디렉토리
    misclassified_dir = os.path.join(args.report_dir, "misclassified")
    if args.visualize and not os.path.exists(misclassified_dir):
        os.makedirs(misclassified_dir)

    # 각 클래스별 폴더 순회
    for class_name in os.listdir(test_dir):
        class_dir = os.path.join(test_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        # 해당 클래스 이름이 학습된 클래스 목록에 없으면 스킵
        if class_name not in class_names:
            print(f"Warning: Class '{class_name}' not in trained classes. Skipping.")
            continue

        true_class_idx = class_names.index(class_name)

        # 클래스 폴더 내 이미지 파일 검색
        image_files = [f for f in os.listdir(class_dir)
                       if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]

        results["class_results"][class_name]["total"] = len(image_files)
        results["summary"]["total_images"] += len(image_files)

        # 배치 처리를 위한 경로 목록 생성
        batch_paths = []
        for img_file in image_files:
            batch_paths.append(os.path.join(class_dir, img_file))

        # 배치 단위로 처리
        for i in range(0, len(batch_paths), args.batch_size):
            batch_slice = batch_paths[i:i + args.batch_size]

            # 배치 이미지 로드
            batch_inputs, valid_paths, errors = load_test_images_batch(batch_slice, transform)

            # 에러 기록
            for path, error in errors:
                results["errors"].append({
                    "file": path,
                    "error": error
                })
                results["summary"]["failed_images"] += 1

            if batch_inputs is None or len(valid_paths) == 0:
                continue

            results["summary"]["processed_images"] += len(valid_paths)

            # 배치 추론 시작
            start_time = time.time()
            top_k_indices, top_k_probs = pytorch_batch_inference(model, batch_inputs, args.top_k)
            end_time = time.time()

            batch_inference_time = (end_time - start_time) / len(valid_paths)
            inference_times.append(batch_inference_time)

            # 배치 결과 처리
            for idx, (img_path, indices, probs) in enumerate(zip(valid_paths, top_k_indices, top_k_probs)):
                pred_idx = indices[0]  # 첫 번째가 최상위 예측
                confidence = probs[0]
                predicted_class = class_names[pred_idx]
                is_correct = (pred_idx == true_class_idx)

                # Top-K 정확도 계산
                for k in range(1, args.top_k + 1):
                    if true_class_idx in indices[:k]:
                        top_k_correct[k] += 1

                # 결과 수집
                y_true.append(true_class_idx)
                y_pred.append(pred_idx)
                all_confidences.append(confidence)
                all_correct_flags.append(is_correct)

                # Top-K 클래스 이름과 확률
                top_k_classes = [class_names[idx] for idx in indices]

                # 결과 기록
                img_result = {
                    "file": img_path,
                    "predicted": predicted_class,
                    "confidence": float(confidence),
                    "correct": is_correct,
                    "inference_time": float(batch_inference_time),
                    "top_k_predictions": [
                        {"class": cls, "confidence": float(prob)}
                        for cls, prob in zip(top_k_classes, probs)
                    ]
                }

                results["class_results"][class_name]["images"].append(img_result)

                if is_correct:
                    results["class_results"][class_name]["correct"] += 1
                    results["summary"]["correct_predictions"] += 1
                elif args.visualize:
                    # 오분류된 이미지 시각화 저장
                    base_name = os.path.basename(img_path)
                    save_path = os.path.join(misclassified_dir,
                                             f"{class_name}_to_{predicted_class}_{base_name}")
                    visualize_prediction(img_path, class_name, predicted_class,
                                         confidence, top_k_classes, probs, save_path)

                # 낮은 신뢰도 예측 기록
                if confidence < args.confidence_threshold:
                    results["low_confidence_predictions"].append(img_result)
                    results["summary"]["low_confidence_count"] += 1

                print(
                    f"[{class_name}] {os.path.basename(img_path)}: 예측={predicted_class}, 신뢰도={confidence:.4f}, 정확함={is_correct}")

    # 정확도 계산
    if results["summary"]["processed_images"] > 0:
        results["summary"]["accuracy"] = results["summary"]["correct_predictions"] / results["summary"][
            "processed_images"]

    # 클래스별 정확도 계산
    for class_name, class_data in results["class_results"].items():
        if class_data["total"] > 0:
            class_data["accuracy"] = class_data["correct"] / class_data["total"]
            results["summary"]["class_accuracies"][class_name] = class_data["accuracy"]

    # 평균 추론 시간 계산
    if inference_times:
        results["summary"]["avg_inference_time"] = sum(inference_times) / len(inference_times)

    # 평균 신뢰도 계산
    if all_confidences:
        results["summary"]["avg_confidence"] = sum(all_confidences) / len(all_confidences)

        correct_confidences = [conf for conf, correct in zip(all_confidences, all_correct_flags) if correct]
        incorrect_confidences = [conf for conf, correct in zip(all_confidences, all_correct_flags) if not correct]

        if correct_confidences:
            results["summary"]["avg_confidence_correct"] = sum(correct_confidences) / len(correct_confidences)

        if incorrect_confidences:
            results["summary"]["avg_confidence_incorrect"] = sum(incorrect_confidences) / len(incorrect_confidences)

    # Top-K 정확도 계산
    for k in range(1, args.top_k + 1):
        if results["summary"]["processed_images"] > 0:
            results["summary"]["top_k_accuracy"][f"top_{k}"] = top_k_correct[k] / results["summary"]["processed_images"]

    # 오분류 분석
    misclassified = [(true, pred) for true, pred in zip(y_true, y_pred) if true != pred]
    misclassified_counts = defaultdict(int)

    for true, pred in misclassified:
        true_class = class_names[true]
        pred_class = class_names[pred]
        key = f"{true_class} -> {pred_class}"
        misclassified_counts[key] += 1

    # 상위 오분류 케이스 기록
    top_misclassified = sorted(misclassified_counts.items(), key=lambda x: x[1], reverse=True)
    results["misclassified_analysis"] = {
        "top_misclassifications": dict(top_misclassified[:20]),
        "total_misclassifications": len(misclassified)
    }

    # 시각화 결과 생성
    if len(y_true) > 0 and len(y_pred) > 0:
        # 혼동 행렬 시각화
        plot_confusion_matrix(y_true, y_pred, class_names,
                              os.path.join(args.report_dir, "confusion_matrix.png"))

        # 클래스별 정확도 시각화
        plot_class_accuracies(results["summary"]["class_accuracies"],
                              os.path.join(args.report_dir, "class_accuracies.png"))

        # 신뢰도 분포 시각화
        plot_confidence_distribution(all_confidences, all_correct_flags,
                                     os.path.join(args.report_dir, "confidence_distribution.png"))

        # 분류 보고서 생성 및 저장
        print("y_true: ", len(y_true))
        print("y_pred: ", len(y_pred))
        print("class_names", len(class_names))
        unique_true = np.unique(y_true)
        unique_pred = np.unique(y_pred)
        print("Unique classes in y_true:", len(unique_true), unique_true)
        print("Unique classes in y_pred:", len(unique_pred), unique_pred)
        print("Length of combined_class_names:", len(class_names))

        # 중복 확인
        if len(class_names) != len(set(class_names)):
            print("Warning: Duplicates found in combined_class_names")
            from collections import Counter
            print("Duplicates:", {k: v for k, v in Counter(class_names).items() if v > 1})

        # y_true에 맞게 target_names 조정
        adjusted_class_names = [class_names[i] for i in unique_true]
        report = classification_report(
            y_true,
            y_pred,
            labels=unique_true,
            target_names=adjusted_class_names,
            output_dict=True,
            zero_division=0
        )
        results["classification_report"] = report
        # report = classification_report(y_true, y_pred, labels=class_names, target_names=class_names, output_dict=True)
        # results["classification_report"] = report

        # CSV 형식으로 분류 보고서 저장
        report_csv_path = os.path.join(args.report_dir, "classification_report.csv")
        with open(report_csv_path, 'w', newline='') as csvfile:
            fieldnames = ['class', 'precision', 'recall', 'f1-score', 'support']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for class_name in class_names:
                if class_name in report:
                    writer.writerow({
                        'class': class_name,
                        'precision': report[class_name]['precision'],
                        'recall': report[class_name]['recall'],
                        'f1-score': report[class_name]['f1-score'],
                        'support': report[class_name]['support']
                    })

    return results


# ---------------
# [ 메인 함수 ]
# ---------------
def main():
    args = parse_arguments()

    # 테스트 변환 정의
    test_transforms = get_transforms(train=False)

    # 데이터셋 구성 로드
    dataset_configs = get_dataset_configs()
    if not dataset_configs:
        raise RuntimeError("사용할 데이터셋 구성이 없습니다.")

    # 클래스 이름 목록 준비
    combined_class_names = []
    for config in dataset_configs:
        train_dir = config["train_dir"]
        if os.path.exists(train_dir):
            dataset = ImageFolder(train_dir)
            print("Test dataset classes:", len(dataset.classes), dataset.classes)
            print("Test dataset class_to_idx:", dataset.class_to_idx)
            classes = [str(config['name'])+str(c.replace(" ","")) for c in dataset.classes]
            combined_class_names.extend(classes)
    num_classes = len(combined_class_names)
    print("전체 클래스 수:", num_classes)
    print("통합 클래스 목록:", combined_class_names)
    print("5:", combined_class_names[5])
    print("11:", combined_class_names[11])
    print("12:", combined_class_names[12])
    print("29:", combined_class_names[29])
    print("34:", combined_class_names[34])
    print("55:", combined_class_names[55])

    # 모델 준비
    if args.model == "resnet50":
        model = resnet50(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:  # convnext_tiny
        model = convnext_tiny(pretrained=True)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
        print("ConvNeXt output num_classes:", num_classes)
        print("ConvNeXt output dimension:", model.classifier[2])

    model = model.to(device)

    # 모델 로드
    print(f">> 모델 로드 중: {args.model_path}")
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found: {args.model_path}")

    model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
    model.eval()

    # 테스트 진행
    print(f">> {args.test_dir} 폴더 내 이미지 테스트 진행 중")
    print(f">> 결과는 {args.report_dir} 폴더에 저장됩니다.")
    test_results = test_images(model, args.test_dir, combined_class_names, test_transforms, args)

    # 결과 출력
    print("\n===== 테스트 결과 요약 =====")
    print(f"총 이미지: {test_results['summary']['total_images']}")
    print(f"처리된 이미지: {test_results['summary']['processed_images']}")
    print(f"실패한 이미지: {test_results['summary']['failed_images']}")
    print(f"정확히 예측된 이미지: {test_results['summary']['correct_predictions']}")
    print(f"전체 정확도: {test_results['summary']['accuracy']:.4f}")

    # Top-K 정확도 출력
    for k, acc in test_results['summary']['top_k_accuracy'].items():
        print(f"{k.replace('_', '-')} 정확도: {acc:.4f}")

    print(f"평균 추론 시간: {test_results['summary']['avg_inference_time']:.4f}초/이미지")
    print(f"평균 신뢰도: {test_results['summary']['avg_confidence']:.4f}")
    print(f"낮은 신뢰도 예측 수: {test_results['summary']['low_confidence_count']}")

    # 상위 오분류 케이스 출력
    print("\n===== 상위 오분류 케이스 =====")
    for case, count in list(test_results['misclassified_analysis']['top_misclassifications'].items())[:10]:
        print(f"{case}: {count}회")

    # 클래스별 정확도 출력
    print("\n===== 클래스별 정확도 =====")
    accuracies = [(cls, acc) for cls, acc in test_results['summary']['class_accuracies'].items()]
    # 정확도가 높은 순으로 정렬 (내림차순)
    accuracies.sort(key=lambda x: x[1], reverse=True)
    for class_name, acc in accuracies:
        print(f"{class_name}: {acc:.4f}")

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.floating):  # float32, float64 등
                return float(obj)
            elif isinstance(obj, np.bool_):  # NumPy 불리언 타입
                return bool(obj)
            elif isinstance(obj, np.ndarray):  # NumPy 배열
                return obj.tolist()
            return super().default(obj)

    # 최종 결과를 JSON 파일로 저장
    with open(args.output, 'w') as f:
        json.dump(test_results, f, indent=4, ensure_ascii=False, cls=NumpyEncoder)

if __name__ == '__main__':
    main()
