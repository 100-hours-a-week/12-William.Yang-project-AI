## **서론 - 연구 배경 및 목적**

컴퓨터 비전 분야에서 합성곱 신경망(CNN)은 이미지 처리와 인식 기술의 근간으로, 지난 10여 년간 혁신적인 발전을 거듭해왔습니다. 

초기 AlexNet에서 시작된 CNN 모델은 ResNet50과 같은 아키텍처를 통해 깊은 신경망 학습의 새로운 지평을 열었으며, 의료 영상 분석, 자율주행, 얼굴 인식 등 다양한 분야에 혁명적 변화를 가져왔습니다.
최근 인공지능 연구는 단순한 성능 개선을 넘어 인간의 인지 능력에 근접하는 모델 개발에 주력하고 있습니다. 특히 ConvNeXt와 Vision Transformer(ViT)는 기존 CNN 아키텍처의 한계를 극복하고, Transformer 기반 구조로 모델의 표현력과 일반화 능력을 크게 향상시켰습니다.
본 연구의 핵심 목표는 CNN 모델의 성능을 혁신적으로 개선하고, 미래 컴퓨터 비전 기술의 새로운 가능성을 탐구하는 것입니다. 

## 기초적인 비교 분석 - **ResNet50 vs ConvNeXt**

이미지 분야에서 ResNet50은 오랜 기간 표준 모델로 자리잡아 왔으나, 최근 Vision Transformer(ViT) 및 ConvNeXt와 같은 신규 아키텍처가 등장하며 경쟁이 가속화되고 있습니다. 

**"A ConvNet for the 2020s" 논문 내용을 참고하여**  ResNet50과 같은 전통적인 모델과 ConvNeXt와 같은 최신 모델을 비교하여 CNN기반 딥러닝 아키텍처 설계의 진화를 이해하는 데 중요한 통찰력을 제공합니다.  

**[**"A ConvNet for the 2020s" 논문은 비전 트랜스포머의 등장에도 불구하고 순수한 컨볼루션 신경망(ConvNet)의 잠재력을 재조명한 연구입니다.]

**ResNet50**과 **ConvNeXt**의 성능 차이를 동일한 데이터셋에서의 실험 결과를 통해 검증하고, 개선 사항을 체계적으로 제시합니다. 첨부된 JSON 파일(results.json, results2.json)은 두 모델의 평가 결과를 포함하며, 이를 바탕으로 정확도, 추론 시간, 클래스별 성능 등을 다각도로 비교합니다.

### 대상 데이터셋

[](https://www.notion.so/1b9394a4806181a087fee3a9c26c62db?pvs=21) 10건 전체

### 결과 분석 - 1차 (epoch 10회)

| Model | Accuracy | Processed Images | Avg Inference Time (s) | Avg Confidence | Avg Confidence (Correct) | Avg Confidence (Incorrect) | low_confidence_count |
| --- | --- | --- | --- | --- | --- | --- | --- |
| convnext_tiny | 88.51% | 7078 | 0.00268 | 0.9442 | 0.9773 | 0.6894 | 420 |
| resnet50 | 86.55% | 7078 | 0.00217 | 0.9237 | 0.9666 | 0.6474 | 574 |

convnext_tiny_class_accuracies

![class_accuracies.png](attachment:e3c428ff-23e0-42e8-8321-9a32b90be333:class_accuracies.png)

convnext_tiny_confusion_matrix

![confusion_matrix.png](attachment:aac00186-0ca6-414c-a065-d25ea6f3119e:confusion_matrix.png)

resnet50_class_accuracies

![class_accuracies.png](attachment:6150a928-55b5-461e-abdd-12d26c96bb68:class_accuracies.png)

resnet50_confusion_matrix

![confusion_matrix.png](attachment:f4e27d92-2269-46f8-886b-a64900975387:confusion_matrix.png)

### 결과 분석 - 2차 (epoch 20회)

| Model | Accuracy | Processed Images | Avg Inference Time (s) | Avg Confidence | Avg Confidence (Correct) | Avg Confidence (Incorrect) | low_confidence_count |
| --- | --- | --- | --- | --- | --- | --- | --- |
| convnext_tiny | 90.35% | 7078 | 0.00258 | 0.9608 | 0.9882 | 0.7036 | 556 |
| resnet50 | 89.29% | 7078 | 0.00249 | 0.9455 | 0.9808 | 0.6516 | 768 |

convnext_tiny_class_accuracies

![class_accuracies.png](attachment:6b04c147-43a5-4857-b3d4-aa77043a0f27:class_accuracies.png)

convnext_tiny_confusion_matrix

![confusion_matrix.png](attachment:32cb3e11-5a5d-4be2-ad3a-d306c98477a0:confusion_matrix.png)

resnet50_class_accuracies

![class_accuracies.png](attachment:d2c6718b-44aa-4114-bb01-70335312069d:class_accuracies.png)

resnet50_confusion_matrix

![confusion_matrix.png](attachment:8172a52d-198f-4115-ae49-aa143d5c1342:confusion_matrix.png)

분석 코드 및 결과 파일 첨부.

학습 코드

[all_classcification.py](attachment:76e80f5c-e10d-4adf-b4c7-611c0954a489:all_classcification.py)

테스트 코드

[pth_tester.py](attachment:ba9cd332-5432-4a0d-a93b-57e2673e7655:pth_tester.py)

1차 결과 ResNet50

[results.json](attachment:e5514368-fd06-4f69-be10-98122e06b1c9:results.json)

1차 결과 ConvNeXt_tiny

[results.json](attachment:e5514368-fd06-4f69-be10-98122e06b1c9:results.json)

2차 결과 ResNet50

[results.json](attachment:3ce7fabf-c7dc-4a92-ab48-1967eda2d9db:results.json)

2차 결과 ConvNeXt_tiny

[results.json](attachment:7e9fb2fa-7e8c-4bee-ac0c-506bf030f078:results.json)

전체 데이터셋을 대상으로 한 ConvNext_tiny와 ResNet50의 성능 비교 결과, 흥미로운 차이점들이 발견되었습니다 

- **ConvNeXt**가 정확도와 신뢰도 측면에서 우수하지만, **ResNet50**이 추론 속도에서 약간 앞섭니다.
- Low-Confidence 이미지 수가 ResNet50에서 더 많다는 것은 모델의 확신도가 상대적으로 낮음을 의미합니다.

이미지 처리에서 뛰어난 성능을 보여주고 있지만, 그 일반화 성능에 대해서는 여전히 아쉬운 점이 많습니다. 특히, 정확도와 관련된 여러 사항에서 한계를 드러내는 경우가 많습니다.

## 성능 저하 내용 분석

## **1. 부분 가림(Occlusion) 문제**

**현상:**

- 대상 물체의 **일부가 가려졌을 때 오답 빈도가 증가**

![misrakahmed_vegetable-image-datasetTomato_to_kritikseth_fruit-and-vegetable-image-recognitionjalepeno_1190.jpg](attachment:f511e234-f537-4ac3-b12d-5c4d036117e8:misrakahmed_vegetable-image-datasetTomato_to_kritikseth_fruit-and-vegetable-image-recognitionjalepeno_1190.jpg)

![jonathansilva2020_orange-diseases-datasetblackspot_to_kritikseth_fruit-and-vegetable-image-recognitionmango_citrus black orange-diseases blackspot_22.jpg](attachment:e87c0b41-2f80-489b-8191-51f9bf2b5bcd:jonathansilva2020_orange-diseases-datasetblackspot_to_kritikseth_fruit-and-vegetable-image-recognitionmango_citrus_black_orange-diseases_blackspot_22.jpg)

- 특히, **주요 특징이 포함된 영역이 가려지면 오류율 급증**

**추정 원인:**

- 학습 데이터에 **완전히 보이는 이미지가 많고, 부분적으로 가려진 이미지는 적음**
- CNN 기반 구조가 특정 국소적 특징을 중심으로 학습하므로, 중요한 특징이 사라지면 예측 불안정

---

## **2. 배경이 복잡할 때 오답 증가**

**현상:**

- 대상 객체와 배경이 명확히 분리되지 않는 경우 **배경의 일부를 잘못 인식하여 오류 발생**

![jay7080dev_rice-plant-diseases-datasetBacterialblight_to_jay7080dev_rice-plant-diseases-datasetLeafsmut_rice-plant-diseases leaf Bacterialblight_52.jpg](attachment:9fbedc78-1317-46e5-a0bf-e81a8e3399fe:jay7080dev_rice-plant-diseases-datasetBacterialblight_to_jay7080dev_rice-plant-diseases-datasetLeafsmut_rice-plant-diseases_leaf_Bacterialblight_52.jpg)

![imsparsh_flowers-datasetsunflower_to_imsparsh_flowers-datasettulip_imsparsh_flowers-datasetsunflower_2.jpg](attachment:26cc07a9-a180-41bd-8ab4-ef0281663b48:imsparsh_flowers-datasetsunflower_to_imsparsh_flowers-datasettulip_imsparsh_flowers-datasetsunflower_2.jpg)

- 학습 데이터는 주로 **단순한 배경(스튜디오 촬영, 균일한 색상 등)**을 포함하는 경우가 많지만,
    
    인터넷에서 수집한 테스트 이미지에서는 **배경이 복잡하거나 잡음이 많음**
    

**추정 원인:**

- CNN 기반이므로 **주변 픽셀의 특성을 반영**
- 배경과 주요 객체가 섞이면 모델이 혼동을 일으켜 잘못된 클래스 예측

---

## **3. 특정 클래스 간 혼동(Class Confusion) 문제**

**현상:**

- 유사한 클래스 간 혼동이 자주 발생

![kritikseth_fruit-and-vegetable-image-recognitionlemon_to_kritikseth_fruit-and-vegetable-image-recognitionorange_Image_7.jpg](attachment:91b4ab62-0a40-48d2-8963-936838d9e9e9:kritikseth_fruit-and-vegetable-image-recognitionlemon_to_kritikseth_fruit-and-vegetable-image-recognitionorange_Image_7.jpg)

![kritikseth_fruit-and-vegetable-image-recognitionbellpepper_to_kritikseth_fruit-and-vegetable-image-recognitionpaprika_object only bell pepper_85.jpg](attachment:f00619c7-e572-41c0-b487-17a4be07c538:kritikseth_fruit-and-vegetable-image-recognitionbellpepper_to_kritikseth_fruit-and-vegetable-image-recognitionpaprika_object_only_bell_pepper_85.jpg)

- 특정한 각도(예: 측면 사진)에서 보면 **다른 객체와 비슷한 특징을 공유하여 오답 발생**

**추정 원인:**

- 학습 데이터에서 **서로 다른 클래스가 비슷한 패턴을 공유하는 경우**
- ConvNeXt-Tiny가 전체적인 구조보다는 국소적인 패턴(텍스처, 엣지 등)을 중점적으로 학습했을 가능성

---

## **4. 복수 개체 포함 시 오답 발생 문제**

**현상:**

- 이미지 내 **여러 개체가 동시에 포함된 경우** 모델이 어느 개체를 기준으로 판단해야 할지 혼동

![kritikseth_fruit-and-vegetable-image-recognitionbeetroot_to_kritikseth_fruit-and-vegetable-image-recognitionraddish_object only beetroot_45.jpg](attachment:694b4ba4-77c1-49ab-bb92-25eac3b348e2:kritikseth_fruit-and-vegetable-image-recognitionbeetroot_to_kritikseth_fruit-and-vegetable-image-recognitionraddish_object_only_beetroot_45.jpg)

![kritikseth_fruit-and-vegetable-image-recognitionbanana_to_kritikseth_fruit-and-vegetable-image-recognitionginger_object only banana_89.jpg](attachment:cf7c5ae4-4a47-413e-80b9-e6bcfa9fc11c:kritikseth_fruit-and-vegetable-image-recognitionbanana_to_kritikseth_fruit-and-vegetable-image-recognitionginger_object_only_banana_89.jpg)

- 특히, **서로 다른 클래스의 개체가 함께 있을 때 잘못된 클래스로 분류되는 경우 증가**
- 같은 클래스 내에서도 **크기가 다른 개체(예: 큰 개 & 작은 개)가 함께 있을 때 오답 발생 가능성 증가**

**추정 원인:**

- CNN기반 모델의 경우 **이미지를 전체적으로 하나의 클래스로 분류하는 방식(이미지 단위 분류)**이므로, 복수 개체가 있을 때 **어떤 개체를 기준으로 예측해야 하는지 명확하지 않음**
- 학습 데이터가 **단일 개체 중심으로 구성된 경우** 다중 개체 상황에서 성능 저하 가능성
- **Saliency(시각적 주목도) 문제가 발생**하여, 작은 개체가 중요하게 학습되지 못할 가능성

---

#
