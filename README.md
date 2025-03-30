# 12-William.Yang-Project-AI  
**개인 과제: CNN 모델 학습**  

**프로젝트 개요**  
- **기간**: 2주  
- **목표**: 주어진 데이터셋을 활용해 원하는 사전 훈련된 CNN 모델을 학습시키고, 연구 보고서를 작성  
- **사용 모델**: ResNet50, ConvNeXt_tiny, YOLO, SAM 등  

---

## 서론  
컴퓨터 비전 분야에서 **합성곱 신경망(CNN)**은 이미지 처리와 인식 기술의 핵심으로 자리 잡고 있습니다. 본 프로젝트는 CNN 모델의 성능을 혁신적으로 개선하고, 미래 컴퓨터 비전 기술의 새로운 가능성을 탐구하는 데 초점을 맞췄습니다.  

프로젝트는 총 **4단계**로 설계되었으며, 현재 **1, 2단계**가 완료된 상태입니다. 시간 제약으로 인해 **3, 4단계**는 추후 진행 예정입니다.  
- **1단계**: ResNet50과 ConvNeXt_tiny 모델 비교 분석  
- **2단계**: YOLO, SAM, ConvNeXt 기술 통합 가능성 탐구  
- **3단계**: ConvNeXt_tiny 모델의 앙상블 (미완료)  
- **4단계**: 3단계 결과를 기반으로 단일 모델 구현 (미완료)  

---

## 개인 과제 수행  
### ResNet50 vs ConvNeXt_tiny 비교 분석  
1단계에서는 ResNet50과 ConvNeXt_tiny 모델을 비교 분석했습니다. 결과는 다음과 같습니다:  
- **ConvNeXt_tiny**: 정확도와 신뢰도에서 우수한 성능을 보임  
- **ResNet50**: 추론 속도에서 약간의 이점  

**세부 결과**:  
[1차 개인 과제 CNN 모델 학습 및 고찰 (PDF)](https://github.com/YangTaeUk/12-William.Yang-project-AI/blob/main/(1%EC%B0%A8)%EA%B0%9C%EC%9D%B8_%EA%B3%BC%EC%A0%9C_CNN_%EB%AA%A8%EB%8D%B8_%ED%95%99%EC%8A%B5_%EB%B0%8F_%EA%B3%A0%EC%B0%B0(ResNet50_vs_ConvNeXt_tiny).pdf)  
(*참고*: 데이터 크기 문제로 GitHub에 첨부 불가. [Notion 페이지](https://www.notion.so/adapterz/1-CNN-ResNet50-vs-ConvNeXt_tiny-8de8ab49aabc4ddc8469099dde98eeff) 참조)  

**주요 도전 과제**:  
1. 복수 개체 포함 시 분류 혼란  
2. 특정 각도에서 오답 발생  
3. 국소적 패턴에 대한 과도한 의존  

**해결 방안 제안**: 다양한 시각적 학습 전략과 모델 구조 개선 필요  

---

## 새로운 접근 모색  
### YOLO, SAM, ConvNeXt 통합 시도  
2단계에서는 YOLO, SAM, ConvNeXt 모델의 혁신적인 통합을 탐구했습니다. 각 모델의 역할은 다음과 같습니다:  
- **YOLO**: 실시간 객체 검출, 바운딩 박스 생성  
- **SAM**: 프롬프트 기반 객체 분할, YOLO의 바운딩 박스를 입력으로 활용  
- **ConvNeXt**: 최종 이미지 분류  

**세부 결과**:  
[2차 새로운 접근 방식 모색 (PDF)](https://github.com/YangTaeUk/12-William.Yang-project-AI/blob/main/(2%EC%B0%A8)%EC%83%88%EB%A1%9C%EC%9A%B4_%EC%A0%91%EA%B7%BC_%EB%B0%A9%EC%8B%9D_%EB%AA%A8%EC%83%89(YOLOSAMConvNeXt).pdf)  
(*참고*: 데이터 크기 문제로 GitHub에 첨부 불가. [Notion 페이지](https://www.notion.so/adapterz/2-YOLO-SAM-ConvNeXt-1c6394a480618005b384cbc0d8e6238f) 참조)  

**주요 도전 과제**:  
- YOLO의 부정확한 객체 검출 → SAM의 세그멘테이션 성능 저하 → ConvNeXt 분류 정확도 하락  
- 특정 객체(예: 바나나)에서 모델 간 오류 전파 현상  

**고찰**:  
이 접근법은 모델 간 상호 보완성을 탐구한 의미 있는 시도였으나, 각 모델의 한계와 성능 차이를 명확히 드러냈습니다.  

---

## CNN 이미지 분류 심화  
### ConvNeXt 앙상블 계획  
3단계에서는 ConvNeXt_tiny 모델의 앙상블을 통해 이미지 분류의 정확도와 신뢰성을 높이는 것을 목표로 했습니다.  
- **목표**: 최대 1000개 클래스 분류 가능 모델의 성능 최적화  
- **방법**: 다양한 학습 데이터셋과 모델 가중치를 활용한 앙상블  

**기대 효과**:  
- 개별 모델의 한계 극복  
- ConvNeXt의 정확도와 빠른 추론 속도 결합  

(*참고*: 시간 제약으로 미완료 상태)  

---

## 결론  
본 프로젝트는 CNN 모델의 성능 개선과 혁신적 접근을 목표로 진행되었습니다.  
- **1단계**: ResNet50과 ConvNeXt_tiny 비교를 통해 모델 특성 분석  
- **2단계**: YOLO, SAM, ConvNeXt 통합 가능성 탐구  

시간 제약으로 3, 4단계는 완료되지 않았으나, 현재까지의 연구에서 다중 개체 분류와 모델 간 상호운용성 등 중요한 기술적 과제를 도출했습니다.  
**향후 계획**: 발견된 한계를 극복하고, CNN 모델의 성능과 일반화 능력을 더욱 향상시키는 연구를 지속할 예정입니다.  
