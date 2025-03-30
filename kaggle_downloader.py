import os
import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile

# 1. Kaggle API 인증
api = KaggleApi()
api.authenticate()

# 2. 다운로드할 데이터셋 URL 리스트 (예: Kaggle 데이터셋 URL에서 추출한 슬러그)
datasets = [
    "mahyeks/almond-varieties",
    "jay7080dev/rice-plant-diseases-dataset",
    "kritikseth/fruit-and-vegetable-image-recognition",
    "misrakahmed/vegetable-image-dataset",
    "imsparsh/flowers-dataset",
    "muratkokludataset/pistachio-image-dataset",
    "cookiefinder/tomato-disease-multiple-sources",
    "hafiznouman786/potato-plant-diseases-data",
    "jonathansilva2020/orange-diseases-dataset",
    "muratkokludataset/rice-image-dataset"
]


# 3. 데이터셋 다운로드 및 정리 함수
def download_and_organize(dataset_slug, base_dir='datasets'):
    # 데이터셋 이름에서 슬래시(/)를 언더스코어(_)로 변환하여 폴더명 생성
    dataset_name = dataset_slug.replace('/', '_')
    download_path = os.path.join(base_dir, dataset_name)

    # 디렉토리가 없으면 생성
    if not os.path.exists(download_path):
        os.makedirs(download_path)

    try:
        # 데이터셋 다운로드 (압축 파일로 저장됨)
        print(f"Downloading {dataset_slug} to {download_path}...")
        api.dataset_download_files(dataset_slug, path=download_path, unzip=False)

        # 다운로드된 zip 파일 경로
        zip_file = os.path.join(download_path, f"{dataset_name}.zip")

        # 압축 해제
        if os.path.exists(zip_file):
            print(f"Unzipping {zip_file}...")
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(download_path)
            # 압축 파일 삭제 (선택 사항)
            os.remove(zip_file)
            print(f"Completed: {dataset_slug}")
        else:
            print(f"Zip file not found for {dataset_slug}")

    except Exception as e:
        print(f"Error downloading {dataset_slug}: {str(e)}")


# 4. 모든 데이터셋 다운로드 실행
base_directory = 'datasets'  # 데이터를 저장할 기본 디렉토리
if not os.path.exists(base_directory):
    os.makedirs(base_directory)

for dataset in datasets:
    download_and_organize(dataset, base_directory)

print("모든 데이터셋 다운로드 및 정리 완료!")