# import os
#
# def rename_folders_with_prefix(target_dir, prefix):
#     # 지정된 디렉터리 내 모든 항목 조회
#     for entry in os.listdir(target_dir):
#         full_path = os.path.join(target_dir, entry)
#         # 폴더인 경우 이름 변경 수행
#         if os.path.isdir(full_path):
#             new_name = prefix + entry
#             new_path = os.path.join(target_dir, new_name)
#             os.rename(full_path, new_path)
#             print(f"폴더 이름 변경: '{entry}' → '{new_name}'")
#
# # 사용 예시
# target_directory = r"C:\Users\xodnr\Downloads\dev\project\KTB_AI_study\8week_task\datasets\use\misrakahmed_vegetable-image-dataset\test"  # 대상 폴더 경로 지정
# prefix_text = "misrakahmed_vegetable-image-dataset"# 붙이고자 하는 텍스트
# rename_folders_with_prefix(target_directory, prefix_text)
import os


def print_folders(path):
    """
    주어진 경로 내의 모든 폴더명을 출력하는 함수

    Args:
        path (str): 탐색할 디렉토리 경로
    """
    try:
        # listdir() 메서드로 경로 내 모든 항목 가져오기
        items = os.listdir(path)

        # 폴더(디렉토리)만 필터링하여 출력
        folders = [item for item in items if os.path.isdir(os.path.join(path, item))]

        print(f"'{path}' 경로의 폴더 목록:")
        for folder in folders:
            print(f"- {folder}")

        print(f"\n총 {len(folders)}개의 폴더가 있습니다.")

    except FileNotFoundError:
        print(f"경로 '{path}'를 찾을 수 없습니다.")
    except PermissionError:
        print(f"경로 '{path}'에 접근 권한이 없습니다.")


# 사용 예시
# 현재 디렉토리의 폴더 출력
print_folders('./datasets/use')

# 특정 경로의 폴더 출력 (예: '/home/user/documents')
# print_folders('/home/user/documents')