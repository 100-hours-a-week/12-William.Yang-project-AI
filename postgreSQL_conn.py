import os
import argparse
from dotenv import load_dotenv
import psycopg2
from psycopg2 import sql

# .env 파일 로드
load_dotenv()


def save_pth_file(file_content, save_path):
    """
    .env 파일의 설정을 사용하여 .pth 파일을 특정 경로에 저장하는 함수

    :param file_content: .pth 파일에 저장할 내용
    :param save_path: 파일을 저장할 전체 경로
    """
    # .env에서 데이터베이스 연결 매개변수 로드
    connection_params = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'database': os.getenv('DB_NAME'),
        'user': os.getenv('DB_USER'),
        'password': os.getenv('DB_PASSWORD'),
        'port': os.getenv('DB_PORT', '5432')
    }

    try:
        # 데이터베이스 연결
        conn = psycopg2.connect(**connection_params)
        cursor = conn.cursor()

        # 파일 내용을 데이터베이스에 저장 (선택사항)
        cursor.execute(
            sql.SQL("INSERT INTO pth_files (filename, content) VALUES (%s, %s)"),
            (os.path.basename(save_path), file_content)
        )

        # 디렉토리 존재 확인 및 생성
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # .pth 파일 저장
        with open(save_path, 'w') as f:
            f.write(file_content)

        # 변경사항 커밋
        conn.commit()
        print(f"{save_path}에 .pth 파일 저장 성공")

    except (Exception, psycopg2.Error) as error:
        print(f"파일 저장 중 오류 발생: {error}")

    finally:
        # 연결 종료
        if conn:
            cursor.close()
            conn.close()


def main():
    # 명령줄 인자 파서 설정
    parser = argparse.ArgumentParser(description='PostgreSQL .pth 파일 저장 스크립트')
    parser.add_argument('save_path', help='저장할 .pth 파일의 전체 경로')
    parser.add_argument('--content', default='/path/to/default/libraries',
                        help='저장할 라이브러리 경로 (기본값: /path/to/default/libraries)')

    # 인자 파싱
    args = parser.parse_args()

    # .pth 파일 저장 함수 호출
    save_pth_file(args.content, args.save_path)


if __name__ == "__main__":
    main()