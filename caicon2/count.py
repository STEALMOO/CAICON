import os

# 이미지 파일이 있는 디렉터리 경로
image_dir = '/root/caicon/data/train/images'

# 지원하는 이미지 파일 확장자 목록 (필요에 따라 추가 가능)
image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')

# 디렉터리에서 이미지 파일 개수를 세기
image_count = len([file for file in os.listdir(image_dir) if file.lower().endswith(image_extensions)])

print(f"이미지 파일의 개수: {image_count}장")
