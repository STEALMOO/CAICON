from PIL import Image
import os

# 원본 이미지 폴더와 저장할 폴더 경로 설정
input_folder = "/root/caicon/data/test/images"
output_folder = "/root/caicon/resized_test/images"

# 저장할 폴더가 없으면 생성
os.makedirs(output_folder, exist_ok=True)

# 이미지 파일들을 처리
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        img_path = os.path.join(input_folder, filename)
        with Image.open(img_path) as img:
            # 이미지 리사이즈
            resized_img = img.resize((640, 640))
            # 새로운 경로에 이미지 저장
            resized_img.save(os.path.join(output_folder, filename))

print("이미지 리사이즈 및 저장 완료!")