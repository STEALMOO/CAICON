import os
import shutil
from collections import defaultdict
import pandas as pd

def move_images_with_class_balance(train_img_dir, train_label_dir, val_img_dir, val_label_dir, val_ratio=0.2):
    # 클래스 ID별로 레이블 파일을 그룹화하기 위해 딕셔너리 생성
    class_to_files = defaultdict(list)
    
    # 모든 레이블 파일을 읽어 각 클래스 ID에 따라 파일 분류
    for label_file in os.listdir(train_label_dir):
        if label_file.endswith('.txt'):
            label_path = os.path.join(train_label_dir, label_file)
            with open(label_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    class_id = line.split()[0]  # 클래스 ID 추출
                    # 이미지 파일은 .txt 파일의 이름과 동일한 .png 형식으로 가정
                    image_file = label_file.replace('.txt', '.png')
                    # 해당 클래스의 파일 목록에 추가
                    class_to_files[class_id].append((image_file, label_file))

    # 클래스 비율에 맞춰 파일을 이동할 목록을 선택
    files_to_move = []
    for class_id, files in class_to_files.items():
        # 각 클래스별로 이동할 파일의 수를 계산
        target_count = int(len(files) * val_ratio)
        files_to_move.extend(files[:target_count])  # 비율에 맞게 파일 선택

    # val 폴더가 없으면 생성
    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(val_label_dir, exist_ok=True)

    # 선택된 파일들을 val 폴더로 이동
    for img_file, label_file in files_to_move:
        src_img_path = os.path.join(train_img_dir, img_file)
        src_label_path = os.path.join(train_label_dir, label_file)
        # 이미지와 라벨 파일이 존재하는지 확인 후 이동
        if os.path.exists(src_img_path) and os.path.exists(src_label_path):
            shutil.move(src_img_path, os.path.join(val_img_dir, img_file))
            shutil.move(src_label_path, os.path.join(val_label_dir, label_file))

    print(f"총 {len(files_to_move)}개의 파일이 {val_img_dir} 및 {val_label_dir}로 이동되었습니다.")

# 사용 예시
train_img_dir = '/root/caicon/train/images'  # 이미지 파일들이 있는 디렉토리
train_label_dir = '/root/caicon/train/labels'  # 라벨 파일들이 있는 디렉토리
val_img_dir = '/root/caicon/valid/images'  # 검증용 이미지 파일들이 저장될 디렉토리
val_label_dir = '/root/caicon/val/labels'  # 검증용 라벨 파일들이 저장될 디렉토리

move_images_with_class_balance(train_img_dir, train_label_dir, val_img_dir, val_label_dir)