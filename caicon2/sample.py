import os
import json
import cv2
from PIL import Image

def draw_bboxes_from_json(json_file, images_folder, output_folder):
    # JSON 파일을 열고 데이터 로드
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 출력 폴더가 없다면 생성
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 이미지와 어노테이션 정보를 사용하여 바운딩 박스 그리기
    for image_info in data['images']:
        image_id = image_info['id']
        file_name = image_info['file_name']
        image_path = os.path.join(images_folder, file_name)

        # 이미지 로드
        image = cv2.imread(image_path)
        if image is None:
            print(f"이미지를 불러올 수 없습니다: {image_path}")
            continue

        # 어노테이션에서 해당 이미지의 바운딩 박스를 가져옴
        annotations = [anno for anno in data['annotations'] if anno['image_id'] == image_id]
        
        for anno in annotations:
            # 바운딩 박스 좌표
            x_min, y_min, width, height = anno['bbox']
            x_min, y_min = int(x_min), int(y_min)
            x_max, y_max = int(x_min + width), int(y_min + height)
            
            # 클래스 ID로 색상을 지정 (여기서는 간단히 ID를 색상으로 사용)
            category_id = anno['category_id']
            color = (0, 255, 0) if category_id == 0 else (0, 0, 255)  # 간단한 클래스별 색상

            # 바운딩 박스 그리기
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
            # 클래스 ID 텍스트 추가
            cv2.putText(image, f"Class {category_id}", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # 결과 이미지를 출력 폴더에 저장
        output_path = os.path.join(output_folder, file_name)
        cv2.imwrite(output_path, image)
        print(f"바운딩 박스가 그려진 이미지 저장: {output_path}")

# 사용 예시
json_file = '/root/caicon/train_data.json'  # JSON 파일 경로
images_folder = '/root/caicon/train/images'  # 이미지 파일들이 있는 폴더
output_folder = '/root/caicon'  # 바운딩 박스가 그려진 이미지를 저장할 폴더

draw_bboxes_from_json(json_file, images_folder, output_folder)