import os
import json
from PIL import Image

def convert_txt_to_json(images_folder, labels_folder, output_json):
    json_data = {
        "images": [],
        "annotations": []
    }
    annotation_id = 1

    for image_file in os.listdir(images_folder):
        if image_file.endswith(('.png', '.jpg', '.jpeg')):
            image_id = os.path.splitext(image_file)[0]  # 파일 이름에서 확장자 제거
            image_path = os.path.join(images_folder, image_file)
            label_file = f"{image_id}.txt"
            label_path = os.path.join(labels_folder, label_file)

            # 이미지 정보 가져오기
            with Image.open(image_path) as img:
                width, height = img.size

            # 이미지 메타 정보 추가
            json_data["images"].append({
                "file_name": image_file,
                "height": height,
                "width": width,
                "id": image_id
            })

            # 어노테이션 정보 추가
            if os.path.exists(label_path):
                with open(label_path, 'r') as lf:
                    for line in lf:
                        # 클래스, x, y, width, height 값을 읽음
                        class_id, x_center, y_center, box_width, box_height = map(float, line.strip().split())

                        # 절대 좌표로 변환
                        x_min = (x_center - box_width / 2) * width
                        y_min = (y_center - box_height / 2) * height
                        box_width = box_width * width
                        box_height = box_height * height
                        area = box_width * box_height

                        # 어노테이션 추가
                        json_data["annotations"].append({
                            "id": annotation_id,
                            "image_id": image_id,
                            "bbox": [x_min, y_min, box_width, box_height],
                            "area": area,
                            "category_id": int(class_id),
                            "segmentation": [],
                            "iscrowd": 0
                        })
                        annotation_id += 1

    # JSON 파일 저장
    with open(output_json, 'w', encoding='utf-8') as json_file:
        json.dump(json_data, json_file, indent=4, ensure_ascii=False)

# 사용 예시
images_folder = '/root/caicon/resized_train/images'
labels_folder = '/root/caicon/data/train/labels'
output_json = '/root/caicon/train_data.json'

convert_txt_to_json(images_folder, labels_folder, output_json)