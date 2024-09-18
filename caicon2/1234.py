import pandas as pd
import json

# CSV 파일 불러오기
file_path = '/root/caicon/submission_efficientnet2.csv'
df = pd.read_csv(file_path)

# 이미지 크기 설정 (가로, 세로)
img_width, img_height = 640, 640

# 좌상단 기준의 bbox를 중앙 기준으로 변환하면서 가로/세로 길이를 정규화하는 함수
def convert_bbox(bbox, img_width, img_height):
    # bbox는 [x, y, w, h] 형식의 리스트
    x = bbox['x']
    y = bbox['y']
    w = bbox['w']
    h = bbox['h']
    
    # 중앙 좌표로 변환
    x_center = (x + w / 2) / img_width
    y_center = (y + h / 2) / img_height
    
    # 가로와 세로 길이는 그대로 두고 정규화
    box_width = w / img_width
    box_height = h / img_height
    
    # 변환된 값을 반환
    bbox['x'] = x_center
    bbox['y'] = y_center
    bbox['w'] = box_width
    bbox['h'] = box_height
    return bbox

# 각 행의 'labels' 열을 파싱하여 bbox 변환 적용
def convert_labels(row):
    labels = json.loads(row['labels'])
    for label in labels:
        label = convert_bbox(label, img_width, img_height)
    return json.dumps(labels)

# 'labels' 열에 변환 적용
df['labels'] = df.apply(convert_labels, axis=1)

# 변환된 데이터를 새로운 CSV 파일로 저장
df.to_csv('/root/caicon/submission_converted2.csv', index=False)
