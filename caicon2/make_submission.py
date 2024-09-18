import json
import pandas as pd

# JSON 파일 읽기
with open('caicon.bbox.json', 'r') as file:
    json_data = json.load(file)

# JSON 데이터를 CSV 형식으로 변환
csv_data = {}
for item in json_data:
    # labels 구조에 맞게 변환
    label = {
        'class_id': item['category_id'],
        'conf': item['score'],
        'x': item['bbox'][0],
        'y': item['bbox'][1],
        'w': item['bbox'][2],
        'h': item['bbox'][3]
    }
    
    image_id = item['image_id']
    
    # 동일한 image_id에 대해 여러 bbox를 처리
    if image_id in csv_data:
        csv_data[image_id].append(label)
    else:
        csv_data[image_id] = [label]

# 데이터프레임 생성
csv_output = []
for image_id, labels in csv_data.items():
    csv_output.append({
        'id': image_id,
        'labels': json.dumps(labels)  # labels를 JSON 문자열로 변환
    })

csv_df = pd.DataFrame(csv_output)

# CSV 파일로 저장
csv_df.to_csv('submission_efficientnet2.csv', index=False)