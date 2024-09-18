import pandas as pd

# CSV 파일 불러오기
submission_file_path = '/root/caicon/submission_efficientnet.csv'
submission_df = pd.read_csv(submission_file_path)

# width와 height 값 설정
img_width, img_height = 640, 640

# 변환 함수
def convert_bbox(row):
    x = row['x']
    y = row['y']
    w = row['w']
    h = row['h']
    
    # 변환 수식 적용
    x_center = (x / img_width) + (w / (2 * img_width))
    y_center = (y / img_height) + (h / (2 * img_height))
    box_width = w / img_width
    box_height = h / img_height
    
    return pd.Series([x_center, y_center, box_width, box_height])

# 기존의 x, y, w, h 값을 변환하여 새로운 컬럼으로 추가
submission_df[['x_center', 'y_center', 'box_width', 'box_height']] = submission_df.apply(convert_bbox, axis=1)

# 변환된 데이터로 새로운 CSV 파일 저장
converted_submission_file_path = '/root/caicon'
submission_df.to_csv(converted_submission_file_path, index=False)

converted_submission_file_path