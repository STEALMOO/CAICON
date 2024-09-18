import pandas as pd
import ast

# 경로 설정
path1 = '/root/caicon/Epoch_26.csv'
#path2 = '/root/caicon/Epoch_27.csv'
#path3 = '/root/caicon/Epoch_29.csv'
#path4 = '/root/caicon/Epoch_30.csv'
output_path = '/root/caicon/submission3.csv'

# CSV 파일 읽기
df1 = pd.read_csv(path1)
#df2 = pd.read_csv(path2)
#df3 = pd.read_csv(path3)
#df4 = pd.read_csv(path4)

# labels을 리스트로 변환하는 함수
def convert_labels_to_list(labels):
    return ast.literal_eval(labels)

# conf 값이 0.1 이상인 labels만 남기는 함수
def filter_labels(labels):
    return [label for label in labels if label['conf'] >= 0.095]

# labels 컬럼을 리스트 형태로 변환하고 conf 0.1 이상 필터링
df1['labels'] = df1['labels'].apply(lambda x: filter_labels(convert_labels_to_list(x)))
#df2['labels'] = df2['labels'].apply(lambda x: filter_labels(convert_labels_to_list(x)))
#df3['labels'] = df3['labels'].apply(lambda x: filter_labels(convert_labels_to_list(x)))
#df4['labels'] = df4['labels'].apply(lambda x: filter_labels(convert_labels_to_list(x)))

# 두 데이터프레임을 병합
merged_df = pd.concat([df1])

# 같은 id별로 labels 통합
merged_df = merged_df.groupby('id').agg({'labels': 'sum'}).reset_index()

# 최종 결과 CSV 파일로 저장
merged_df.to_csv(output_path, index=False)

print(f"병합된 CSV 파일이 {output_path}에 저장되었습니다.")
