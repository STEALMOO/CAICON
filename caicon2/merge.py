import pandas as pd
import ast

# 첫 번째 CSV 파일 읽기
file1 = '/root/caicon/submission_converted.csv'  # 첫 번째 파일 경로
df1 = pd.read_csv(file1, delimiter='\t', names=['id', 'labels'])

# 두 번째 CSV 파일 읽기
file2 = '/root/caicon/submission_efficientnet.csv'  # 두 번째 파일 경로
df2 = pd.read_csv(file2, delimiter='\t', names=['id', 'labels'])

# labels를 리스트로 변환 (string에서 dict로 파싱)
df1['labels'] = df1['labels'].apply(ast.literal_eval)
df2['labels'] = df2['labels'].apply(ast.literal_eval)

# 두 파일을 id를 기준으로 병합 (left join 사용)
merged_df = pd.merge(df1, df2, on='id', how='outer', suffixes=('_file1', '_file2'))

# labels 값 합치기
merged_df['combined_labels'] = merged_df.apply(
    lambda row: row['labels_file1'] + row['labels_file2'] if pd.notnull(row['labels_file1']) and pd.notnull(row['labels_file2']) else row['labels_file1'] if pd.notnull(row['labels_file1']) else row['labels_file2'],
    axis=1
)

# 결과 출력
print(merged_df[['id', 'combined_labels']])

# CSV로 저장
merged_df[['id', 'combined_labels']].to_csv('merged_labels.csv', index=False)