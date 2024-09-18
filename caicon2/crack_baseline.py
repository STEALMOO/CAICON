#!/usr/bin/env python
# coding: utf-8

# # 작전도로 상태 점검을 위한 노면 이상 탐지 모델 개발

# ## 환경 설정
# 본 베이스라인 코드를 활용하기 위한 베이스라인 환경을 설정합니다.

# In[ ]:


#get_ipython().system('pip install -r requirements.txt')


# ## 데이터 불러오기

# In[ ]:


import json
import os
import shutil

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import pandas as pd
import yaml
from PIL import Image
from tqdm import tqdm

DATASET_ROOT = "/root/caicon/data"  # 데이터 셋의 루트 디렉토리
LABELS_DIR = "labels"
IMAGE_DIR = "images"
TRAIN_DIR = "train"
TEST_DIR = "test"

# DATASET_ROOT/train/labels: 학습 데이터의 라벨 파일이 있는 디렉토리
# DATASET_ROOT/train/images: 학습 데이터의 이미지 파일이 있는 디렉토리
# DATASET_ROOT/test/labels: 테스트 데이터의 라벨 파일이 있는 디렉토리
# DATASET_ROOT/test/images: 테스트 데이터의 이미지 파일이 있는 디렉토리


# ### 학습 데이터 불러오기

# In[ ]:


train_data = []

for image in tqdm(os.listdir(os.path.join(DATASET_ROOT, TRAIN_DIR, IMAGE_DIR))):
    image_id = image.split(".")[0]
    image_path = os.path.join(DATASET_ROOT, TRAIN_DIR, IMAGE_DIR, image)
    label_path = os.path.join(DATASET_ROOT, TRAIN_DIR, LABELS_DIR, image_id + ".txt")
    labels = []
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            for line in f.readlines():
                class_id = int(line.split()[0])
                x = float(line.split()[1])
                y = float(line.split()[2])
                w = float(line.split()[3])
                h = float(line.split()[4])
                labels.append({"class_id": class_id, "x": x, "y": y, "w": w, "h": h})

    train_data.append({"id": image_id, "image_path": image_path, "label_path": label_path, "labels": labels})

df_train = pd.DataFrame(train_data)
df_train.head()


# ### 테스트 데이터 불러오기

# In[ ]:


test_data = []

for image in tqdm(os.listdir(os.path.join(DATASET_ROOT, TEST_DIR, IMAGE_DIR))):
    image_id = image.split(".")[0]
    image_path = os.path.join(DATASET_ROOT, TEST_DIR, IMAGE_DIR, image)

    # 테스트 데이터는 라벨 파일이 없습니다.
    test_data.append({"id": image_id, "image_path": image_path, "label_path": "", "labels": []})

df_test = pd.DataFrame(test_data)
df_test.head()


# ## 샘플 데이터 확인하기

# ### 샘플 이미지

# In[ ]:


sample = df_train.iloc[1]

# 이미지 읽기
image_sample = Image.open(sample["image_path"])
image_sample


# ### 샘플 이미지의 라벨

# 라벨은 정규화된 XYWH 형식으로 작성되어 있습니다.
# 
# 정규화된 XYWH 형식
# `<object-class> <x> <y> <width> <height>`
# - object-class: 물체의 클래스 (0부터 시작하는 정수)
# - x, y: 바운딩 박스의 중심의 상대 좌표(0~1 사이 값)
# - width, height: 바운딩 박스의 너비와 높이(0~1 사이 값)

# In[ ]:


sample = df_train.iloc[1]
fig, ax = plt.subplots()

img = Image.open(sample["image_path"])

# 이미지 표시
ax.imshow(img)

# 박스 그리기
for label in sample["labels"]:
    # 좌표 변환
    c_x = label["x"] * img.width
    c_y = label["y"] * img.height
    w = label["w"] * img.width
    h = label["h"] * img.height

    left_top_x = c_x - w / 2
    left_top_y = c_y - h / 2

    rect = patches.Rectangle((left_top_x, left_top_y), w, h, linewidth=2, edgecolor="r", facecolor="none")
    ax.add_patch(rect)

fig.show()


# ## 데이터 전처리

# ### 데이터 분리

# In[ ]:


df_train_set = df_train.sample(frac=0.8, random_state=0)
df_valid_set = df_train.drop(df_train_set.index)

df_train_set.to_csv("train_set.csv", index=False)
df_valid_set.to_csv("valid_set.csv", index=False)

new_train_path = os.path.abspath("train")
new_valid_path = os.path.abspath("valid")

os.makedirs(new_train_path, exist_ok=True)
os.makedirs(os.path.join(new_train_path, IMAGE_DIR), exist_ok=True)
os.makedirs(os.path.join(new_train_path, LABELS_DIR), exist_ok=True)

os.makedirs(new_valid_path, exist_ok=True)
os.makedirs(os.path.join(new_valid_path, IMAGE_DIR), exist_ok=True)
os.makedirs(os.path.join(new_valid_path, LABELS_DIR), exist_ok=True)


# ### 데이터 전처리 및 복사

# 이미지의 크기를 변경하고 각각 폴더에 복사합니다.

# In[ ]:


# 원본 이미지 크기: 1280x720
# 이미지 크기를 480x480으로 변경
IMAGE_SIZE = 480

for i, row in tqdm(df_train_set.iterrows(), total=len(df_train_set)):
    image = Image.open(row["image_path"])
    image.resize((IMAGE_SIZE, IMAGE_SIZE)).save(f"{new_train_path}/{IMAGE_DIR}/{row['id']}.jpg")
    shutil.copy(row["label_path"], f"{new_train_path}/{LABELS_DIR}/{row['id']}.txt")

for i, row in tqdm(df_valid_set.iterrows(), total=len(df_valid_set)):
    image = Image.open(row["image_path"])
    image.resize((IMAGE_SIZE, IMAGE_SIZE)).save(f"{new_valid_path}/{IMAGE_DIR}/{row['id']}.jpg")
    shutil.copy(row["label_path"], f"{new_valid_path}/{LABELS_DIR}/{row['id']}.txt")


# ## 모델 학습

# ### 모델 준비

# 외부 모델을 사용하는 경우 아래처럼 모델을 불러올 수 있습니다.
# 
# - 다운로드 하는 코드를 명시
# - 필요한 종속성을 설치하는 코드를 명시

# In[ ]:


# YOLOv5 클론 및 설치 
#get_ipython().system('git clone https://github.com/ultralytics/yolov5.git')
#get_ipython().system('pip install --quiet -r yolov5/requirements.txt')


#git clone https://github.com/ultralytics/yolov5.git
#pip install --quiet -r yolov5/requirements.txt


# ### 모델 학습

# In[ ]:


dataset = {
    "path": os.path.abspath("."),
    "train": "train",
    "val": "valid",
    "nc": 2,
    # crack = 도로 균열, pothole = 포트홀
    "names": ["crack", "pothole"],
}

YAML_PATH = os.path.abspath("pothole.yaml")
RESULT_PATH = os.path.abspath("result")
os.makedirs(RESULT_PATH, exist_ok=True)
with open(YAML_PATH, "w") as f:
    yaml.dump(dataset, f)


# In[ ]:


#get_ipython().system('PYTHONWARNINGS="ignore::FutureWarning" python yolov5/train.py --img 480 --batch 32 --epoch 3 --data {YAML_PATH} --cfg yolov5s.yaml --exist-ok --name pothole --project {RESULT_PATH}')
PYTHONWARNINGS="ignore::FutureWarning" python yolov5/train.py --img 480 --batch 32 --epoch 3 --data {YAML_PATH} --cfg yolov5s.yaml --exist-ok --name pothole --project {RESULT_PATH}

# ## 모델 평가

# ### 검증 데이터 예측

# In[ ]:


get_ipython().system('python yolov5/detect.py --source {os.path.join(new_valid_path, IMAGE_DIR)} --weights {RESULT_PATH}/pothole/weights/best.pt --conf 0.1 --save-txt --save-conf --exist-ok --project {RESULT_PATH}/valid')


# ### 예측 결과 시각화

# In[ ]:


sample = df_valid_set.iloc[3]
fig, ax = plt.subplots()

img = Image.open(sample["image_path"])
ax.imshow(img)

# 예측 결과 시각화
predict = os.path.join(RESULT_PATH, "valid", "exp", "labels", sample["id"] + ".txt")
if os.path.exists(predict):
    with open(predict, "r") as f:
        for line in f.readlines():
            class_id, c_x, c_y, w, h, conf = map(float, line.split())
            c_x *= img.width
            c_y *= img.height
            w *= img.width
            h *= img.height

            left_top_x = c_x - w / 2
            left_top_y = c_y - h / 2

            rect = patches.Rectangle(
                (left_top_x, left_top_y), w, h, linewidth=1, edgecolor="r", facecolor="none"
            )
            ax.add_patch(rect)

# 정답 시각화
with open(sample["label_path"], "r") as f:
    for line in f.readlines():
        class_id, c_x, c_y, w, h = map(float, line.split())
        c_x *= img.width
        c_y *= img.height
        w *= img.width
        h *= img.height

        left_top_x = c_x - w / 2
        left_top_y = c_y - h / 2

        rect = patches.Rectangle((left_top_x, left_top_y), w, h, linewidth=1, edgecolor="w", facecolor="none")
        ax.add_patch(rect)
fig.set_figheight(10)
fig.show()


# ## 제출 파일 생성

# ### 테스트 데이터 예측

# In[ ]:


get_ipython().system('rm -rf {RESULT_PATH}/test/exp')


# In[ ]:


get_ipython().system('python yolov5/detect.py --source {os.path.join(DATASET_ROOT, TEST_DIR, IMAGE_DIR)} --weights {RESULT_PATH}/pothole/weights/best.pt --conf 0.1 --save-txt --save-conf --exist-ok --project {RESULT_PATH}/test')


# ### 바운드 박스 읽어오기

# In[ ]:


for i, row in tqdm(df_test.iterrows(), total=len(df_test)):
    label_path = os.path.join(RESULT_PATH, "test", "exp", "labels", row["id"] + ".txt")

    labels = []
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            class_id, c_x, c_y, w, h, conf = map(float, line.split())
            labels.append({"class_id": int(class_id), "conf": float(conf), "x": c_x, "y": c_y, "w": w, "h": h})

    df_test.at[i, "labels"] = labels
df_test


# ### 생성되는 CSV 파일의 형식

# CSV 파일은 각 이미지에 대한 정보를 포함하며, 다음과 같은 열(column)로 구성됩니다: 
# 
# | id        | labels                             | 
# |-----------|------------------------------------| 
# | image_id  | [{"class_id": ..., "conf": ..., "x": ..., "y": ..., "w": ..., "h": ...}, ...] | 
# 
# 각 행(row)은 다음과 같은 내용을 포함합니다: 
# - id: 이미지의 고유 식별자 (image_id) 
# - labels: 해당 이미지에 대한 라벨 정보를 포함하는 리스트 
#   - 각 라벨은 다음과 같은 정보를 포함합니다: 
#     - class_id: 객체의 클래스 ID (정수형)
#     - x: 객체의 정규화된 x 좌표 (0 이상 1 이하)
#     - y: 객체의 정규화된 y 좌표 (0 이상 1 이하)
#     - w: 객체의 정규화된 너비 (0 이상 1 이하)
#     - h: 객체의 정규화된 높이  (0 이상 1 이하)
#     - conf: 객체 탐지 결과의 신뢰도 (0 이상 1 이하)
# 
# conf 값은 생략할 수 있으며, 생략할 경우 채점 프로그램은 해당 라벨의 conf 값을 1.0 으로 간주합니다.
# 
# 또한, 채점 프로그램은 각 라벨이 정규화된 XYWH 포맷으로 작성되었다고 간주합니다.
# 
# 예시: 
# | id         | labels                                                       | 
# |------------|--------------------------------------------------------------| 
# | image_001  | [{"class_id": 1, "conf": 0.3, "x": 0.5, "y": 0.5, "w": 0.2, "h": 0.3}] | 
# | image_002  | [{"class_id": 0, "conf": 0.8, "x": 0.3, "y": 0.4, "w": 0.1, "h": 0.2}, {"class_id": 1, "conf": 1, "x": 0.7, "y": 0.8, "w": 0.15, "h": 0.25}] | 
# 
# 주의: x, y, w, h는 각각 이미지의 너비와 높이에 대한 비율로 표현되며, 
#       x와 y는 객체의 중심 좌표를 기준으로 하고, w와 h는 객체의 크기를 나타냅니다.

# ### 제출 파일 생성

# In[ ]:


submission = []

for i, row in tqdm(df_test.iterrows(), total=len(df_test)):
    image_id = row["id"]
    labels = []
    for label in row["labels"]:
        class_id = label["class_id"]
        x = label["x"]
        y = label["y"]
        w = label["w"]
        h = label["h"]
        conf = label["conf"]

        labels.append({"class_id": class_id, "conf": conf, "x": x, "y": y, "w": w, "h": h})
    submission.append({"id": image_id, "labels": labels})

df_submission = pd.DataFrame(submission)
df_submission.to_csv("submission.csv", index=False)


# ### 제출 파일 확인

# 제출한 파일이 올바른 형식인지 확인합니다.

# In[ ]:


df_submission = pd.read_csv("submission.csv")

sample = df_submission.iloc[9]
sample_labels = eval(sample["labels"])

if len(sample_labels) == 0:
    print("예측결과: 해당 도로 이미지에선 발견된 균열 및 구멍이 없습니다.")
else:
    fig, ax = plt.subplots()
    img = Image.open(os.path.join(DATASET_ROOT, TEST_DIR, IMAGE_DIR, sample["id"] + ".jpg"))
    ax.imshow(img)

    # 예측 결과 시각화
    for label in sample_labels:
        class_id = label["class_id"]
        cx = label["x"] * img.width
        cy = label["y"] * img.height
        w = label["w"] * img.width
        h = label["h"] * img.height

        left_top_x = cx - (w / 2)
        left_top_y = cy - h / 2

        rect = patches.Rectangle((left_top_x, left_top_y), w, h, linewidth=1, edgecolor="r", facecolor="none")
        ax.add_patch(rect)


# ## 제출
# 
# 1. 규정에 따라 `code.ipynb`, `requirements.txt` 파일을 포함하여 모델의 성능 재현에 필요한 모든 파일을 압축한 `code.zip` 파일을 제출 환경에 업로드 합니다.
#    1. 리눅스 환경에서는 다음 명령어를 활용하여 압축 파일을 생성할 수 있습니다: `zip code.zip code.ipynb requirements.txt`
#    2. 다른 파이썬 파일을 사용한 경우 위 명령어 뒤에 파일명을 추가하면 됩니다.
#    3. 특정 폴더를 압축해야 하는 경우 `zip -r code.zip code.ipynb requirements.txt 원하는/폴더/경로` 와 같은 명령어를 활용할 수 있습니다.
# 2. 그 후, `submission.csv` 파일을 업로드 합니다.
# 3. 두 파일이 업로드 된 것을 확인한 후, 오른쪽 위의 '제출' 버튼을 클릭하여 제출합니다.

# ## 제출 주의사항
# 1. `code.zip` 파일과 `submission.csv` 파일을 시스템이 압축한 파일의 용량이 50MB를 넘으면 안됩니다.
# 2. `code.zip` 파일 안에 `code.ipynb` 파일과 `requirements.txt` 파일이 있어야 합니다. 
# 
# 아래 코드를 실행하면 제출 주의사항을 모두 만족했는지 확인하실 수 있습니다.

# In[ ]:


import os
import zipfile

is_ready = True

# code.zip 파일이 생성되었는지 확인합니다.
if os.path.exists("code.zip"):
    result_zip = zipfile.ZipFile('code.zip')
    # 파일 목록을 확인합니다.
    submission_filelist = result_zip.namelist()
    if (
        "code.ipynb" not in submission_filelist
        or "requirements.txt" not in submission_filelist
    ):
        is_ready = False
        print("압축 파일에 필요한 파일이 모두 있는지 확인해주세요.")
else:
    is_ready = False
    print("code.zip 파일이 생성되었는지 확인해주세요.")

# submission.csv 파일이 생성되었는지 확인합니다.
if not os.path.exists("submission.csv"):
    is_ready = False
    print("submission.csv 파일이 생성되었는지 확인해주세요.")

if is_ready:
    # 파일의 용량을 확인합니다. 
    # 시스템 상 두 제출 파일을 압축한 후 50MB를 넘지 않으면 제출할 수 있으므로, 용량의 합이 50MB보다 약간 클 경우 제출이 가능할 수 있습니다.
    if os.path.getsize("code.zip") + os.path.getsize("submission.csv") > 50e6:
        print("제출 파일의 용량이 50MB를 초과하고 있습니다. 제출 전 용량을 줄여주세요.")
    else:
        print("제출할 준비가 되었습니다.")
else:
    print("제출 주의사항을 모두 만족하는지 확인해주세요.")

