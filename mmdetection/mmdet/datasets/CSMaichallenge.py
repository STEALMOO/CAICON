# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
from typing import List, Union

from mmengine.fileio import get_local_path

from mmdet.registry import DATASETS
from .api_wrappers import COCO
from .base_det_dataset import BaseDetDataset

from .coco import CocoDataset # .coco python 파일 안에서 CocoDataset에 있는 파일을 불러오겠다.

@DATASETS.register_module()
class CSMaichallenge(CocoDataset):
    """Dataset for COCO."""

    METAINFO = {
        'classes':
        ('crack', 'pothole'),
        # palette is a list of color tuples, which is used for visualization.
        'palette':
        [(220, 20, 60), (119, 11, 32)]
    }
## palette는 앞에서부터 클래스 개수 만큼만 남기고 다 지우면 된다.
    