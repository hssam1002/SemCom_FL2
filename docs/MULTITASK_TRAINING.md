# Multi-Task Training Guide

Florence-2는 multi-task 모델이므로 여러 task를 동시에 학습할 수 있습니다.

## COCO Dataset의 Annotation 종류

COCO dataset은 다음 annotation들을 제공합니다:

1. **Captions** (`captions_train2017.json`)
   - 이미지에 대한 텍스트 설명
   - 각 이미지당 약 5개의 caption 제공
   - Task prompts: `<CAPTION>`, `<DETAILED_CAPTION>`, `<MORE_DETAILED_CAPTION>`

2. **Object Detection** (`instances_train2017.json`)
   - Bounding boxes와 category labels
   - 80개의 object categories
   - Task prompt: `<OD>`

3. **Instance Segmentation** (`instances_train2017.json`)
   - Segmentation masks (polygon 또는 RLE)
   - Object detection과 함께 제공
   - Task prompt: `<OD>` (Florence-2는 OD로 segmentation도 처리)

4. **Keypoints** (`person_keypoints_train2017.json`)
   - Person pose estimation
   - 17개의 keypoints (nose, eyes, shoulders, etc.)
   - Task prompt: `<OD>` (Florence-2는 OD로 keypoints도 처리)

## Florence-2의 다른 Tasks

Florence-2는 더 많은 task를 지원하지만, COCO dataset으로는 일부만 학습 가능합니다:

### ✅ COCO로 학습 가능
- Caption variants: `<CAPTION>`, `<DETAILED_CAPTION>`, `<MORE_DETAILED_CAPTION>`
- Object Detection: `<OD>`
- Segmentation: `<OD>` (OD로 처리)
- Keypoints: `<OD>` (OD로 처리)
- Dense Region Captioning: `<DENSE_REGION_CAPTION>` (OD 결과 기반, 파생 가능)
- Region Proposal: `<REGION_PROPOSAL>` (OD 결과 기반, 파생 가능)

### ❌ COCO로 학습 불가능
- Grounding tasks: `<GROUNDING>`, `<CAPTION_TO_PHRASE_GROUNDING>` (text input 필요)
- OCR tasks: `<OCR>`, `<OCR_WITH_REGION>` (COCO에 text annotation 없음)
- Open Vocabulary Detection: `<OPEN_VOCABULARY_DETECTION>` (text input 필요)

자세한 내용은 `docs/FLORENCE2_TASKS.md`를 참조하세요.

## Multi-Task Dataset 사용

### 1. Dataset 초기화

```python
from data.coco_multitask_dataset import COCOMultiTaskDataset

# 모든 task 포함
dataset = COCOMultiTaskDataset(
    data_root="/data4/hongsik/data/COCO",
    tasks=['caption', 'od', 'segmentation', 'keypoints']
)

# 특정 task만 포함
dataset = COCOMultiTaskDataset(
    data_root="/data4/hongsik/data/COCO",
    tasks=['caption', 'od']
)
```

### 2. Dataset 구조

각 sample은 다음 정보를 포함합니다:

```python
{
    'image': PIL.Image,           # 이미지
    'task': str,                  # 'caption', 'od', 'keypoints', etc.
    'task_prompt': str,           # '<CAPTION>', '<OD>', etc.
    'ground_truth': dict or str,  # Task별 ground truth
    'image_id': int,             # COCO image ID
    'filename': str              # 이미지 파일명
}
```

### 3. Task별 Ground Truth 형식

#### Caption
```python
ground_truth = "A green car parked in front of a yellow building."
```

#### Object Detection
```python
ground_truth = {
    'bboxes': [[x, y, width, height], ...],  # List of bboxes
    'labels': ['car', 'person', ...]         # List of category names
}
```

#### Keypoints
```python
ground_truth = [
    {
        'keypoints': [x1, y1, v1, x2, y2, v2, ...],  # Flattened keypoints
        'num_keypoints': 17,
        'bbox': [x, y, width, height]
    },
    ...
]
```

## 필요한 Annotation 파일

Multi-task training을 위해서는 다음 파일들이 필요합니다:

```
/data4/hongsik/data/COCO/
├── annotations/
│   ├── captions_train2017.json          # Caption annotations
│   ├── instances_train2017.json         # Detection/Segmentation annotations
│   └── person_keypoints_train2017.json # Keypoints annotations (optional)
└── train2017/
    └── *.jpg
```

### 다운로드 방법

`annotations_trainval2017.zip`을 다운로드하면 모든 annotation 파일이 포함되어 있습니다:

```bash
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip
cp annotations/captions_train2017.json /data4/hongsik/data/COCO/annotations/
cp annotations/instances_train2017.json /data4/hongsik/data/COCO/annotations/
cp annotations/person_keypoints_train2017.json /data4/hongsik/data/COCO/annotations/
```

## Training 전략

### Option 1: Task별로 순차 학습
각 task를 순차적으로 학습:
- Epoch 1-10: Caption task
- Epoch 11-20: OD task
- 등등

### Option 2: Task별로 배치 구성
각 batch를 같은 task로 구성:
- Batch 1: Caption samples만
- Batch 2: OD samples만
- 등등

### Option 3: Mixed Batch (권장)
각 batch에 여러 task를 섞어서 학습:
- Batch에 caption, OD, keypoints 등이 섞여있음
- 각 sample의 task_prompt에 따라 다른 loss 계산

## Loss 계산

각 task별로 다른 loss를 계산해야 합니다:

1. **Caption**: Cross-entropy loss (현재 구현됨)
2. **OD**: Detection loss (bbox regression + classification)
3. **Segmentation**: Segmentation loss
4. **Keypoints**: Keypoint loss

Florence-2의 경우, 모든 task가 language model을 통해 처리되므로,
각 task의 output format에 맞게 loss를 계산해야 합니다.

## 다음 단계

1. `train.py`를 수정하여 `COCOMultiTaskDataset` 지원
2. Task별 loss 계산 로직 추가
3. Mixed batch training 구현
