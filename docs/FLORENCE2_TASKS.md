# Florence-2 Tasks & COCO Dataset 지원 현황

Florence-2는 다양한 vision-language task를 지원합니다. 이 문서는 각 task와 COCO dataset으로 학습 가능 여부를 정리합니다.

## Florence-2 지원 Tasks

### 1. Captioning Tasks

| Task Prompt | 설명 | COCO 지원 | 비고 |
|------------|------|-----------|------|
| `<CAPTION>` | 기본 이미지 캡션 | ✅ **가능** | `captions_train2017.json` |
| `<DETAILED_CAPTION>` | 상세 캡션 | ✅ **가능** | 같은 caption annotation 사용 |
| `<MORE_DETAILED_CAPTION>` | 더 상세한 캡션 | ✅ **가능** | 같은 caption annotation 사용 |

**COCO Dataset:**
- `captions_train2017.json`에 각 이미지당 약 5개의 caption 제공
- 모든 caption task에 동일한 annotation 사용 가능

### 2. Object Detection & Segmentation

| Task Prompt | 설명 | COCO 지원 | 비고 |
|------------|------|-----------|------|
| `<OD>` | Object Detection (bboxes + labels) | ✅ **가능** | `instances_train2017.json` |
| `<DENSE_REGION_CAPTION>` | Dense region captioning | ✅ **파생 가능** | OD 결과 기반으로 생성 |
| `<REGION_PROPOSAL>` | Region proposals | ✅ **파생 가능** | OD 결과 기반으로 생성 |

**COCO Dataset:**
- `instances_train2017.json`에 80개 category의 bboxes와 segmentation masks 제공
- Segmentation은 `<OD>` task로 처리 (Florence-2는 OD로 segmentation도 처리)

### 3. Keypoints

| Task Prompt | 설명 | COCO 지원 | 비고 |
|------------|------|-----------|------|
| `<OD>` (keypoints) | Person pose estimation | ✅ **가능** | `person_keypoints_train2017.json` |

**COCO Dataset:**
- `person_keypoints_train2017.json`에 person keypoints (17개) 제공
- Florence-2는 `<OD>` task로 keypoints도 처리

### 4. Text Input이 필요한 Tasks (COCO로 학습 어려움)

| Task Prompt | 설명 | COCO 지원 | 비고 |
|------------|------|-----------|------|
| `<GROUNDING>` | Referring expression grounding | ❌ **불가능** | Text input 필요, COCO에 해당 annotation 없음 |
| `<CAPTION_TO_PHRASE_GROUNDING>` | Caption phrase grounding | ❌ **불가능** | Text input 필요 |
| `<REFERRING_EXPRESSION_SEGMENTATION>` | Referring expression segmentation | ❌ **불가능** | Text input 필요 |
| `<REGION_TO_CATEGORY>` | Region to category | ⚠️ **제한적** | Region (bbox) input 필요 |
| `<REGION_TO_DESCRIPTION>` | Region to description | ⚠️ **제한적** | Region (bbox) input 필요 |
| `<REGION_TO_SEGMENTATION>` | Region to segmentation | ⚠️ **제한적** | Region (bbox) input 필요 |
| `<OPEN_VOCABULARY_DETECTION>` | Open vocabulary detection | ❌ **불가능** | Text input 필요 |

**COCO Dataset:**
- COCO는 이미지와 annotation만 제공
- Text input이 필요한 task들은 COCO dataset으로 직접 학습 불가능
- Region-based tasks는 OD 결과를 기반으로 학습 가능하지만, 추가 구현 필요

### 5. OCR Tasks (COCO로 학습 불가능)

| Task Prompt | 설명 | COCO 지원 | 비고 |
|------------|------|-----------|------|
| `<OCR>` | Text recognition | ❌ **불가능** | COCO에 text annotation 없음 |
| `<OCR_WITH_REGION>` | OCR with bounding boxes | ❌ **불가능** | COCO에 text annotation 없음 |

**COCO Dataset:**
- COCO는 scene understanding에 초점 (objects, captions)
- Text/OCR annotation은 제공하지 않음
- OCR 학습을 위해서는 다른 dataset 필요 (예: TextVQA, ICDAR, etc.)

## COCO Dataset으로 학습 가능한 Tasks 요약

### ✅ 직접 학습 가능
1. **Caption** (`<CAPTION>`, `<DETAILED_CAPTION>`, `<MORE_DETAILED_CAPTION>`)
   - Annotation: `captions_train2017.json`
   - 각 이미지당 약 5개 caption

2. **Object Detection** (`<OD>`)
   - Annotation: `instances_train2017.json`
   - 80개 categories, bboxes + segmentation masks

3. **Keypoints** (`<OD>` with keypoints)
   - Annotation: `person_keypoints_train2017.json`
   - Person pose estimation (17 keypoints)

### ✅ 파생 가능 (추가 구현 필요)
4. **Dense Region Captioning** (`<DENSE_REGION_CAPTION>`)
   - OD 결과를 기반으로 각 region에 대한 caption 생성
   - COCO caption annotation 활용 가능

5. **Region Proposal** (`<REGION_PROPOSAL>`)
   - OD 결과를 기반으로 region proposals 생성
   - Category label 없이 bboxes만 제공

### ❌ 학습 불가능
- **Grounding tasks**: Text input 필요, COCO에 해당 annotation 없음
- **OCR tasks**: COCO에 text annotation 없음
- **Open vocabulary detection**: Text input 필요

## Multi-Task Training 전략

### Option 1: COCO에서 제공하는 모든 task 사용
```python
from data import COCOMultiTaskDataset

dataset = COCOMultiTaskDataset(
    data_root="/data4/hongsik/data/COCO",
    tasks=['caption', 'od', 'keypoints']
)
```

### Option 2: Caption variants 추가
```python
# 같은 caption annotation을 다른 task prompt로 사용
# - '<CAPTION>'
# - '<DETAILED_CAPTION>'
# - '<MORE_DETAILED_CAPTION>'
```

### Option 3: Dense Region Captioning 구현
OD 결과를 기반으로 각 detected object에 대한 caption을 생성하는 방식으로 구현 가능

## 필요한 Annotation 파일

Multi-task training을 위해서는:

```
/data4/hongsik/data/COCO/
├── annotations/
│   ├── captions_train2017.json          # Caption tasks
│   ├── instances_train2017.json         # OD, Segmentation
│   └── person_keypoints_train2017.json  # Keypoints (optional)
└── train2017/
    └── *.jpg
```

## 참고

- Florence-2의 모든 task prompt 목록은 HuggingFace model card 참조
- COCO dataset의 annotation 형식은 COCO API 문서 참조
- OCR이나 Grounding task 학습을 위해서는 다른 dataset 필요
