# Semantic Communication using Florence-2

Florence-2를 백본 네트워크로 사용한 시맨틱 통신 연구 프로젝트입니다.

## 프로젝트 구조

```
SemCom_FL2/
├── transmitter/          # 송신기 모듈
│   ├── __init__.py
│   └── transmitter.py   # Vision tower까지 처리 (두 가지 mode 지원)
├── receiver/            # 수신기 모듈
│   ├── __init__.py
│   └── receiver.py      # Vision tower 이후 처리 (두 가지 mode 지원)
├── channel/             # 채널 모델
│   ├── __init__.py
│   └── channel.py       # Noiseless, AWGN, Rayleigh 채널
├── models/              # 모델 정의
│   ├── __init__.py
│   └── florence2_model.py  # Florence-2 모델 래퍼
├── shared/              # 공유 모듈
│   ├── __init__.py
│   └── csi.py           # Channel State Information
├── utils/               # 유틸리티 함수
│   ├── __init__.py
│   └── image_utils.py
├── data/                # 데이터 로더
│   ├── __init__.py
│   └── coco_dataset.py  # COCO Caption Dataset
├── tests/                # 테스트 스크립트
│   ├── __init__.py
│   ├── test_semcom.py   # 전체 파이프라인 테스트
│   ├── test_component_separation.py  # 컴포넌트 분리 검증
│   └── test_dummy_image_embedding.py  # Text embedding 일관성 테스트
├── docs/                # 문서
│   ├── README.md        # 문서 인덱스
│   ├── README_TRAINING.md  # Training 가이드
│   ├── TRAINING_ARCHITECTURE.md  # Training 아키텍처
│   ├── PROJECT_STRUCTURE.md  # 프로젝트 구조 상세
│   ├── assets/          # 이미지 및 기타 리소스
│   └── *.md             # 기타 상세 문서들
├── scripts/             # 유틸리티 스크립트
│   └── download_coco.sh # COCO dataset 다운로드 스크립트
├── main.py              # 메인 스크립트 (inference)
├── train.py             # Training 스크립트
├── requirements.txt     # 의존성 패키지
└── README.md           # 이 파일
```

## 시스템 개요

### 아키텍처: 두 가지 Mode 지원

**Mode 1 (vision_tower)**: 
- Transmitter: Vision Tower까지 처리 (1024차원 출력)
- Receiver: image_pos_embed → temporal_embed → pooling → projection → norm

**Mode 2 (image_proj_norm)**:
- Transmitter: image_proj_norm까지 처리 (768차원 출력)
- Receiver: 바로 merge 및 language_model

**압축 지점** (향후 추가 예정):
- Mode 1: Vision Tower 출력 이후 (1024차원)
- Mode 2: Image Projection Norm 출력 이후 (768차원)

**장점**:
- 더 나은 압축 품질 (1024차원 특징)
- 알고리즘 유연성 (다양한 압축 기법 적용 가능)
- 강건성 (공간 정보 보존)

### 공유 정보

1. **Text Embeddings**: Top level (main.py/test_semcom.py)에서 생성하여 Tx/Rx에 공유
   - All-zero dummy image와 task prompt를 사용하여 processor로 생성
   - Processor는 이미지 내용이 아닌 존재 여부만 확인하므로 dummy image로도 동일한 결과
2. **CSI (Channel State Information)**: Effective SNR 정보

### Transmitter (송신기)

**처리 단계:**
1. 이미지 입력 (PIL Image 또는 Tensor)
2. Processor를 통한 전처리 (resize, normalize)
3. Vision Tower (`vision_tower.forward_features_unpool`)
4. **압축 지점** → Vision Tower 출력 (1024차원)

**출력**: Vision Tower features `(batch_size, seq_len, 1024)`

### Channel (채널)

- **Noiseless**: 잡음 없는 완벽한 채널
- **AWGN**: Additive White Gaussian Noise
- **Rayleigh**: Rayleigh 페이딩 + AWGN

### Receiver (수신기)

**처리 단계:**
1. 수신된 Vision Tower features (1024차원)
2. Image Position Embedding (2D 공간 위치 정보)
3. Visual Temporal Embedding (시간 정보, 비디오용)
4. Image Feature Source Pooling (spatial_avg_pool, temporal_avg_pool, last_frame)
5. Image Projection (1024 → 768차원)
6. Image Projection Normalization
7. Text Embeddings와 병합 (shared from top level)
8. Language Model을 통한 텍스트 생성

**출력**: Merged embeddings `(batch, vision_seq_len + text_seq_len, 768)`

## Text Embedding 공유 구조

Text embeddings는 **top level (main.py 또는 test_semcom.py)**에서 생성하여 Tx와 Rx에 공유됩니다.

```python
# Top level에서 text_embeddings 생성
import numpy as np
from PIL import Image as PILImage

dummy_image = PILImage.fromarray(np.zeros((768, 768, 3), dtype=np.uint8))
inputs = florence2_model.processor(
    text=task_prompts,
    images=[dummy_image] * len(task_prompts),
    return_tensors="pt"
)
text_embeddings = embedding_layer(inputs["input_ids"])

# Receiver에 전달
merged_embeds, attention_mask = receiver(received_signal, text_embeddings)
```

**이유:**
- Processor는 이미지 내용이 아닌 존재 여부만 확인
- Dummy image로도 동일한 text embedding 생성
- 일관성 보장 및 효율성 향상

## 설치

```bash
pip install -r requirements.txt
```

## 사용법

### 기본 사용 (Inference)

```bash
# 기본 실행 (test image URL 사용, noiseless 채널)
python main.py --task_prompt "<CAPTION>" --channel_type noiseless

# Mode 선택
python main.py --mode vision_tower --task_prompt "<CAPTION>"
python main.py --mode image_proj_norm --task_prompt "<CAPTION>"
```

### 옵션 포함

```bash
# AWGN 채널, SNR 20dB
python main.py \
    --channel_type awgn \
    --snr_db 20.0 \
    --task_prompt "<CAPTION>"
```

### 이미지 입력

```bash
python main.py --image_path /path/to/image.jpg --task_prompt "<CAPTION>"
```

### Task prompt 지정

```bash
python main.py --task_prompt "<CAPTION>"
python main.py --task_prompt "<DETAILED_CAPTION>"
python main.py --task_prompt "<OD>"
```

## 테스트

### 전체 파이프라인 테스트

```bash
# 기본 실행 (noiseless 채널)
python tests/test_semcom.py

# AWGN 채널, SNR 10dB
python tests/test_semcom.py --channel_type awgn --snr_db 10.0

# Rayleigh 채널, SNR 15dB
python tests/test_semcom.py --channel_type rayleigh --snr_db 15.0

# Mode 변경 (image_proj_norm)
python tests/test_semcom.py --mode image_proj_norm --channel_type awgn --snr_db 10.0
```

**test_semcom.py 인자:**
- `--channel_type`: 채널 타입 (`noiseless`, `awgn`, `rayleigh`, 기본값: `noiseless`)
- `--snr_db`: Signal-to-noise ratio in dB (기본값: `20.0`)
- `--mode`: Processing mode (`vision_tower` 또는 `image_proj_norm`, 기본값: `vision_tower`)

### 컴포넌트 분리 검증

```bash
python tests/test_component_separation.py
```

### Text Embedding 일관성 테스트

```bash
python tests/test_dummy_image_embedding.py
```

## 주요 인자 (main.py)

- `--mode`: Processing mode (`vision_tower` 또는 `image_proj_norm`)
- `--task_prompt`: Task prompt 문자열 (예: `<CAPTION>`, `<DETAILED_CAPTION>`, `<OD>`)
- `--channel_type`: 채널 타입 (noiseless, awgn, rayleigh)
- `--snr_db`: Effective SNR (dB)
- `--image_path`: 입력 이미지 경로 (없으면 test image URL 사용)
- `--image_size`: 이미지 크기 (H=W, 기본값: 224)
- `--batch_size`: 배치 크기 (기본값: 1)

자세한 training 인자는 `docs/README_TRAINING.md`를 참조하세요.

## 모듈 설명

### Transmitter
- Florence-2 vision encoder (DaViT)를 사용하여 이미지를 vision embedding으로 변환
- Method 2: Vision Tower 출력까지 처리 (1024차원)
- 출력: Vision Tower features `(batch_size, seq_len, 1024)`

### Channel
- 세 가지 채널 모델 지원 (Noiseless, AWGN, Rayleigh)
- CSI를 기반으로 SNR에 따른 잡음 추가

### Receiver
- 수신된 vision tower features 처리
- Image position embedding, temporal embedding, pooling, projection, normalization
- Text embeddings (shared from top level)와 병합
- Language model을 통한 최종 텍스트 생성

## Training

COCO dataset을 사용한 training이 지원됩니다.

### 빠른 시작
```bash
# COCO dataset 다운로드 (선택사항)
bash scripts/download_coco.sh

# Training 실행
python train.py --data_root /data4/hongsik/data/COCO --batch_size 4 --num_epochs 10
```

자세한 내용은 `docs/README_TRAINING.md`를 참조하세요.

## 문서

자세한 내용은 `docs/` 폴더를 참조하세요:
- `TEST_GUIDE.md`: 테스트 스크립트 사용 가이드 (test_semcom.py 인자 포함)
- `README_TRAINING.md`: Training 가이드
- `TRAINING_ARCHITECTURE.md`: Training 아키텍처 (Frozen vs Trainable)
- `COMPRESSION_POINT_COMPARISON.md`: 압축 지점 비교 분석
- `tokenization_difference_summary.md`: Tokenization 차이 요약
- `TEXT_EMBEDDING_SHARING_UPDATE.md`: Text embedding 공유 구조 설명
- `PROJECT_STRUCTURE.md`: 프로젝트 구조 상세 설명
- 기타 상세 문서들

## 참고사항

- Florence-2 모델은 HuggingFace에서 자동으로 다운로드됩니다.
- 첫 실행 시 모델 다운로드로 인해 시간이 걸릴 수 있습니다.
- GPU 사용을 권장합니다 (CUDA 사용 가능 시 자동 사용).
- Text embeddings는 top level에서 생성하여 Tx/Rx에 공유됩니다.

## 라이선스

이 프로젝트는 연구 목적으로 개발되었습니다.
