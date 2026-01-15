# Semantic Communication using Florence-2

Florence-2를 백본 네트워크로 사용한 시맨틱 통신 연구 프로젝트입니다.

## 프로젝트 구조

```
SemCom_FL2/
├── transmitter/          # 송신기 모듈
│   ├── __init__.py
│   └── transmitter.py
├── receiver/            # 수신기 모듈
│   ├── __init__.py
│   └── receiver.py
├── channel/             # 채널 모델
│   ├── __init__.py
│   └── channel.py
├── models/              # 모델 정의
│   ├── __init__.py
│   └── florence2_model.py
├── shared/              # 공유 모듈
│   ├── __init__.py
│   ├── task_embedding.py
│   └── csi.py
├── utils/               # 유틸리티 함수
│   ├── __init__.py
│   └── image_utils.py
├── main.py              # 메인 스크립트
├── requirements.txt     # 의존성 패키지
└── README.md           # 이 파일
```

## 시스템 개요

### 공유 정보
1. **Task Prompt**: Task prompt 문자열 (Florence-2 내부에서 task embedding으로 변환)
2. **CSI (Channel State Information)**: Effective SNR 정보

### Transmitter (송신기)
1. 이미지를 입력으로 받음
2. Florence-2의 vision encoder (DaViT)로 vision embedding 생성
   - 옵션: Task embedding과 dimension을 맞추는 linear embedding 포함
3. Vision embedding을 채널을 통해 전송

**출력**: Vision embedding (Image Encoder output)

### Channel (채널)
- **Noiseless**: 잡음 없는 완벽한 채널
- **AWGN**: Additive White Gaussian Noise
- **Rayleigh**: Rayleigh 페이딩 + AWGN

### Receiver (수신기)
1. 채널을 통과한 vision embedding을 수신
2. Task prompt를 Florence-2 processor로 task embedding으로 변환
3. Vision embedding + Task embedding 결합
4. Florence-2의 transformer encoder를 통과
5. Florence-2의 transformer decoder를 통과하여 최종 출력 생성

**처리 과정**: Received Vision Embedding + Task Embedding → Transformer Encoder → Transformer Decoder → Output

## DaViT 출력 차원

Florence-2의 vision encoder (DaViT) 출력 차원:
- **Florence-2-base**: 768 차원
- **Florence-2-large**: 1024 차원

출력 형태:
- **Full sequence**: `(batch_size, num_patches, vision_dim)`
  - 224x224 이미지, patch size 16인 경우: `(batch_size, 196, vision_dim)`
- **Pooled features (CLS token)**: `(batch_size, vision_dim)`

## 설치

```bash
pip install -r requirements.txt
```

## 사용법

### 기본 사용

```bash
python main.py
```

### 옵션 포함

```bash
# Linear embedding 포함, pooled features 사용, AWGN 채널, SNR 20dB
python main.py \
    --include-linear-embedding \
    --use-pooled-features \
    --channel-type awgn \
    --snr-db 20.0
```

### 이미지 입력

```bash
python main.py --image-path /path/to/image.jpg
```

### Task prompt 지정

```bash
python main.py --task-prompt "What does the image describe?"
```

### 모든 옵션

```bash
python main.py \
    --model-name microsoft/Florence-2-base \
    --model-size base \
    --task-prompt "What does the image describe?" \
    --task-embedding-dim 768 \
    --include-linear-embedding \
    --use-pooled-features \
    --channel-type rayleigh \
    --snr-db 15.0 \
    --image-path /path/to/image.jpg \
    --image-size 224 \
    --batch-size 1
```

## 주요 인자

- `--task-prompt`: Task prompt 문자열 (예: "What does the image describe?")
- `--include-linear-embedding`: Task embedding과 dimension을 맞추는 linear embedding 포함 여부
- `--use-pooled-features`: Full sequence 대신 pooled features (CLS token) 사용
- `--channel-type`: 채널 타입 (noiseless, awgn, rayleigh)
- `--snr-db`: Effective SNR (dB)
- `--task-embedding-dim`: Task embedding 차원 (linear embedding 옵션 사용 시, 기본값: 768)

## 모듈 설명

### Transmitter
- Florence-2 vision encoder (DaViT)를 사용하여 이미지를 vision embedding으로 변환
- 옵션으로 linear embedding을 통해 task embedding dimension과 맞춤
- 출력: Vision embedding (Image Encoder output)

### Channel
- 세 가지 채널 모델 지원 (Noiseless, AWGN, Rayleigh)
- CSI를 기반으로 SNR에 따른 잡음 추가

### Receiver
- 수신된 vision embedding과 task prompt를 결합
- Task prompt는 Florence-2 processor를 통해 task embedding으로 변환
- Vision embedding + Task embedding → Transformer Encoder → Transformer Decoder
- Florence-2의 transformer encoder와 decoder를 통과하여 최종 출력 생성

## 참고사항

- Florence-2 모델은 HuggingFace에서 자동으로 다운로드됩니다.
- 첫 실행 시 모델 다운로드로 인해 시간이 걸릴 수 있습니다.
- GPU 사용을 권장합니다 (CUDA 사용 가능 시 자동 사용).
