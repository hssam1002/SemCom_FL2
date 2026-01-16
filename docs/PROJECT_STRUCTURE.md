# 프로젝트 구조 (Project Structure)

## 최종 정리된 프로젝트 구조

```
SemCom_FL2/
├── transmitter/              # 송신기 모듈
│   ├── __init__.py          # 모듈 초기화 및 설명
│   └── transmitter.py       # Transmitter 클래스 (Method 2: Vision Tower까지 처리)
│
├── receiver/                 # 수신기 모듈
│   ├── __init__.py          # 모듈 초기화 및 설명
│   └── receiver.py          # Receiver 클래스 (Vision Tower 이후 처리)
│
├── channel/                  # 채널 모델
│   ├── __init__.py          # 모듈 초기화 및 설명
│   └── channel.py           # Channel 클래스 (Noiseless, AWGN, Rayleigh)
│
├── models/                   # 모델 정의
│   ├── __init__.py          # 모듈 초기화 및 설명
│   └── florence2_model.py   # Florence2Model 래퍼 클래스
│
├── shared/                    # 공유 모듈
│   ├── __init__.py          # 모듈 초기화 및 설명
│   ├── csi.py               # Channel State Information
│   └── task_embedding.py    # Task Embedding (호환성용)
│
├── utils/                     # 유틸리티 함수
│   ├── __init__.py          # 모듈 초기화 및 설명
│   └── image_utils.py       # 이미지 처리 유틸리티
│
├── tests/                     # 테스트 스크립트
│   ├── __init__.py          # 테스트 모듈 설명
│   ├── test_semcom.py       # 전체 파이프라인 테스트
│   ├── test_component_separation.py  # 컴포넌트 분리 검증
│   └── test_dummy_image_embedding.py # Text embedding 일관성 테스트
│
├── docs/                      # 문서
│   ├── README.md             # 문서 인덱스
│   ├── COMPRESSION_POINT_COMPARISON.md
│   ├── tokenization_analysis.md
│   ├── tokenization_difference_explanation.md
│   ├── tokenization_difference_summary.md
│   ├── dummy_image_text_embedding_result.md
│   ├── TEXT_EMBEDDING_SHARING_UPDATE.md
│   └── tx_rx_separation_analysis.md
│
├── main.py                    # 메인 스크립트
├── requirements.txt           # 의존성 패키지
├── README.md                  # 프로젝트 메인 README
└── PROJECT_STRUCTURE.md       # 이 파일
```

## 파일 설명

### 핵심 모듈

#### transmitter/
- **transmitter.py**: 
  - Method 2 아키텍처 구현
  - Vision Tower까지 처리 (1024차원 출력)
  - 압축 지점: Vision Tower 출력 이후

#### receiver/
- **receiver.py**: 
  - Vision Tower 이후 처리
  - pos_embed, temporal_embed, pooling, projection, norm
  - Text embeddings와 병합
  - Language model을 통한 텍스트 생성

#### channel/
- **channel.py**: 
  - Noiseless, AWGN, Rayleigh 채널 구현
  - CSI 기반 잡음 생성

### 모델 및 유틸리티

#### models/
- **florence2_model.py**: 
  - Florence-2 모델 래퍼
  - HuggingFace 모델 로딩 및 관리
  - Vision tower, language model 접근

#### shared/
- **csi.py**: Channel State Information (SNR, 채널 타입)
- **task_embedding.py**: Task embedding 모듈 (호환성용)

#### utils/
- **image_utils.py**: 이미지 로딩 및 전처리 함수

### 테스트

#### tests/
- **test_semcom.py**: 전체 semantic communication 파이프라인 테스트
- **test_component_separation.py**: 컴포넌트 분리 검증 (reference)
- **test_dummy_image_embedding.py**: Text embedding 일관성 검증

### 문서

#### docs/
- 모든 상세 문서 및 분석 자료
- 아키텍처, tokenization, text embedding 관련 문서

## 정리된 내용

### 삭제된 파일
- `debug_receiver_step_by_step.py` (디버깅용)
- `test_receiver_actual_call.py` (디버깅용)
- `test_receiver_comparison.py` (비교용)
- `test_rx_detailed_comparison.py` (비교용)
- `test_tx_rx_comparison.py` (비교용)
- `tx_rx_comparison_implementation.py` (비교용)
- `test_image_pos_embed.py` (테스트용)
- `test_florence2.py` (기본 테스트)

### 이동된 파일
- 모든 MD 문서 → `docs/` 폴더
- 테스트 파일 → `tests/` 폴더

### 개선 사항
- 모든 `__init__.py`에 상세한 모듈 설명 추가
- README 업데이트 (최신 구조 반영)
- 코드 주석 정리 및 보강
- 폴더 구조 명확화

## 사용 방법

### 메인 스크립트 실행
```bash
python main.py --task-prompt "<CAPTION>" --channel-type awgn --snr-db 20.0
```

### 테스트 실행
```bash
# 전체 파이프라인 테스트 (기본: noiseless 채널)
python tests/test_semcom.py

# AWGN 채널, SNR 10dB
python tests/test_semcom.py --channel_type awgn --snr_db 10.0

# Rayleigh 채널, SNR 15dB
python tests/test_semcom.py --channel_type rayleigh --snr_db 15.0

# Mode 변경 (image_proj_norm)
python tests/test_semcom.py --mode image_proj_norm --channel_type awgn --snr_db 10.0

# 컴포넌트 분리 검증
python tests/test_component_separation.py

# Text embedding 일관성 테스트
python tests/test_dummy_image_embedding.py
```

**test_semcom.py 인자:**
- `--channel_type`: 채널 타입 (`noiseless`, `awgn`, `rayleigh`, 기본값: `noiseless`)
- `--snr_db`: Signal-to-noise ratio in dB (기본값: `20.0`)
- `--mode`: Processing mode (`vision_tower` 또는 `image_proj_norm`, 기본값: `vision_tower`)

## 참고

- 프로젝트의 메인 README는 `README.md`를 참조하세요.
- 상세 문서는 `docs/` 폴더를 참조하세요.
