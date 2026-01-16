# 테스트 가이드 (Test Guide)

이 문서는 프로젝트의 테스트 스크립트 사용법을 설명합니다.

## test_semcom.py

전체 semantic communication 파이프라인을 테스트하는 스크립트입니다.
Transmitter → Channel → Receiver 파이프라인을 통해 이미지를 처리하고,
Reference 모델과 결과를 비교합니다.

### 기본 사용법

```bash
# 기본 실행 (noiseless 채널, mode: vision_tower)
python tests/test_semcom.py
```

### Command Line Arguments

| 인자 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `--channel_type` | str | `noiseless` | 채널 타입: `noiseless`, `awgn`, `rayleigh` |
| `--snr_db` | float | `20.0` | Signal-to-noise ratio (dB) |
| `--mode` | str | `vision_tower` | Processing mode: `vision_tower` 또는 `image_proj_norm` |

### 사용 예제

#### 1. Noiseless 채널 (기본)
```bash
python tests/test_semcom.py
```

#### 2. AWGN 채널, SNR 10dB
```bash
python tests/test_semcom.py --channel_type awgn --snr_db 10.0
```

#### 3. AWGN 채널, SNR 5dB (낮은 SNR)
```bash
python tests/test_semcom.py --channel_type awgn --snr_db 5.0
```

#### 4. Rayleigh 채널, SNR 15dB
```bash
python tests/test_semcom.py --channel_type rayleigh --snr_db 15.0
```

#### 5. Mode 변경 (image_proj_norm)
```bash
python tests/test_semcom.py --mode image_proj_norm --channel_type awgn --snr_db 10.0
```

#### 6. 다양한 조합
```bash
# Mode 2, AWGN, 낮은 SNR
python tests/test_semcom.py --mode image_proj_norm --channel_type awgn --snr_db 5.0

# Mode 1, Rayleigh, 높은 SNR
python tests/test_semcom.py --mode vision_tower --channel_type rayleigh --snr_db 20.0
```

### 출력 내용

스크립트는 다음 정보를 출력합니다:

1. **파이프라인 초기화**: Transmitter, Channel, Receiver 초기화 상태
2. **신호 전력 정보**: 
   - Transmitted signal power
   - Received signal power
   - Noise power (noiseless가 아닌 경우)
   - Actual SNR (noiseless가 아닌 경우)
3. **Task별 결과 비교**:
   - `<CAPTION>`: 이미지 캡션 생성
   - `<DETAILED_CAPTION>`: 상세 캡션 생성
   - `<OD>`: 객체 탐지
4. **Receiver vs Reference 비교**: 각 task에 대해 결과 일치 여부 확인

### 예상 출력

```
======================================================================
Semantic Communication Pipeline Test
======================================================================

[1/5] Loading test image...
✓ Image loaded: (640, 480)

[2/5] Initializing Florence-2 model...
✓ Florence-2 model loaded

[3/5] Initializing Transmitter...
  Mode: vision_tower
✓ Transmitter initialized

[4/5] Initializing Channel...
✓ Channel initialized (AWGN, SNR: 10.0 dB)

[5/5] Initializing Receiver...
✓ Receiver initialized

======================================================================
Processing Pipeline
======================================================================

[Step 1] Transmitter: Encoding image to vision embedding...
✓ Transmitter output shape: torch.Size([1, 576, 1024])

[Step 3] Channel: Transmitting through awgn channel (SNR: 10.0 dB)...
✓ Received signal shape: torch.Size([1, 576, 1024])
  Transmitted power: 0.502441
  Received power: 0.553223
  Noise power: 0.050781
  Actual SNR: 9.95 dB

[Step 4] Receiver: Processing received signal with task prompts...

--- Testing Caption (<CAPTION>) ---
  [Receiver Result]
    <CAPTION>: A green car parked in front of a yellow building.

  [Reference Result]
    <CAPTION>: A green car parked in front of a yellow building.

  [Comparison]
  ✓ Results match! Semantic communication pipeline works correctly.

======================================================================
Test Summary
======================================================================
✓ Full semantic communication pipeline tested successfully!
```

### 주의사항

1. **SNR이 낮을수록**: 노이즈가 증가하여 Receiver와 Reference 결과의 차이가 커질 수 있습니다.
2. **Mode 차이**: 
   - `vision_tower`: Transmitter가 vision_tower 출력까지 처리 (1024차원)
   - `image_proj_norm`: Transmitter가 image_proj_norm까지 처리 (768차원)
3. **채널 타입**:
   - `noiseless`: 완벽한 채널 (노이즈 없음)
   - `awgn`: Additive White Gaussian Noise
   - `rayleigh`: Rayleigh 페이딩 + AWGN

## test_component_separation.py

컴포넌트 분리 검증 테스트입니다. 전체 모델과 분리된 컴포넌트의 결과가 일치하는지 확인합니다.

```bash
python tests/test_component_separation.py
```

## test_dummy_image_embedding.py

Text embedding 일관성 테스트입니다. Dummy image를 사용한 text embedding이 실제 이미지와 동일한지 확인합니다.

```bash
python tests/test_dummy_image_embedding.py
```
