# Tx/Rx 분리점 선택 분석: Compression 관점

## 현재 상황
- **Florence-2 base 모델**: Pre-trained weights 로드 완료
- **목표**: Tx에서 compression을 적용할 최적의 분리점 선택

## 두 가지 방법 비교

### 방법 1: image_proj_norm 이후 Compression (현재 구조)

**Transmitter (Tx):**
1. Image processing (preprocessing)
2. Vision Tower
3. image_pos_embed
4. visual_temporal_embed
5. image_feature_source_pooling
6. image_projection
7. image_proj_norm
8. **→ Compression 여기서 적용**

**Receiver (Rx):**
9. language_model
10. Decoding (batch_decode, post_process_generation)

#### 특징 분석:

| 항목 | 값 | 비고 |
|------|-----|------|
| **Output 크기** | (batch, ~577, 768) | Sequence length: 577 (spatial_avg_pool + temporal_avg_pool) |
| **Hidden dimension** | 768 | 언어 모델 차원과 일치 |
| **총 요소 수** | batch × 577 × 768 ≈ 443,000 per batch | |
| **정보 수준** | High-level semantic features | 이미 집계되고 정규화됨 |
| **공간 정보** | 일부 손실 (pooling 이후) | Spatial structure 정보 감소 |

**장점:**
- ✅ 이미 언어 모델 차원(768)으로 맞춰짐 → compression 후에도 Rx에서 바로 사용 가능
- ✅ Feature source pooling으로 이미 일부 압축 효과 (spatial 정보 요약)
- ✅ 이미 정규화되어 있어 compression 알고리즘에 안정적
- ✅ Tx에서 계산 부담 적음 (pooling, projection, norm은 가벼움)
- ✅ Task-agnostic features → 다양한 task에 재사용 가능

**단점:**
- ❌ 이미 집계된 정보라 detail 손실 가능
- ❌ Positional/temporal 정보 이미 추가되어 compression 후 복구 어려움
- ❌ Rx에서 feature source 선택 조정 불가능 (이미 pooling 완료)
- ❌ Spatial structure 정보 손실로 adaptive compression 제한적

---

### 방법 2: vision_tower 이후 Compression

**Transmitter (Tx):**
1. Image processing (preprocessing)
2. Vision Tower
3. **→ Compression 여기서 적용**

**Receiver (Rx):**
4. image_pos_embed
5. visual_temporal_embed
6. image_feature_source_pooling
7. image_projection
8. image_proj_norm
9. language_model
10. Decoding

#### 특징 분석:

| 항목 | 값 | 비고 |
|------|-----|------|
| **Output 크기** | (batch, 576, 1024) | Sequence length: 576 patches |
| **Hidden dimension** | 1024 | Vision encoder 원본 차원 |
| **총 요소 수** | batch × 576 × 1024 ≈ 590,000 per batch | 방법 1보다 ~33% 큼 |
| **정보 수준** | Mid-level visual features | 원시 시각적 특징 |
| **공간 정보** | 완전 보존 | 24×24 spatial grid 구조 유지 |

**장점:**
- ✅ **원시 시각적 특징** → 더 많은 정보 보존, compression 복원 품질 향상 가능
- ✅ **Spatial structure 완전 보존** → Adaptive/spatial-aware compression 활용 가능
- ✅ **Flexibility 높음** → Rx에서 task에 따라 feature source pooling 선택 가능
- ✅ **Positional/temporal embedding을 Rx에서 처리** → Compression noise에 더 robust
- ✅ **Compression algorithm 선택 폭 넓음** → Spatial-aware methods (VQ-VAE, patch-wise compression 등) 활용 가능

**단점:**
- ❌ 데이터 크기 더 큼 (1024 vs 768, 약 33% 증가) → Compression 비율이 더 커야 함
- ❌ Rx에 더 많은 계산 부담 (pooling, projection, norm 모두 Rx에서)
- ❌ Compression 후 dimension mismatch 처리 필요 (1024 → 768 projection)

---

## Compression 관점에서의 추천: **방법 2 (Vision Tower 이후)**

### 추천 이유:

#### 1. **Compression 복원 품질**
```
Vision Tower output = 원시 시각적 특징
→ Spatial structure 정보 완전 보존
→ Compression 후에도 공간 정보 활용 가능
→ 더 나은 복원 품질 기대
```

#### 2. **Compression 알고리즘 선택의 폭**
```
- Patch-wise compression: 24×24 spatial grid 구조 활용
- Spatial-aware methods: VQ-VAE, PatchGAN 등
- Adaptive compression: 중요 패치에 더 높은 비트 할당 가능
- Transformer-based compression: Spatial attention 활용
```

#### 3. **Robustness**
```
Positional/temporal embedding을 Rx에서 처리
→ Compression noise가 positional 정보에 직접 영향 적음
→ Rx에서 embedding 재계산 가능
```

#### 4. **Flexibility & Adaptability**
```
Rx에서 task에 따라 feature source 선택 가능:
- High-detail task → last_frame 사용
- Summarization task → spatial_avg_pool 사용
- Dynamic task adaptation 가능
```

#### 5. **정보 보존**
```
Vision Tower output: 원시 특징 (576, 1024)
→ 더 많은 정보 보존
→ Compression loss에 더 robust
```

---

## 실용적 고려사항

### 방법 1이 더 나은 경우:
- ✅ **Bandwidth 매우 제한적** → 이미 압축된 768-dim이 더 작음
- ✅ **Rx computation 자원 매우 제한적** → Tx에서 미리 처리
- ✅ **Simple compression** → 이미 정규화된 데이터가 더 쉽게 압축
- ✅ **Task가 고정** → Feature source 선택 불필요

### 방법 2가 더 나은 경우:
- ✅ **Compression 품질이 중요** → 원시 특징이 복원 품질 향상
- ✅ **Spatial-aware compression 사용** → Grid structure 활용
- ✅ **Rx에 충분한 자원** → 추가 계산 가능
- ✅ **Multiple tasks 지원** → Flexible feature source 선택
- ✅ **Research/Experiment 목적** → 다양한 compression 방법 시도

---

## 최종 추천: **방법 2 (Vision Tower 이후 Compression)**

### 이유 요약:
1. **Compression 복원 품질**: 원시 특징이 더 나은 복원 가능
2. **알고리즘 선택 폭**: Spatial-aware compression 활용 가능
3. **Robustness**: Positional embedding을 Rx에서 처리하여 더 robust
4. **Flexibility**: Task에 따른 feature source 선택 가능
5. **미래 확장성**: 다양한 compression 연구에 적합

### Trade-off:
- **데이터 크기 증가** (~33%): 더 강력한 compression 필요
- **Rx 계산 증가**: 하지만 modern compression (학습된 모델)은 보통 가벼움

---

## 구현 제안

### 방법 2 구현 구조:

```python
# Transmitter
class Transmitter(nn.Module):
    def forward(self, images):
        # 1. Image preprocessing
        pixel_values = self.processor(images)
        
        # 2. Vision Tower
        vision_features = self.vision_tower.forward_features_unpool(pixel_values)
        # Output: (batch, 576, 1024)
        
        # 3. Compression (여기서 적용!)
        compressed_features = self.compressor(vision_features)
        
        return compressed_features

# Receiver  
class Receiver(nn.Module):
    def forward(self, compressed_features, task_prompts):
        # 1. Decompression
        vision_features = self.decompressor(compressed_features)
        
        # 2. image_pos_embed
        vision_features = self.apply_pos_embed(vision_features)
        
        # 3. visual_temporal_embed
        vision_features = self.apply_temporal_embed(vision_features)
        
        # 4. image_feature_source_pooling
        vision_features = self.apply_feature_pooling(vision_features)
        
        # 5. image_projection
        vision_features = vision_features @ self.image_projection
        
        # 6. image_proj_norm
        vision_features = self.image_proj_norm(vision_features)
        
        # 7. Language model
        output = self.language_model(vision_features, task_prompts)
        
        return output
```

---

## Compression 전략 예시 (방법 2)

### Option 1: Learned Compression (추천)
- VQ-VAE 기반 codebook 학습
- Spatial structure 활용한 patch-wise quantization
- Transformer-based rate-distortion optimization

### Option 2: Traditional + Neural Hybrid
- PCA/SVD로 차원 축소 (1024 → 512)
- Learned quantization
- Rx에서 projection으로 복원

### Option 3: Attention-based Adaptive
- Spatial attention으로 중요 패치 선택
- 중요 패치: 높은 비트, 나머지: 낮은 비트
- Rx에서 adaptive reconstruction

---

## 결론

**방법 2 (Vision Tower 이후)**를 추천합니다. 

Compression 품질, 알고리즘 선택의 폭, robustness, flexibility 측면에서 우수하며, 
semantic communication 연구에 더 적합합니다.

다만 bandwidth가 극도로 제한적이거나 Rx 자원이 매우 부족한 경우에는 방법 1도 고려할 수 있습니다.
