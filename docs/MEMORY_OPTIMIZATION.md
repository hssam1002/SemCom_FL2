# 메모리 최적화 가이드 (Memory Optimization Guide)

GPU 메모리가 제한적인 환경에서 학습을 위한 메모리 최적화 전략입니다.

## 구조: Tx (frozen) -> Tx(trainable) -> Channel -> Rx(trainable) -> Rx(frozen) -> output

```
Image → Tx_frozen (Florence-2 vision_tower) → Tx_trainable (compression) 
→ Channel → Rx_trainable (decompression) → Rx_frozen (Florence-2 modules) → output → Loss
```

## 핵심 답변: Frozen 부분만 no_grad() 사용 가능

**네, Frozen 부분들만 `no_grad()`로 해도 됩니다!**

### ✅ 사용 가능 (메모리 절약)

1. **Tx(frozen) 내부**: Florence-2 vision_tower
   - `requires_grad=False`로 설정되어 있어 gradient가 계산되지 않음
   - `no_grad()` 사용 시 computation graph를 만들지 않아 메모리 절약

2. **Rx(frozen) 내부**: Florence-2 modules (image_pos_embed, image_proj_norm, language_model)
   - Frozen 모듈이므로 `no_grad()` 사용 가능
   - 메모리 절약 효과

3. **Text embedding generation**: Processor, embedding layer
   - Frozen 모듈이므로 `no_grad()` 사용 가능

### ❌ 사용 불가 (Gradient 필요)

1. **Tx(trainable)**: Compression module
2. **Rx(trainable)**: Decompression module  
3. **Loss computation**: Output과 label 사이

## 구현 방법: Module 내부에서 selective no_grad() 사용 (현재 구현)

Transmitter와 Receiver 내부에서 frozen 부분에만 `no_grad()`를 사용하도록 구현했습니다:

### Transmitter (transmitter.py)

```python
def forward(self, images):
    # Frozen Florence-2 vision_tower: use no_grad() for memory efficiency
    with torch.no_grad():
        vision_output = model.vision_tower.forward_features_unpool(pixel_values)
    vision_features = vision_output[0] if isinstance(vision_output, tuple) else vision_output
    
    # Future: Trainable compression module will be inserted here (NO no_grad())
    # vision_features = self.compression_module(vision_features)  # Trainable
    
    return vision_features
```

### Receiver (receiver.py)

```python
def forward(self, received_vision_embedding, text_embeddings):
    vision_features = received_vision_embedding
    
    # Future: Trainable decompression module will be inserted here (NO no_grad())
    # vision_features = self.decompression_module(received_vision_embedding)  # Trainable
    
    if self.mode == 'vision_tower':
        # Frozen Florence-2 modules: use no_grad() for memory efficiency
        with torch.no_grad():
            # image_pos_embed, visual_temporal_embed, pooling, projection, norm
            vision_features = process_frozen_modules(vision_features)
    
    # Frozen merge operation: use no_grad()
    with torch.no_grad():
        merged_embeds = model._merge_input_ids_with_image_features(vision_features, text_embeddings)
    
    return merged_embeds, attention_mask
```

### Train Loop (train.py)

```python
# Text embedding generation (frozen) - use no_grad()
with torch.no_grad():
    text_embeddings = embedding_layer(input_ids_task)

# Transmitter forward (frozen part uses no_grad() internally)
tx_output = transmitter(images)  # no_grad() is inside transmitter.forward()

# Channel (typically no_grad() - deterministic noise)
received_signal = channel(tx_output)

# Receiver forward (frozen part uses no_grad() internally)
merged_embeds, attention_mask = receiver(received_signal, text_embeddings)

# Language model forward (frozen) - use no_grad()
with torch.no_grad():
    outputs = language_model(inputs_embeds=merged_embeds_gt, attention_mask=attention_mask_gt)
    logits = outputs.logits

# Loss computation (NO no_grad() - needed for gradient flow to trainable modules)
loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
loss.backward()  # Gradients flow to trainable modules (compression/decompression)
```

## 주의사항: Gradient Flow 유지

### ⚠️ Frozen → Trainable 경계

**Frozen 모듈의 output을 trainable 모듈로 전달할 때는 `no_grad()` 밖에서 연결:**

```python
# ✅ 올바른 방법
with torch.no_grad():
    frozen_output = frozen_module(input)  # Frozen - no_grad() 사용
# no_grad() 블록 밖으로 나옴
trainable_output = trainable_module(frozen_output)  # Trainable - gradient flow 가능

# ❌ 잘못된 방법
with torch.no_grad():
    frozen_output = frozen_module(input)
    trainable_output = trainable_module(frozen_output)  # Gradient 없음!
```

### ⚠️ Trainable → Frozen 경계

**Trainable 모듈의 output을 frozen 모듈로 전달할 때는 `no_grad()` 밖에서 연결 후 frozen 모듈에만 `no_grad()` 사용:**

```python
# ✅ 올바른 방법
trainable_output = trainable_module(input)  # Trainable - NO no_grad()
# no_grad() 블록 안에서 frozen 모듈만 처리
with torch.no_grad():
    frozen_output = frozen_module(trainable_output)  # Frozen - no_grad() 사용

# ❌ 잘못된 방법
trainable_output = trainable_module(input)
# trainable_output에 no_grad()를 적용하면 안 됨
```

### ⚠️ detach() 사용 금지

- `detach()`는 gradient flow를 완전히 끊음
- Trainable module에는 절대 사용하지 말 것
- `requires_grad=False`만으로 충분

```python
# ❌ 잘못된 방법
frozen_output = frozen_module(input).detach()
trainable_output = trainable_module(frozen_output)  # Gradient 없음!

# ✅ 올바른 방법 (requires_grad=False만으로 충분)
frozen_output = frozen_module(input)  # requires_grad=False이므로 자동으로 gradient 없음
trainable_output = trainable_module(frozen_output)  # Gradient 있음
```

## 메모리 절약 효과

### no_grad() 사용 시 메모리 절약
- **Computation graph 미생성**: Frozen module forward pass의 중간 activation 저장 안 함
- **메모리 절약**: 30-50% 정도 절약 가능 (모델 크기에 따라 다름)
- **속도 향상**: Graph 생성 오버헤드 제거

### 예상 메모리 사용량
- **Without no_grad()**: ~100% (baseline)
- **With no_grad() on frozen parts**: ~60-70% (30-40% 절약)
- **Full no_grad()** (trainable도 포함): ~40-50% (하지만 학습 불가능)

## 현재 구현 상태

✅ **구현 완료:**
1. **Transmitter 내부**: vision_tower forward에 `no_grad()` 적용
2. **Receiver 내부**: 
   - Mode 1 (vision_tower): image_pos_embed ~ image_proj_norm에 `no_grad()` 적용
   - Text embedding generation 및 merge에 `no_grad()` 적용
3. **Train loop**: 
   - Text embedding generation에 `no_grad()` 적용
   - Language model forward에 `no_grad()` 적용
   - Loss computation은 `no_grad()` 없음 (trainable module을 위한 gradient flow)

## Future: Trainable Module 추가 시

Trainable module (compression/decompression)을 추가할 때:
1. Module 정의 시 `requires_grad=True` (기본값)
2. Transmitter/Receiver의 forward에서 trainable module 호출 시 **`no_grad()` 사용 안 함**
3. Gradient는 자동으로 trainable module로 flow됨

```python
# Future implementation example:
class Transmitter(nn.Module):
    def forward(self, images):
        # Frozen part - use no_grad()
        with torch.no_grad():
            vision_features = self.florence2_model.vision_tower(images)
        
        # Trainable part - NO no_grad()
        compressed_features = self.compression_module(vision_features)  # Gradient flow!
        
        return compressed_features
```

이렇게 하면 메모리를 절약하면서도 trainable module에 gradient가 정상적으로 flow됩니다.
