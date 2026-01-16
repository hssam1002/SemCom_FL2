# Dummy Image vs Real Image: Text Embedding 동일성 결과

## 질문
**어떠한 image든 상관없이 (dummy image를 넣어도) text embedding 결과는 동일한가?**

## 답변
**✅ 네, 맞습니다!**

## 테스트 결과

### 실험 조건
- **Task Prompt**: `<CAPTION>`
- **이미지 종류**:
  1. Real Image (실제 이미지)
  2. Dummy Image (zeros, 768x768)
  3. Dummy Image (random, 768x768)
  4. Dummy Image (zeros, 512x512)
  5. Dummy Image (zeros, 224x224)

### 결과

```
모든 경우에 동일한 Input IDs 생성:
[0, 2264, 473, 5, 2274, 6190, 116, 2]

Decoded text: "<s>What does the image describe?</s>"
Token count: 8
```

### 비교 결과

| 비교 항목 | 결과 |
|----------|------|
| Real vs Dummy(zeros, 768x768) | ✅ **동일** |
| Real vs Dummy(random, 768x768) | ✅ **동일** |
| Real vs Dummy(512x512) | ✅ **동일** |
| Real vs Dummy(224x224) | ✅ **동일** |

### 다른 Task Prompts 테스트

| Task Prompt | Real vs Dummy | Tokens |
|-------------|---------------|--------|
| `<CAPTION>` | ✅ **동일** | 8 |
| `<DETAILED_CAPTION>` | ✅ **동일** | 13 |
| `<OD>` | ✅ **동일** | 13 |

## 결론

### 핵심 발견

1. **Processor는 이미지 내용을 보지 않습니다**
   - 이미지의 실제 픽셀 값은 무관
   - 이미지 크기도 무관 (224x224, 512x512, 768x768 모두 동일)
   - zeros든 random이든 결과는 동일

2. **Processor는 이미지 존재 여부만 확인합니다**
   - `processor(text, images=image)` 형태로 호출되면
   - 이미지가 제공되었는지만 확인
   - 제공되면 task prompt를 자연어 질문으로 변환

3. **Text Embedding은 Input IDs에만 의존**
   - Input IDs가 같으면 Text Embedding도 동일
   - 따라서 어떤 이미지를 넣든 Text Embedding은 동일

## 실용적 의미

### Dummy Image 사용의 정당성

**✅ Dummy Image를 사용해도 완전히 안전합니다!**

- 어떤 이미지를 넣든 상관없음
- Text Embedding 결과는 항상 동일
- Processor는 이미지 존재 여부만 확인

### 권장사항

```python
# ✅ 이렇게 해도 됨 (어떤 이미지든 OK)
dummy_image = Image.fromarray(np.zeros((768, 768, 3), dtype=np.uint8))
inputs = processor(text='<CAPTION>', images=dummy_image, return_tensors='pt')
text_embeddings = model.get_input_embeddings()(inputs['input_ids'])

# ✅ 이것도 됨 (더 작은 크기)
dummy_image = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
inputs = processor(text='<CAPTION>', images=dummy_image, return_tensors='pt')
text_embeddings = model.get_input_embeddings()(inputs['input_ids'])

# ✅ 이것도 됨 (random 값)
dummy_image = Image.fromarray(np.random.randint(0, 255, (768, 768, 3), dtype=np.uint8))
inputs = processor(text='<CAPTION>', images=dummy_image, return_tensors='pt')
text_embeddings = model.get_input_embeddings()(inputs['input_ids'])
```

**모든 경우에 동일한 Text Embedding이 생성됩니다!**

## 참고

- Processor는 이미지 존재 여부만 확인하여 task prompt 변환 여부를 결정
- 실제 이미지 픽셀 값은 text embedding 생성에 영향을 주지 않음
- 따라서 Tx에서 처리한 text_embeddings를 Rx에 공유하는 것이 효율적
