# Tokenization 차이 2의 원인: 최종 정리

## 핵심 답변

### Q: 차이 2가 정확히 무엇 때문인가?

**A: Florence2Processor가 task prompt를 자연어 질문으로 변환하기 때문입니다.**

```python
# processor(text='<CAPTION>', images=image)
Input:  '<CAPTION>'
Output: "What does the image describe?"
Tokens: ['<s>', 'What', 'Ġdoes', 'Ġthe', 'Ġimage', 'Ġdescribe', '?', '</s>']  # 8개

# tokenizer(['<CAPTION>'])
Input:  '<CAPTION>'
Output: "<CAPTION>"
Tokens: ['<s>', '<', 'CAP', 'TION', '>', '</s>']  # 6개

차이: 8 - 6 = 2 토큰
```

### Task별 변환 예시

| Task Prompt | processor(text, images) | tokenizer([text]) | 차이 |
|-------------|------------------------|-------------------|------|
| `<CAPTION>` | "What does the image describe?" (8) | "<CAPTION>" (6) | +2 |
| `<DETAILED_CAPTION>` | "Describe in detail what is shown in the image." (13) | "<DETAILED_CAPTION>" (11) | +2 |
| `<OD>` | "Locate the objects with category name in the image." (13) | "<OD>" (5) | +8 |

## 왜 이런 변환이 일어나는가?

### Florence-2 Processor의 설계 철학

1. **Task Prompt는 축약형**
   - `<CAPTION>`, `<DETAILED_CAPTION>` 등은 사용자 편의를 위한 축약형
   - 실제 모델은 자연어 질문으로 학습됨

2. **Processor의 역할**
   - 이미지가 제공될 때: task prompt → 자연어 질문 변환
   - 이 변환은 Florence-2 모델이 올바르게 이해하도록 도와줌

3. **Tokenizer의 역할**
   - Extended BART tokenizer
   - 단순히 텍스트를 토큰화
   - **Processor의 변환 로직을 모름**
   - Task prompt 변환 기능 없음

### Processor 내부 매핑 (예상)

```python
# Florence2Processor 내부에 다음과 같은 매핑이 있을 것으로 추정:
TASK_PROMPT_MAPPING = {
    '<CAPTION>': "What does the image describe?",
    '<DETAILED_CAPTION>': "Describe in detail what is shown in the image.",
    '<OD>': "Locate the objects with category name in the image.",
    # ... 기타 task prompts
}
```

## Extended BART Tokenizer의 한계

**Extended BART Tokenizer는:**
- BART 기반의 토크나이저
- 자연어 텍스트를 토큰화하는 역할
- **Task prompt → 질문 변환 기능 없음**
- Processor의 변환 로직을 모름

**따라서:**
- `tokenizer(['<CAPTION>'])`는 `<CAPTION>`을 그대로 토큰화
- Processor의 변환 없이 단순 토큰화

## 실용적 해결책

### 현재 구조 (Text Prompt/Embedding 공유)

```python
# main.py 또는 test_semcom.py에서
processor = florence2_model.processor
inputs = processor(text=task_prompt, images=image, return_tensors="pt")
text_embeddings = model.get_input_embeddings()(inputs["input_ids"])

# Tx에 전달
tx_output = transmitter(image)

# Rx에 전달 (text_embeddings는 이미 공유되어 있음)
rx_output = receiver(received_signal, text_embeddings)  # text_embeddings 사용
```

이렇게 하면:
- ✅ Dummy image 불필요
- ✅ 일관된 tokenization 보장
- ✅ Tx/Rx 분리 명확
- ✅ Processor의 정상 동작 유지

## 결론

**차이 2의 원인:**
1. Processor가 `<CAPTION>` → `"What does the image describe?"`로 변환
2. 변환된 질문은 8개 토큰 (BOS/EOS 포함)
3. 원본 `<CAPTION>`은 6개 토큰
4. **차이 = 2 토큰**

**이것은 Florence-2의 정상 동작이며, processor가 이미지가 있을 때 task prompt를 자연어 질문으로 변환하는 설계입니다.**

**Extended BART Tokenizer는 단순히 텍스트를 토큰화할 뿐, processor의 변환 로직은 없습니다.**
