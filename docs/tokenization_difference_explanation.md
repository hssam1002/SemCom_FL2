# Tokenization 차이 2의 원인 분석

## 문제 요약

- `processor(text, images)`: 8개 토큰
- `tokenizer([text])`: 6개 토큰
- **차이 = 2 토큰**

## 원인 분석

### Florence2Processor의 Task Prompt 변환

**Florence-2 Processor는 이미지가 제공될 때 task prompt를 실제 질문으로 변환합니다.**

```python
# Input: '<CAPTION>'
# processor(text='<CAPTION>', images=image) 호출 시:

Output: "What does the image describe?"
Tokens: ['<s>', 'What', 'Ġdoes', 'Ġthe', 'Ġimage', 'Ġdescribe', '?', '</s>']
Token count: 8
```

**Florence-2 Processor 내부 매핑:**
- `<CAPTION>` → "What does the image describe?"
- `<DETAILED_CAPTION>` → 다른 질문으로 변환
- `<OD>` → 또 다른 질문으로 변환

이것이 **Florence-2의 정상 동작**입니다. Processor가 task prompt를 자연어 질문으로 변환합니다.

### Tokenizer 직접 사용

```python
# Input: '<CAPTION>'
# tokenizer(['<CAPTION>']) 호출 시:

Output: "<CAPTION>" (그대로)
Tokens: ['<s>', '<', 'CAP', 'TION', '>', '</s>']
Token count: 6
```

**Tokenizer는 단순히 텍스트를 토큰화합니다.**
- Task prompt 변환 없음
- 그대로 토큰화

### 차이 상세 분석

| 항목 | processor(text, images) | tokenizer([text]) |
|------|------------------------|-------------------|
| **입력** | `<CAPTION>` | `<CAPTION>` |
| **내부 변환** | `"What does the image describe?"` | 변환 없음 |
| **출력 텍스트** | "What does the image describe?" | "<CAPTION>" |
| **토큰** | 8개 | 6개 |
| **차이** | +2 토큰 | - |

### 왜 이런 차이가 생기는가?

1. **Florence2Processor의 설계:**
   - Processor는 **multimodal input**을 처리하도록 설계됨
   - 이미지가 있을 때 task prompt를 자연어 질문으로 변환
   - 이렇게 하면 모델이 더 잘 이해할 수 있음

2. **Tokenizer의 역할:**
   - Tokenizer는 단순히 **텍스트를 토큰화**만 함
   - Processor의 변환 로직을 모름
   - Extended BART tokenizer이지만, task prompt 변환 기능은 없음

3. **Florence-2의 의도:**
   - Task prompt (`<CAPTION>`)는 **사용자 편의를 위한 축약형**
   - 실제 모델 입력은 **자연어 질문** ("What does the image describe?")
   - Processor가 이 변환을 담당

## 해결 방법

### 옵션 1: processor(text, images) 사용 (현재 방법)
- Dummy image 필요
- Processor의 정상 동작 유지
- Tokenization 일관성 보장

### 옵션 2: Text Embedding 공유 (추천)
- Main/test_semcom에서 processor로 처리
- Text embedding을 Tx/Rx에 공유
- Dummy image 불필요
- 더 효율적

### 옵션 3: Processor의 변환 로직 직접 구현
- Processor 내부의 task prompt → 질문 매핑을 직접 구현
- Dummy image 불필요하지만 복잡함

## 결론

**차이 2의 원인:**
- Processor가 `<CAPTION>` → `"What does the image describe?"`로 변환
- "What does the image describe?" = 6개 단어 토큰
- `<CAPTION>` = 4개 토큰 (BOS/EOS 제외하면)
- **차이 = 2 토큰**

**이것은 Florence-2의 정상 동작이며, processor가 이미지가 있을 때 task prompt를 자연어 질문으로 변환하는 것입니다.**
