# Tokenization 차이 분석

## 차이 2의 원인

### Method 1: processor(text, images)
```
Input: '<CAPTION>'
Output: ['<s>', 'What', 'Ġdoes', 'Ġthe', 'Ġimage', 'Ġdescribe', '?', '</s>']
Tokens: 8개
```

**Florence2Processor는 이미지가 있을 때 task prompt를 실제 질문으로 변환합니다!**
- `<CAPTION>` → "What does the image describe?"
- 이것이 Florence-2의 정상 동작입니다.

### Method 2: tokenizer([text])
```
Input: '<CAPTION>'
Output: ['<s>', '<', 'CAP', 'TION', '>', '</s>']
Tokens: 6개
```

**tokenizer는 task prompt를 그대로 토큰화합니다.**

### 차이
- **차이 = 2 토큰**
- processor가 task prompt를 변환하는 과정에서 추가 토큰이 생깁니다.

## 해결 방안

### 현재 방법: Dummy Image 사용
- Receiver에서 dummy image를 만들어 processor 호출
- 장점: processor의 정상 동작 유지
- 단점: 불필요한 이미지 생성, 비효율적

### 개선 방법: Text Embedding 공유 (추천)
- Tx에서 text embedding을 처리하고 Rx로 전달
- 장점: 
  - Dummy image 불필요
  - 일관된 tokenization 보장
  - 더 효율적
  - Tx/Rx 분리 명확
