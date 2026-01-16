# Text Embedding 공유 구조 업데이트

## 변경 사항 요약

Top level (main.py, test_semcom.py)에서 all-zero dummy image와 task_prompt를 사용하여 text_embeddings를 생성하고, 이를 Tx와 Rx에 공유하도록 수정했습니다.

## 수정된 파일

### 1. test_semcom.py

**변경 전:**
- `transmitter.encode_task_prompts()`를 호출하여 Tx에서 text_embeddings 생성
- 실제 이미지를 사용

**변경 후:**
- Top level에서 all-zero dummy image와 task_prompt로 text_embeddings 생성
- 생성된 text_embeddings를 Tx와 Rx에 공유
- `receiver.generate()`에도 text_embeddings 전달

```python
# Top level에서 text_embeddings 생성
import numpy as np
dummy_image = Image.fromarray(np.zeros((768, 768, 3), dtype=np.uint8))

inputs = florence2_model.processor(
    text=[task_prompt],
    images=[dummy_image],
    return_tensors="pt"
)
input_ids = inputs["input_ids"].to(device=device, dtype=torch.long)
embedding_layer = florence2_model.model.get_input_embeddings()
text_embeddings = embedding_layer(input_ids)

# Tx와 Rx에 공유
merged_embeds, attention_mask = receiver(received_signal, text_embeddings)
generated_ids_rx = receiver.generate(received_signal, text_embeddings, ...)
```

### 2. main.py

**변경 전:**
- task_prompts를 문자열로 전달
- Receiver에서 문자열을 처리

**변경 후:**
- Top level에서 all-zero dummy image와 task_prompts로 text_embeddings 생성
- 생성된 text_embeddings를 Receiver에 전달

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
input_ids = inputs["input_ids"].to(device=device, dtype=torch.long)
embedding_layer = florence2_model.model.get_input_embeddings()
text_embeddings = embedding_layer(input_ids)

# Receiver에 전달
merged_embeds, attention_mask = receiver(received_signal, text_embeddings)
```

### 3. transmitter/transmitter.py

**변경 사항:**
- `encode_task_prompts()` 메서드에 주석 추가
- Top level에서 처리하는 것을 권장한다는 주석 추가
- Backward compatibility를 위해 메서드는 유지

```python
def encode_task_prompts(...):
    """
    NOTE: This method is kept for backward compatibility.
    It is recommended to generate text_embeddings at top level (main/test_semcom.py)
    using processor with all-zero dummy image, and share them with Tx/Rx.
    """
```

### 4. receiver/receiver.py

**변경 사항:**
- 주석 업데이트: Top level에서 text_embeddings를 생성하고 공유하는 것을 권장
- Backward compatibility를 위해 문자열 입력도 여전히 지원

```python
# 주석 업데이트:
# Note: task_prompts should be text_embeddings (shared from top level) or task prompt strings
# It is recommended to generate text_embeddings at top level (main/test_semcom.py)
# using processor with all-zero dummy image, and share them with Tx/Rx
```

## 장점

1. **일관성**: Top level에서 한 번만 text_embeddings 생성하여 Tx/Rx에 공유
2. **효율성**: Dummy image를 한 번만 생성 (top level에서)
3. **명확성**: Text embedding 생성 위치가 명확 (top level)
4. **유연성**: Backward compatibility 유지 (문자열 입력도 지원)

## 사용 방법

### test_semcom.py
```python
# Top level에서 text_embeddings 생성
dummy_image = Image.fromarray(np.zeros((768, 768, 3), dtype=np.uint8))
text_embeddings = generate_text_embeddings(task_prompt, dummy_image)

# Tx/Rx에 공유
merged_embeds, attention_mask = receiver(received_signal, text_embeddings)
generated_ids = receiver.generate(received_signal, text_embeddings, ...)
```

### main.py
```python
# Top level에서 text_embeddings 생성
dummy_image = PILImage.fromarray(np.zeros((768, 768, 3), dtype=np.uint8))
text_embeddings = generate_text_embeddings(task_prompts, dummy_image)

# Receiver에 전달
merged_embeds, attention_mask = receiver(received_signal, text_embeddings)
```

## 참고

- **Dummy Image**: All-zero dummy image 사용 (어떤 이미지든 상관없음)
- **Processor**: `processor(text, images=dummy_image)` 형태로 호출하여 올바른 tokenization 보장
- **Backward Compatibility**: 문자열 입력도 여전히 지원 (receiver에서 처리)
