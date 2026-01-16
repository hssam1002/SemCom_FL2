"""
Dummy Image vs Real Image: Text Embedding 동일성 테스트

질문: 어떠한 image든 상관없이 (dummy image를 넣어도) text embedding 결과는 동일한가?
"""
from transformers import AutoProcessor
from PIL import Image
import requests
import numpy as np
import torch

def test_dummy_image_embedding():
    processor = AutoProcessor.from_pretrained('microsoft/Florence-2-base', trust_remote_code=True)
    
    print('='*70)
    print('Dummy Image vs Real Image: Text Embedding 동일성 테스트')
    print('='*70)
    
    task_prompt = '<CAPTION>'
    
    # 1. Real image
    print('\n[1] Real Image 사용')
    url = 'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg?download=true'
    real_image = Image.open(requests.get(url, stream=True).raw)
    inputs_real = processor(text=task_prompt, images=real_image, return_tensors='pt')
    input_ids_real = inputs_real['input_ids']
    
    print(f'  Input IDs: {input_ids_real.tolist()[0]}')
    print(f'  Decoded text: "{processor.tokenizer.decode(input_ids_real[0])}"')
    print(f'  Token count: {input_ids_real.shape[1]}')
    
    # 2. Dummy image (zeros, 768x768)
    print('\n[2] Dummy Image (zeros, 768x768) 사용')
    dummy_image_zeros = Image.fromarray(np.zeros((768, 768, 3), dtype=np.uint8))
    inputs_dummy_zeros = processor(text=task_prompt, images=dummy_image_zeros, return_tensors='pt')
    input_ids_dummy_zeros = inputs_dummy_zeros['input_ids']
    
    print(f'  Input IDs: {input_ids_dummy_zeros.tolist()[0]}')
    print(f'  Decoded text: "{processor.tokenizer.decode(input_ids_dummy_zeros[0])}"')
    print(f'  Token count: {input_ids_dummy_zeros.shape[1]}')
    
    # 3. Dummy image (random, 768x768)
    print('\n[3] Dummy Image (random, 768x768) 사용')
    dummy_image_random = Image.fromarray(np.random.randint(0, 255, (768, 768, 3), dtype=np.uint8))
    inputs_dummy_random = processor(text=task_prompt, images=dummy_image_random, return_tensors='pt')
    input_ids_dummy_random = inputs_dummy_random['input_ids']
    
    print(f'  Input IDs: {input_ids_dummy_random.tolist()[0]}')
    print(f'  Decoded text: "{processor.tokenizer.decode(input_ids_dummy_random[0])}"')
    print(f'  Token count: {input_ids_dummy_random.shape[1]}')
    
    # 4. Dummy image (다른 크기: 512x512)
    print('\n[4] Dummy Image (다른 크기: 512x512) 사용')
    dummy_image_small = Image.fromarray(np.zeros((512, 512, 3), dtype=np.uint8))
    inputs_dummy_small = processor(text=task_prompt, images=dummy_image_small, return_tensors='pt')
    input_ids_dummy_small = inputs_dummy_small['input_ids']
    
    print(f'  Input IDs: {input_ids_dummy_small.tolist()[0]}')
    print(f'  Decoded text: "{processor.tokenizer.decode(input_ids_dummy_small[0])}"')
    print(f'  Token count: {input_ids_dummy_small.shape[1]}')
    
    # 5. Dummy image (다른 크기: 224x224)
    print('\n[5] Dummy Image (다른 크기: 224x224) 사용')
    dummy_image_tiny = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
    inputs_dummy_tiny = processor(text=task_prompt, images=dummy_image_tiny, return_tensors='pt')
    input_ids_dummy_tiny = inputs_dummy_tiny['input_ids']
    
    print(f'  Input IDs: {input_ids_dummy_tiny.tolist()[0]}')
    print(f'  Decoded text: "{processor.tokenizer.decode(input_ids_dummy_tiny[0])}"')
    print(f'  Token count: {input_ids_dummy_tiny.shape[1]}')
    
    # 비교
    print('\n' + '='*70)
    print('비교 결과:')
    print('='*70)
    
    # Input IDs 비교
    print('\n[Input IDs 비교]')
    print(f'  Real vs Dummy(zeros, 768x768):  {torch.equal(input_ids_real, input_ids_dummy_zeros)}')
    print(f'  Real vs Dummy(random, 768x768): {torch.equal(input_ids_real, input_ids_dummy_random)}')
    print(f'  Real vs Dummy(512x512):         {torch.equal(input_ids_real, input_ids_dummy_small)}')
    print(f'  Real vs Dummy(224x224):         {torch.equal(input_ids_real, input_ids_dummy_tiny)}')
    print(f'  Dummy(zeros) vs Dummy(random):  {torch.equal(input_ids_dummy_zeros, input_ids_dummy_random)}')
    
    # 정확한 값 비교
    all_equal = all([
        torch.equal(input_ids_real, input_ids_dummy_zeros),
        torch.equal(input_ids_real, input_ids_dummy_random),
        torch.equal(input_ids_real, input_ids_dummy_small),
        torch.equal(input_ids_real, input_ids_dummy_tiny),
    ])
    
    if all_equal:
        print('\n✓ 결론: 모든 경우에 Input IDs가 동일합니다!')
        print('✓ 따라서 Text Embeddings도 동일합니다!')
        print('✓ Processor는 이미지 내용과 관계없이 텍스트만 처리합니다!')
    else:
        print('\n✗ Input IDs가 다릅니다!')
        print('  차이점:')
        if not torch.equal(input_ids_real, input_ids_dummy_zeros):
            print(f'    Real:        {input_ids_real.tolist()[0]}')
            print(f'    Dummy(zeros): {input_ids_dummy_zeros.tolist()[0]}')
    
    # 여러 task prompt로도 테스트
    print('\n' + '='*70)
    print('추가 테스트: 다른 Task Prompts')
    print('='*70)
    
    task_prompts = ['<DETAILED_CAPTION>', '<OD>']
    dummy_image = Image.fromarray(np.zeros((768, 768, 3), dtype=np.uint8))
    
    for tp in task_prompts:
        inputs_real_tp = processor(text=tp, images=real_image, return_tensors='pt')
        inputs_dummy_tp = processor(text=tp, images=dummy_image, return_tensors='pt')
        is_equal = torch.equal(inputs_real_tp['input_ids'], inputs_dummy_tp['input_ids'])
        
        print(f'\n{tp}:')
        print(f'  Real vs Dummy: {is_equal}')
        if is_equal:
            print(f'  ✓ 동일 (tokens: {inputs_real_tp["input_ids"].shape[1]})')
        else:
            print(f'  ✗ 다름')
            print(f'    Real:   {inputs_real_tp["input_ids"].tolist()[0]}')
            print(f'    Dummy:  {inputs_dummy_tp["input_ids"].tolist()[0]}')

if __name__ == '__main__':
    test_dummy_image_embedding()
