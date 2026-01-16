"""
Florence-2 base 모델을 로드하고 컴포넌트로 분리한 뒤,
reference와 동일한 결과가 나오는지 확인하는 테스트 스크립트.

이 스크립트의 목적:
1. Florence-2 모델을 여러 컴포넌트로 분리
2. 각 컴포넌트를 순차적으로 연결하여 동일한 결과 생성
3. 전체 모델 결과와 비교하여 검증

주요 컴포넌트:
1. vision_tower: 이미지를 시각적 특징으로 변환하는 비전 인코더
2. image_proj_norm: 특징 벡터를 정규화하는 레이어
3. image_pos_embed: 이미지 패치의 공간적 위치 정보를 추가
4. visual_temporal_embed: 시간적 위치 정보를 추가 (비디오용)
5. language_model: 시각적 특징과 텍스트를 결합하여 최종 텍스트 생성
6. image_projection: 비전 특징 차원을 언어 모델 차원으로 변환
"""

import torch  # PyTorch 라이브러리 (텐서 연산, 신경망)
import requests  # HTTP 요청 라이브러리 (이미지 다운로드용)
from PIL import Image  # 이미지 처리 라이브러리
from transformers import AutoProcessor, AutoModelForCausalLM  # HuggingFace Transformers
# - AutoProcessor: 이미지/텍스트 전처리
# - AutoModelForCausalLM: 언어 모델 자동 로드


def test_component_separation():
    """
    Florence-2 컴포넌트 분리 및 연결 테스트
    
    이 함수는 Florence-2 모델을 여러 컴포넌트로 분리한 뒤,
    각 컴포넌트를 순차적으로 연결하여 원본 모델과 동일한 결과를
    생성하는지 검증합니다.
    """
    # GPU가 있으면 GPU 사용, 없으면 CPU 사용
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # GPU가 있으면 float16 (반정밀도) 사용하여 메모리 절약, 없으면 float32 사용
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    # Florence-2 base 모델 이름
    model_name = "microsoft/Florence-2-base"
    
    print("=" * 70)
    print("Florence-2 Component Separation Test")
    print("=" * 70)
    
    # ====================================================================
    # 1. Florence-2 base 모델 로드 (weight까지 모두 로드)
    # ====================================================================
    # 전체 모델을 HuggingFace에서 다운로드하고 로드합니다.
    # 이때 모든 학습된 가중치(weight)도 함께 로드됩니다.
    print("\n[1/6] Loading Florence-2 base model with weights...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,                    # 모델 이름
        dtype=torch_dtype,             # 데이터 타입 (float16 또는 float32)
        attn_implementation="eager",   # Attention 구현 방식 (에러 방지)
        trust_remote_code=True         # 원격 코드 신뢰 (Florence-2는 커스텀 코드 사용)
    ).to(device)  # 지정된 디바이스(GPU/CPU)로 이동
    
    # Processor는 이미지와 텍스트를 전처리하는 도구입니다.
    # - 이미지: 리사이즈, 정규화 등
    # - 텍스트: 토크나이징 (단어를 토큰 ID로 변환)
    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    print("✓ Model and processor loaded")
    
    # ====================================================================
    # 2. 컴포넌트 분리
    # ====================================================================
    # 전체 모델에서 각 컴포넌트를 개별적으로 추출합니다.
    # 이렇게 분리하면 각 컴포넌트를 독립적으로 사용할 수 있습니다.
    print("\n[2/6] Extracting components...")
    
    # vision_tower: 이미지를 처리하는 비전 인코더 (DaViT)
    # 역할: 이미지 픽셀 값을 시각적 특징 벡터로 변환
    # 입력: (batch_size, 3, H, W) 이미지 픽셀 값
    # 출력: (batch_size, num_patches, 1024) 시각적 특징
    if hasattr(model, 'vision_tower'):
        vision_tower = model.vision_tower
        print("✓ vision_tower extracted")
    else:
        raise AttributeError("vision_tower not found")
    
    # image_proj_norm: Layer Normalization 레이어
    # 역할: 이미지 특징 벡터를 정규화하여 학습 안정성 향상
    # 입력: (batch_size, seq_len, 768) 프로젝션된 특징
    # 출력: (batch_size, seq_len, 768) 정규화된 특징
    if hasattr(model, 'image_proj_norm'):
        image_proj_norm = model.image_proj_norm
        print("✓ image_proj_norm extracted")
    else:
        raise AttributeError("image_proj_norm not found")
    
    # image_pos_embed: 2D 위치 임베딩 (Positional Embedding)
    # 역할: 이미지 패치의 공간적 위치 정보를 추가
    # 예: 위쪽 패치와 아래쪽 패치를 구분할 수 있도록 위치 정보 부여
    # 입력: (batch_size, h, w, hidden_dim) 공간적으로 재구성된 특징
    # 출력: (batch_size, h, w, hidden_dim) 위치 정보가 추가된 특징
    if hasattr(model, 'image_pos_embed'):
        image_pos_embed = model.image_pos_embed
        print("✓ image_pos_embed extracted")
    else:
        print("⚠ image_pos_embed not found (may be optional)")
        image_pos_embed = None
    
    # visual_temporal_embed: 시간적 위치 임베딩 (Temporal Embedding)
    # 역할: 비디오의 경우 프레임 간 시간적 순서 정보를 추가
    # 단일 이미지의 경우에도 사용될 수 있음
    # 입력: (batch_size, T, hidden_dim) 첫 번째 토큰의 특징
    # 출력: (batch_size, T, hidden_dim) 시간 정보가 추가된 특징
    if hasattr(model, 'visual_temporal_embed'):
        visual_temporal_embed = model.visual_temporal_embed
        print("✓ visual_temporal_embed extracted")
    else:
        print("⚠ visual_temporal_embed not found (may be optional)")
        visual_temporal_embed = None
    
    # language_model: 텍스트 생성 모델 (언어 모델)
    # 역할: 시각적 특징과 텍스트를 결합하여 최종 텍스트를 생성
    # 입력: 병합된 임베딩 (이미지 특징 + 텍스트 임베딩)
    # 출력: 생성된 토큰 ID 시퀀스
    if hasattr(model, 'language_model'):
        language_model = model.language_model
        print("✓ language_model extracted")
    else:
        raise AttributeError("language_model not found")
    
    # image_projection: 선형 변환 레이어 (Linear Projection)
    # 역할: 비전 인코더 출력 차원(1024)을 언어 모델 차원(768)으로 변환
    # 입력: (batch_size, seq_len, 1024) 비전 특징
    # 출력: (batch_size, seq_len, 768) 프로젝션된 특징
    if hasattr(model, 'image_projection'):
        image_projection = model.image_projection
        print("✓ image_projection extracted")
    else:
        raise AttributeError("image_projection not found")
    
    # ====================================================================
    # 3. 테스트 이미지 로드
    # ====================================================================
    # 비교 테스트를 위해 샘플 이미지를 다운로드합니다.
    print("\n[3/6] Loading test image...")
    url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg?download=true"
    image = Image.open(requests.get(url, stream=True).raw)
    print(f"✓ Image loaded: {image.size} (width x height)")
    print(f"  - Original image size: {image.size[0]} x {image.size[1]} pixels")
    print(f"  - Image mode: {image.mode}")
    
    # ====================================================================
    # 4. Reference 결과 생성 (전체 모델 사용)
    # ====================================================================
    # 이 부분은 "정답"을 만드는 단계입니다.
    # 전체 모델을 그대로 사용하여 결과를 생성하고,
    # 나중에 분리된 컴포넌트 결과와 비교합니다.
    print("\n[4/6] Generating reference result (full model)...")
    task_prompt = "<CAPTION>"  # 이미지 캡션 생성 태스크
    
    # Processor를 사용하여 이미지와 텍스트를 모델 입력 형식으로 변환
    # - 이미지: 픽셀 값으로 변환 및 정규화
    # - 텍스트: 토큰 ID로 변환
    print(f"\n  [Image Preprocessing]")
    print(f"    Before processor: {image.size[0]} x {image.size[1]} pixels")
    inputs_ref = processor(text=task_prompt, images=image, return_tensors="pt")
    
    # Processor 이후 이미지 사이즈 확인
    if 'pixel_values' in inputs_ref:
        pixel_values_shape = inputs_ref['pixel_values'].shape
        print(f"    After processor (pixel_values): {pixel_values_shape}")
        print(f"      - Batch size: {pixel_values_shape[0]}")
        print(f"      - Channels: {pixel_values_shape[1]} (RGB)")
        print(f"      - Height: {pixel_values_shape[2]} pixels")
        print(f"      - Width: {pixel_values_shape[3]} pixels")
        print(f"      - Total pixels: {pixel_values_shape[2] * pixel_values_shape[3]}")
    
    # 텐서를 적절한 디바이스와 데이터 타입으로 변환
    # - input_ids는 정수형(long)이어야 함
    # - pixel_values는 모델의 dtype(float16/float32)과 일치해야 함
    inputs_ref = {k: v.to(device, dtype=torch_dtype if k != 'input_ids' else torch.long) 
                  for k, v in inputs_ref.items() if isinstance(v, torch.Tensor)}
    if 'input_ids' in inputs_ref:
        inputs_ref['input_ids'] = inputs_ref['input_ids'].long()
    
    if 'input_ids' in inputs_ref:
        print(f"    input_ids shape: {inputs_ref['input_ids'].shape}")
        print(f"      - Sequence length: {inputs_ref['input_ids'].shape[1]} tokens")
    
    # 그래디언트 계산 비활성화 (추론 모드, 메모리 절약)
    with torch.no_grad():
        # model.generate(): 전체 모델을 사용하여 텍스트 생성
        # 이 함수는 내부적으로 다음을 수행합니다:
        #   1. 이미지를 vision_tower로 처리
        #   2. 이미지 특징과 텍스트를 결합
        #   3. language_model로 텍스트 생성
        # 
        # 파라미터 설명:
        # - input_ids: 텍스트 토큰 ID (예: "<CAPTION>"을 토큰 ID로 변환한 것)
        # - pixel_values: 이미지 픽셀 값 (전처리된 이미지)
        # - max_new_tokens: 최대 생성할 토큰 수 (1024개까지)
        # - do_sample: False면 greedy decoding (가장 확률 높은 토큰 선택)
        # - num_beams: Beam search의 beam 수 (3개 후보 중 최선 선택)
        # - use_cache: False로 설정 (과거 키-값 캐시 사용 안 함, 에러 방지)
        generated_ids_ref = model.generate(
            input_ids=inputs_ref["input_ids"],      # 입력 텍스트 토큰 ID
            pixel_values=inputs_ref["pixel_values"], # 입력 이미지 픽셀 값
            max_new_tokens=1024,                     # 최대 생성 토큰 수
            do_sample=False,                         # Greedy decoding 사용
            num_beams=3,                             # Beam search beam 수
            use_cache=False,                         # 캐시 사용 안 함
        )
        # 출력: generated_ids_ref는 생성된 토큰 ID들의 텐서
        # 예: [1, 14] 형태 (batch_size=1, sequence_length=14)
    
    # 생성된 토큰 ID를 실제 텍스트로 변환 (디코딩)
    # skip_special_tokens=False: 특수 토큰(<CAPTION> 등)도 포함하여 디코딩
    generated_text_ref = processor.batch_decode(
        generated_ids_ref, skip_special_tokens=False
    )[0]
    
    # 후처리: 생성된 텍스트를 태스크에 맞게 파싱
    # 예: "<CAPTION> A green car..." -> {'<CAPTION>': 'A green car...'}
    reference_result = processor.post_process_generation(
        generated_text_ref,
        task=task_prompt,
        image_size=(image.width, image.height)
    )
    
    print(f"✓ Reference result: {reference_result}")
    
    # ====================================================================
    # 5. 컴포넌트 분리 방식으로 동일한 결과 생성
    # ====================================================================
    # 이제 전체 모델 대신 분리된 컴포넌트들을 순차적으로 연결하여
    # 동일한 결과를 생성합니다. 각 단계를 명시적으로 수행합니다.
    print("\n[5/6] Generating result using separated components...")
    
    # 입력 준비 (Reference와 동일한 입력 사용)
    pixel_values = inputs_ref["pixel_values"]  # 이미지 픽셀 값
    input_ids = inputs_ref["input_ids"]        # 텍스트 토큰 ID
    
    print(f"\n  [Vision Tower Input Analysis]")
    print(f"    pixel_values shape: {pixel_values.shape}")
    print(f"      - Batch: {pixel_values.shape[0]}")
    print(f"      - Channels: {pixel_values.shape[1]} (RGB)")
    print(f"      - Height: {pixel_values.shape[2]} pixels")
    print(f"      - Width: {pixel_values.shape[3]} pixels")
    print(f"      - Total image size: {pixel_values.shape[2]} x {pixel_values.shape[3]} = {pixel_values.shape[2] * pixel_values.shape[3]} pixels")
    
    with torch.no_grad():
        # ================================================================
        # Step 1: vision_tower - 이미지를 시각적 특징으로 변환
        # ================================================================
        # forward_features_unpool: 이미지를 패치로 나누어 처리
        # 입력: (batch_size, 3, H, W) 이미지 픽셀
        # 출력: (batch_size, num_patches, 1024) 시각적 특징
        print(f"\n  [Step 1: vision_tower]")
        print(f"    Input to vision_tower: {pixel_values.shape}")
        vision_output = vision_tower.forward_features_unpool(pixel_values)
        if isinstance(vision_output, tuple):
            vision_features = vision_output[0]  # 특징 벡터
            size = vision_output[1] if len(vision_output) > 1 else None  # 공간 크기 (H, W)
            if size is not None:
                print(f"    Vision tower returned size: {size}")
        else:
            vision_features = vision_output
            size = None
        
        print(f"    Output from vision_tower: {vision_features.shape}")
        print(f"      - Batch: {vision_features.shape[0]}")
        print(f"      - Sequence length (num_patches): {vision_features.shape[1]}")
        print(f"      - Hidden dimension: {vision_features.shape[2]}")
        print(f"      - Note: {vision_features.shape[1]} patches from {pixel_values.shape[2]}x{pixel_values.shape[3]} image")
        
        # ================================================================
        # Step 2: image_pos_embed - 공간적 위치 정보 추가
        # ================================================================
        # 이미지 패치의 2D 위치(위/아래, 왼쪽/오른쪽) 정보를 추가합니다.
        print(f"\n  [Step 2: image_pos_embed]")
        print(f"    Input shape: {vision_features.shape}")
        batch_size = vision_features.shape[0]
        T = 1  # Single frame (단일 이미지의 경우)
        
        if image_pos_embed is not None:
            # 특징을 (batch*T, seq_len, hidden_dim) 형태로 재구성
            x = vision_features.view(batch_size * T, -1, vision_features.shape[-1])
            num_tokens = x.shape[-2]  # 패치 개수
            h = w = int(num_tokens ** 0.5)  # 정사각형 가정 (h = w)
            
            print(f"    num_tokens: {num_tokens}, h={h}, w={w}, h*w={h*w}")
            
            if h * w == num_tokens:
                # 완전한 정사각형인 경우에만 위치 임베딩 적용
                # (batch*T, h, w, hidden_dim) 형태로 재구성
                x = x.view(batch_size * T, h, w, x.shape[-1])
                print(f"    Reshaped to spatial grid: {x.shape}")
                # 위치 임베딩 계산 및 추가
                pos_embed = image_pos_embed(x)  # 각 위치에 대한 임베딩 생성
                print(f"    Position embeddings shape: {pos_embed.shape}")
                x = x + pos_embed  # 원본 특징에 위치 정보 추가
                # 다시 (batch, T*h*w, hidden_dim) 형태로 변환
                x = x.view(batch_size, T * h * w, x.shape[-1])
                vision_features = x
                print(f"    Output shape: {vision_features.shape}")
            else:
                print(f"    ⚠ Skipped (non-square: {num_tokens} != {h*w})")
        else:
            print(f"    ⚠ image_pos_embed is None, skipped")
        
        # ================================================================
        # Step 3: visual_temporal_embed - 시간적 위치 정보 추가
        # ================================================================
        # 비디오의 경우 프레임 간 시간 순서 정보를 추가합니다.
        # 단일 이미지의 경우에도 적용될 수 있습니다.
        print(f"\n  [Step 3: visual_temporal_embed]")
        print(f"    Input shape: {vision_features.shape}")
        if visual_temporal_embed is not None:
            # (batch, T, seq_len, hidden_dim) 형태로 재구성
            x_reshaped = vision_features.view(batch_size, T, -1, vision_features.shape[-1])
            print(f"    Reshaped to (batch, T, seq_len, hidden): {x_reshaped.shape}")
            # 첫 번째 토큰에만 시간 임베딩 적용
            first_token = x_reshaped[:, :, 0]
            print(f"    First token shape (for temporal embed): {first_token.shape}")
            visual_temporal_emb = visual_temporal_embed(first_token)
            print(f"    Temporal embeddings shape: {visual_temporal_emb.shape}")
            # 모든 토큰에 시간 정보 브로드캐스트하여 추가
            x_reshaped = x_reshaped + visual_temporal_emb.view(1, T, 1, vision_features.shape[-1])
            # 다시 (batch, T*seq_len, hidden_dim) 형태로 변환
            vision_features = x_reshaped.view(batch_size, T * x_reshaped.shape[2], vision_features.shape[-1])
            print(f"    Output shape: {vision_features.shape}")
        else:
            print(f"    ⚠ visual_temporal_embed is None, skipped")
        
        # ================================================================
        # Step 4: image_feature_source pooling - 특징 집계
        # ================================================================
        # Florence-2는 여러 방식으로 특징을 집계할 수 있습니다:
        # - spatial_avg_pool: 공간 차원 평균 (각 프레임의 패치들을 평균)
        #                     출력: (batch, T, hidden_dim) - 시퀀스 길이 = T (보통 1)
        # - temporal_avg_pool: 시간 차원 평균 (모든 프레임을 평균)
        #                      출력: (batch, seq_len, hidden_dim) - 시퀀스 길이 = 원본 seq_len
        # - last_frame: 마지막 프레임만 사용
        #                출력: (batch, seq_len, hidden_dim) - 시퀀스 길이 = 원본 seq_len
        #
        # ⚠ 중요: 576 -> 577로 늘어나는 이유
        # image_feature_source가 여러 소스를 포함하면, 각 소스의 시퀀스 길이가 합쳐집니다.
        # 예: ['spatial_avg_pool', 'temporal_avg_pool']인 경우:
        #   - spatial_avg_pool: (batch, 1, hidden) -> 시퀀스 길이 1
        #   - temporal_avg_pool: (batch, 576, hidden) -> 시퀀스 길이 576
        #   - torch.cat(..., dim=1) 결과: (batch, 1+576=577, hidden)
        print(f"\n  [Step 4: image_feature_source pooling]")
        print(f"    Input shape: {vision_features.shape}")
        print(f"    Before reshaping: batch={batch_size}, T={T}, seq_len={vision_features.shape[1]}, hidden={vision_features.shape[2]}")
        
        x_reshaped = vision_features.view(batch_size, T, -1, vision_features.shape[-1])
        print(f"    After reshaping to (batch, T, seq_len, hidden): {x_reshaped.shape}")
        
        # 기본값: last_frame 사용
        if hasattr(model, 'image_feature_source'):
            image_feature_source = model.image_feature_source
            print(f"    image_feature_source from model: {image_feature_source}")
        else:
            image_feature_source = ['last_frame']
            print(f"    image_feature_source (default): {image_feature_source}")
        
        # 각 집계 방식으로 특징 계산
        # 주의: 각 pooling 방식의 출력 차원이 다릅니다!
        x_feat_dict = {
            'spatial_avg_pool': x_reshaped.mean(dim=2),  # 공간 평균: (batch, T, hidden_dim)
            'temporal_avg_pool': x_reshaped.mean(dim=1),  # 시간 평균: (batch, seq_len, hidden_dim)
            'last_frame': x_reshaped[:, -1]                # 마지막 프레임: (batch, seq_len, hidden_dim)
        }
        
        # 각 집계 방식의 출력 사이즈 확인
        print(f"\n    Pooling outputs:")
        for key, value in x_feat_dict.items():
            print(f"      - {key}: {value.shape} (seq_len={value.shape[1]})")
        
        # 설정된 집계 방식들을 결합
        new_x = []
        print(f"\n    Combining feature sources: {image_feature_source}")
        seq_len_breakdown = []
        for _image_feature_source in image_feature_source:
            if _image_feature_source not in x_feat_dict:
                raise ValueError(f'invalid image feature source: {_image_feature_source}')
            feat = x_feat_dict[_image_feature_source]
            seq_len = feat.shape[1]
            seq_len_breakdown.append(f"{_image_feature_source}({seq_len})")
            print(f"      - Adding {_image_feature_source}: shape {feat.shape} (seq_len={seq_len})")
            new_x.append(feat)
        
        # 여러 집계 결과를 시퀀스 차원으로 연결
        # 중요: torch.cat(new_x, dim=1)은 시퀀스 차원(dim=1)을 따라 연결합니다.
        # 만약 image_feature_source에 여러 소스가 있으면, 각 소스의 시퀀스 길이가 합쳐집니다.
        # 예: ['spatial_avg_pool', 'temporal_avg_pool']인 경우:
        #   - spatial_avg_pool: (batch, 1, hidden)
        #   - temporal_avg_pool: (batch, 576, hidden)
        #   - 결과: (batch, 1+576=577, hidden)
        vision_features = torch.cat(new_x, dim=1)
        print(f"    After concatenation: {vision_features.shape}")
        print(f"      - Sequence length changed from {x_reshaped.shape[2]} to {vision_features.shape[1]}")
        if vision_features.shape[1] != x_reshaped.shape[2]:
            print(f"      ⚠ WHY: Sequence length increased because multiple feature sources were concatenated!")
            print(f"         - Original seq_len: {x_reshaped.shape[2]}")
            print(f"         - Number of sources: {len(image_feature_source)}")
            print(f"         - Sequence length breakdown: {' + '.join(seq_len_breakdown)} = {vision_features.shape[1]}")
            total_seq_len = sum(x_feat_dict[src].shape[1] for src in image_feature_source)
            print(f"         - Total: {total_seq_len} = sum of all source sequence lengths")
            print(f"         - This is why 576 -> 577: spatial_avg_pool(1) + temporal_avg_pool(576) = 577")
        
        # ================================================================
        # Step 5: image_projection - 차원 변환
        # ================================================================
        # 비전 인코더 출력 차원(1024)을 언어 모델 차원(768)으로 변환
        # 행렬 곱셈: (batch, seq_len, 1024) @ (1024, 768) = (batch, seq_len, 768)
        print(f"\n  [Step 5: image_projection]")
        print(f"    Input shape: {vision_features.shape}")
        print(f"    Projecting from {vision_features.shape[2]} to 768 dimensions")
        vision_features = vision_features @ image_projection
        print(f"    Output shape: {vision_features.shape}")
        
        # ================================================================
        # Step 6: image_proj_norm - 정규화
        # ================================================================
        # Layer Normalization을 적용하여 학습 안정성 향상
        # 입력과 출력 차원 동일: (batch, seq_len, 768)
        print(f"\n  [Step 6: image_proj_norm]")
        print(f"    Input shape: {vision_features.shape}")
        vision_features = image_proj_norm(vision_features)
        print(f"    Output shape: {vision_features.shape} (same as input)")
        
        # ================================================================
        # Step 7: language_model - 텍스트 생성
        # ================================================================
        print(f"\n  [Step 7: language_model]")
        # 텍스트 임베딩 생성
        # input_ids를 임베딩 벡터로 변환
        embedding_layer = model.get_input_embeddings()
        print(f"    input_ids shape: {input_ids.shape}")
        text_embeddings = embedding_layer(input_ids)  # (batch, text_len, 768)
        print(f"    text_embeddings shape: {text_embeddings.shape}")
        
        # 이미지 특징과 텍스트 임베딩 병합
        # Florence-2의 내부 메서드를 사용하여 올바르게 병합
        # 출력: 병합된 임베딩과 attention mask
        print(f"    vision_features shape: {vision_features.shape}")
        merged_embeds, attention_mask = model._merge_input_ids_with_image_features(
            vision_features,  # 이미지 특징 (batch, img_seq_len, 768)
            text_embeddings   # 텍스트 임베딩 (batch, text_seq_len, 768)
        )
        # merged_embeds: (batch, img_seq_len + text_seq_len, 768)
        print(f"    merged_embeds shape: {merged_embeds.shape}")
        print(f"      - Image tokens: {vision_features.shape[1]}")
        print(f"      - Text tokens: {text_embeddings.shape[1]}")
        print(f"      - Total tokens: {merged_embeds.shape[1]} = {vision_features.shape[1]} + {text_embeddings.shape[1]}")
        print(f"    attention_mask shape: {attention_mask.shape}")
        
        # 언어 모델을 사용하여 텍스트 생성
        # input_ids=None: 토큰 ID 대신 inputs_embeds 사용
        # inputs_embeds: 병합된 임베딩 (이미지 + 텍스트)
        # attention_mask: 어느 토큰에 주의를 기울일지 지정
        print(f"    Generating text with language_model...")
        generated_ids_sep = language_model.generate(
            input_ids=None,              # 토큰 ID 대신 임베딩 사용
            inputs_embeds=merged_embeds, # 병합된 임베딩 입력
            attention_mask=attention_mask,# 어텐션 마스크
            max_new_tokens=1024,         # 최대 생성 토큰 수
            do_sample=False,             # Greedy decoding
            num_beams=3,                 # Beam search beam 수
            use_cache=False,             # 캐시 사용 안 함
        )
        # 출력: 생성된 토큰 ID (batch, generated_seq_len)
        print(f"    Generated tokens shape: {generated_ids_sep.shape}")
        print(f"      - Generated sequence length: {generated_ids_sep.shape[1]} tokens")
    
    # ====================================================================
    # 디코딩 및 후처리
    # ====================================================================
    # 생성된 토큰 ID를 실제 텍스트로 변환
    generated_text_sep = processor.batch_decode(
        generated_ids_sep, skip_special_tokens=False
    )[0]
    
    # 태스크에 맞게 결과 파싱
    separated_result = processor.post_process_generation(
        generated_text_sep,
        task=task_prompt,
        image_size=(image.width, image.height)
    )
    
    print(f"✓ Separated components result: {separated_result}")
    
    # ====================================================================
    # 6. 결과 비교
    # ====================================================================
    # Reference(전체 모델) 결과와 분리된 컴포넌트 결과를 비교하여
    # 컴포넌트 분리가 올바르게 수행되었는지 검증합니다.
    print("\n[6/6] Comparing results...")
    print("=" * 70)
    print("Reference Result (Full Model):")
    print(f"  {reference_result}")
    print("\nSeparated Components Result:")
    print(f"  {separated_result}")
    print("=" * 70)
    
    # 결과 비교
    if reference_result == separated_result:
        print("\n✓ SUCCESS: Results match exactly!")
        # 결과가 정확히 일치하면 컴포넌트 분리가 성공적으로 수행된 것입니다.
    else:
        print("\n⚠ Results differ:")
        print(f"  Reference: {reference_result}")
        print(f"  Separated: {separated_result}")
        
        # 토큰 레벨 비교 (더 상세한 분석)
        # 생성된 토큰 ID가 동일한지 확인
        if generated_ids_ref.shape == generated_ids_sep.shape:
            if torch.equal(generated_ids_ref, generated_ids_sep):
                print("\n  ✓ Generated token IDs are identical")
            else:
                # 토큰 ID가 다른 위치의 개수 계산
                diff = (generated_ids_ref != generated_ids_sep).sum().item()
                print(f"\n  ⚠ Generated token IDs differ at {diff} positions")
                print(f"    Reference tokens: {generated_ids_ref[0][:20].tolist()}")
                print(f"    Separated tokens:  {generated_ids_sep[0][:20].tolist()}")
        else:
            print(f"\n  ⚠ Generated token IDs have different shapes:")
            print(f"    Reference: {generated_ids_ref.shape}")
            print(f"    Separated: {generated_ids_sep.shape}")
    
    print("\n" + "=" * 70)
    print("Test Complete")
    print("=" * 70)


if __name__ == "__main__":
    test_component_separation()
