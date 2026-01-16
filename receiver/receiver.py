"""
Receiver module.
Processes received vision tower features through remaining Florence-2 pipeline.
Compression point: AFTER_VISION_TOWER (Method 2)

The Receiver processes vision_tower output through:
1. image_pos_embed
2. visual_temporal_embed
3. image_feature_source_pooling
4. image_projection
5. image_proj_norm
6. language_model
7. decoding
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Union, List

from models.florence2_model import Florence2Model


class Receiver(nn.Module):
    """
    Receiver for semantic communication.
    
    Method 2: Compression after Vision Tower
    Processes received vision_tower features (from Transmitter) through remaining steps:
    1. Decompression (if compression was applied)
    2. image_pos_embed
    3. visual_temporal_embed
    4. image_feature_source_pooling
    5. image_projection
    6. image_proj_norm
    7. language_model + decoding
    
    Flow: received_vision_features -> processing -> merge with task_embedding -> language_model.generate
    
    Args:
        florence2_model: Florence-2 model instance
        use_pooled_features: Whether transmitter used pooled features (deprecated, kept for compatibility)
    """
    
    def __init__(
        self,
        florence2_model: Florence2Model,
        use_pooled_features: bool = False
    ):
        super().__init__()
        
        self.florence2_model = florence2_model
        self.use_pooled_features = use_pooled_features
        
        # Transmitter outputs vision_tower features (1024 dimension for base model)
        # After processing, will be 768 dimension (image_proj_norm output)
        self.vision_dim = florence2_model.get_vision_dim()  # 1024 (vision_tower output)
        self.projected_dim = 768  # After image_projection and image_proj_norm

        # Task embedding dimension (768 for Florence-2-base text embeddings)
        self.task_embedding_dim = 768
        
        # No projection needed since both vision (after projection) and task embeddings are 768
        self.task_projection = None
        
        # Access Florence-2's transformer encoder and decoder (optional, for fallback)
        self.transformer_encoder = florence2_model.get_transformer_encoder()
        self.transformer_decoder = florence2_model.get_transformer_decoder()
        
        if self.transformer_encoder is None or self.transformer_decoder is None:
            print("Warning: Florence-2 transformer encoder/decoder not found.")
            print("Using placeholder components.")
    
    def forward(
        self,
        received_vision_embedding: torch.Tensor,
        task_prompts: Union[List[str], torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process received vision_tower features through remaining Florence-2 pipeline.
        
        Method 2 Flow:
        1. received_vision_embedding (vision_tower output from Tx) - shape: (batch, seq_len, 1024)
        2. image_pos_embed (if applicable)
        3. visual_temporal_embed (if applicable)
        4. image_feature_source_pooling
        5. image_projection (1024 -> 768)
        6. image_proj_norm
        7. Merge with task_embeddings
        8. -> Ready for language_model
        
        Args:
            received_vision_embedding: Received vision_tower features from channel
                Shape: (batch_size, seq_len, 1024) for base model
            task_prompts: Task prompts (list of strings) or text_embeddings (tensor)
                Recommended: text_embeddings generated at top level (main/test_semcom.py)
                using processor with all-zero dummy image, and shared with Tx/Rx
                If strings, will be encoded using processor with dummy image (backward compatibility)
        
        Returns:
            Tuple of (merged_embeds, attention_mask) ready for language_model.generate
            - merged_embeds: (batch, vision_seq_len + text_seq_len, 768)
            - attention_mask: (batch, vision_seq_len + text_seq_len)
        """
        model = self.florence2_model.model
        processor = self.florence2_model.processor
        
        batch_size = received_vision_embedding.shape[0]
        T = 1  # Single frame
        
        # Start with vision_tower features from Transmitter
        # Ensure we have the features (not tuple) - match test_component_separation.py
        if isinstance(received_vision_embedding, tuple):
            vision_features = received_vision_embedding[0]  # (batch, seq_len, 1024)
        else:
            vision_features = received_vision_embedding  # (batch, seq_len, 1024)
        
        # ============================================================
        # Step 1: image_pos_embed (if applicable)
        # ============================================================
        # EXACT COPY from test_component_separation.py lines 283-307
        if hasattr(model, 'image_pos_embed') and model.image_pos_embed is not None:
            # 특징을 (batch*T, seq_len, hidden_dim) 형태로 재구성
            x = vision_features.view(batch_size * T, -1, vision_features.shape[-1])
            num_tokens = x.shape[-2]  # 패치 개수
            h = w = int(num_tokens ** 0.5)  # 정사각형 가정 (h = w)
            
            if h * w == num_tokens:
                # 완전한 정사각형인 경우에만 위치 임베딩 적용
                # (batch*T, h, w, hidden_dim) 형태로 재구성
                x = x.view(batch_size * T, h, w, x.shape[-1])
                # 위치 임베딩 계산 및 추가
                pos_embed = model.image_pos_embed(x)  # 각 위치에 대한 임베딩 생성
                x = x + pos_embed  # 원본 특징에 위치 정보 추가
                # 다시 (batch, T*h*w, hidden_dim) 형태로 변환
                x = x.view(batch_size, T * h * w, x.shape[-1])
                vision_features = x
            # else: 완전한 정사각형이 아니면 image_pos_embed 건너뜀 (vision_features 그대로)
        
        # ============================================================
        # Step 2: visual_temporal_embed (if applicable)
        # ============================================================
        # EXACT COPY from test_component_separation.py lines 316-329
        if hasattr(model, 'visual_temporal_embed') and model.visual_temporal_embed is not None:
            # (batch, T, seq_len, hidden_dim) 형태로 재구성
            x_reshaped = vision_features.view(batch_size, T, -1, vision_features.shape[-1])
            # 첫 번째 토큰에만 시간 임베딩 적용
            first_token = x_reshaped[:, :, 0]
            visual_temporal_emb = model.visual_temporal_embed(first_token)
            # 모든 토큰에 시간 정보 브로드캐스트하여 추가
            # EXACT COPY: test_component_separation.py line 326
            x_reshaped = x_reshaped + visual_temporal_emb.view(1, T, 1, vision_features.shape[-1])
            # 다시 (batch, T*seq_len, hidden_dim) 형태로 변환
            # EXACT COPY: test_component_separation.py line 328
            vision_features = x_reshaped.view(batch_size, T * x_reshaped.shape[2], vision_features.shape[-1])
        
        # ============================================================
        # Step 3: image_feature_source_pooling
        # ============================================================
        # EXACT COPY from test_component_separation.py lines 354-408
        x_reshaped = vision_features.view(batch_size, T, -1, vision_features.shape[-1])
        
        # 기본값: last_frame 사용 (하지만 model에서 가져옴)
        if hasattr(model, 'image_feature_source'):
            image_feature_source = model.image_feature_source
        else:
            image_feature_source = ['last_frame']
        
        # 각 집계 방식으로 특징 계산
        # 주의: 각 pooling 방식의 출력 차원이 다릅니다!
        x_feat_dict = {
            'spatial_avg_pool': x_reshaped.mean(dim=2),  # 공간 평균: (batch, T, hidden_dim)
            'temporal_avg_pool': x_reshaped.mean(dim=1),  # 시간 평균: (batch, seq_len, hidden_dim)
            'last_frame': x_reshaped[:, -1]                # 마지막 프레임: (batch, seq_len, hidden_dim)
        }
        
        # 설정된 집계 방식들을 결합
        new_x = []
        for _image_feature_source in image_feature_source:
            if _image_feature_source not in x_feat_dict:
                raise ValueError(f'invalid image feature source: {_image_feature_source}')
            new_x.append(x_feat_dict[_image_feature_source])
        
        # 여러 집계 결과를 시퀀스 차원으로 연결
        vision_features = torch.cat(new_x, dim=1)
        
        # ============================================================
        # Step 4: image_projection (1024 -> 768)
        # ============================================================
        vision_features = vision_features @ model.image_projection
        
        # ============================================================
        # Step 5: image_proj_norm
        # ============================================================
        vision_features = model.image_proj_norm(vision_features)
        
        # Now vision_features is (batch, seq_len, 768) - ready for language_model
        
        # ============================================================
        # Step 6: Merge vision features with task embeddings
        # ============================================================
        # Note: task_prompts should be text_embeddings (shared from top level) or task prompt strings
        # It is recommended to generate text_embeddings at top level (main/test_semcom.py)
        # using processor with all-zero dummy image, and share them with Tx/Rx
        if isinstance(task_prompts, list):
            # Task prompts are strings - encode them
            # Note: For consistency, this should be done at top level and shared
            # But we support it here for backward compatibility
            from PIL import Image as PILImage
            import numpy as np
            
            # Create dummy image for processor (required by Florence2Processor)
            dummy_image = PILImage.fromarray(np.zeros((768, 768, 3), dtype=np.uint8))
            
            inputs = processor(
                text=task_prompts,
                images=[dummy_image] * len(task_prompts),
                return_tensors="pt"
            )
            input_ids = inputs["input_ids"].to(
                device=vision_features.device,
                dtype=torch.long
            )
            embedding_layer = model.get_input_embeddings()
            text_embeddings = embedding_layer(input_ids)
        else:
            # task_prompts is already text_embeddings (from Tx) - use directly
            text_embeddings = task_prompts
        
        # Merge vision and text embeddings using Florence-2's method
        merged_embeds, attention_mask = model._merge_input_ids_with_image_features(
            vision_features, text_embeddings
        )
        
        return merged_embeds, attention_mask

    def generate(
        self,
        received_vision_embedding: torch.Tensor,
        task_prompts: Union[List[str], torch.Tensor],
        max_new_tokens: int = 1024,
        num_beams: int = 3,
        do_sample: bool = False,
    ) -> torch.Tensor:
        """
        Generate text using vision_tower features from transmitter.
        
        Method 2 Flow:
        1. Process vision_tower features through remaining pipeline (forward)
        2. Call language_model.generate with merged embeddings
        
        Steps:
        1. Process received_vision_embedding (vision_tower output) through:
           - image_pos_embed
           - visual_temporal_embed
           - image_feature_source_pooling
           - image_projection
           - image_proj_norm
        2. Merge with task_embeddings
        3. Call language_model.generate()
        
        Args:
            received_vision_embedding: Vision tower features from transmitter
                Shape: (batch, seq_len, 1024)
            task_prompts: Task prompts (list of strings) or text_embeddings (tensor)
                Recommended: text_embeddings generated at top level (main/test_semcom.py)
                using processor with all-zero dummy image, and shared with Tx/Rx
                If strings, will be encoded using processor with dummy image (backward compatibility)
            max_new_tokens: Maximum number of tokens to generate
            num_beams: Number of beams for beam search
            do_sample: Whether to use sampling
        
        Returns:
            Generated token IDs: (batch, generated_seq_len)
        """
        model = self.florence2_model.model
        language_model = model.language_model

        # Process vision features and merge with task embeddings
        merged_embeds, attention_mask = self.forward(
            received_vision_embedding, task_prompts
        )

        # Generate using language_model
        generated_ids = language_model.generate(
            input_ids=None,
            inputs_embeds=merged_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            num_beams=num_beams,
            use_cache=False,
        )

        return generated_ids
