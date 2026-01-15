"""
Tx/Rx 분리점 비교를 위한 구현 예시
방법 1: image_proj_norm 이후 compression
방법 2: vision_tower 이후 compression (추천)
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Union, List
from enum import Enum

from models.florence2_model import Florence2Model


class CompressionPoint(Enum):
    """Compression 적용 시점"""
    AFTER_VISION_TOWER = "after_vision_tower"  # 방법 2 (추천)
    AFTER_PROJ_NORM = "after_proj_norm"        # 방법 1


class TransmitterV2(nn.Module):
    """
    Transmitter with configurable compression point.
    
    Methods:
    1. AFTER_VISION_TOWER: vision_tower 이후 compression (방법 2) - 추천
       - Output: compressed vision tower features (576, 1024)
       - 장점: 원시 특징 보존, spatial-aware compression 가능
       
    2. AFTER_PROJ_NORM: image_proj_norm 이후 compression (방법 1)
       - Output: compressed projected features (577, 768)
       - 장점: 이미 압축된 형태, Tx 계산 부담 적음
    """
    
    def __init__(
        self,
        florence2_model: Florence2Model,
        compression_point: CompressionPoint = CompressionPoint.AFTER_VISION_TOWER,
        # Compression 모듈은 나중에 추가
        compressor: Optional[nn.Module] = None,
        task_embedding_dim: int = 768,
        use_pooled_features: bool = False
    ):
        super().__init__()
        
        self.florence2_model = florence2_model
        self.compression_point = compression_point
        self.compressor = compressor  # Compression 모듈 (나중에 추가)
        self.task_embedding_dim = task_embedding_dim
        self.use_pooled_features = use_pooled_features
        
        # Vision encoder output dimension
        self.vision_dim = florence2_model.get_vision_dim()  # 1024 for base
        
        # Output dimension depends on compression point
        if compression_point == CompressionPoint.AFTER_VISION_TOWER:
            # Output: vision_tower features (1024 dim)
            self.output_dim = self.vision_dim  # 1024
            self.output_seq_len = None  # Depends on image size (typically 576)
        elif compression_point == CompressionPoint.AFTER_PROJ_NORM:
            # Output: projected features (768 dim)
            self.output_dim = 768
            self.output_seq_len = None  # Depends on feature source (typically 577)
        else:
            raise ValueError(f"Unknown compression_point: {compression_point}")
    
    def forward(
        self,
        images: Union[torch.Tensor, List, "PIL.Image.Image"]
    ) -> torch.Tensor:
        """
        Process images through vision encoder up to compression point.
        
        Args:
            images: Input images
            
        Returns:
            Features ready for compression
            - AFTER_VISION_TOWER: (batch, seq_len, 1024)
            - AFTER_PROJ_NORM: (batch, seq_len, 768)
        """
        model = self.florence2_model.model
        processor = self.florence2_model.processor
        
        # Preprocess images
        if not isinstance(images, torch.Tensor):
            if not isinstance(images, list):
                images = [images]
            inputs = processor(images=images, return_tensors="pt")
            pixel_values = inputs["pixel_values"]
            device = next(model.parameters()).device
            pixel_values = pixel_values.to(device=device, dtype=model.dtype)
        else:
            pixel_values = images
            if pixel_values.dtype != model.dtype:
                pixel_values = pixel_values.to(dtype=model.dtype)
        
        if len(pixel_values.shape) != 4:
            raise ValueError(f'invalid image shape {pixel_values.shape}')
        
        batch_size, C, H, W = pixel_values.shape
        T = 1
        
        # ============================================================
        # Step 1: Vision Tower (항상 수행)
        # ============================================================
        x = model.vision_tower.forward_features_unpool(pixel_values)
        if isinstance(x, tuple):
            vision_features = x[0]
        else:
            vision_features = x
        
        # 방법 2: vision_tower 이후 compression
        if self.compression_point == CompressionPoint.AFTER_VISION_TOWER:
            # Output: (batch, seq_len, 1024)
            # Compression will be applied here (if compressor is set)
            if self.compressor is not None:
                return self.compressor(vision_features)
            return vision_features
        
        # ============================================================
        # Step 2-7: 나머지 처리 (방법 1만)
        # ============================================================
        # Step 2: image_pos_embed
        if hasattr(model, 'image_pos_embed') and model.image_pos_embed is not None:
            x = vision_features.view(batch_size * T, -1, vision_features.shape[-1])
            num_tokens = x.shape[-2]
            h = w = int(num_tokens ** 0.5)
            
            if h * w == num_tokens:
                x = x.view(batch_size * T, h, w, x.shape[-1])
                pos_embed = model.image_pos_embed(x)
                x = x + pos_embed
                x = x.view(batch_size, T * h * w, x.shape[-1])
        
        # Step 3: visual_temporal_embed
        if hasattr(model, 'visual_temporal_embed') and model.visual_temporal_embed is not None:
            x_reshaped = x.view(batch_size, T, -1, x.shape[-1])
            visual_temporal_emb = model.visual_temporal_embed(x_reshaped[:, :, 0])
            x_reshaped = x_reshaped + visual_temporal_emb.view(1, T, 1, x.shape[-1])
            x = x_reshaped.view(batch_size, T * x_reshaped.shape[2], x.shape[-1])
        
        # Step 4: image_feature_source pooling
        x_reshaped = x.view(batch_size, T, -1, x.shape[-1])
        
        if hasattr(model, 'image_feature_source'):
            image_feature_source = model.image_feature_source
        else:
            image_feature_source = ['last_frame']
        
        x_feat_dict = {
            'spatial_avg_pool': x_reshaped.mean(dim=2),
            'temporal_avg_pool': x_reshaped.mean(dim=1),
            'last_frame': x_reshaped[:, -1]
        }
        
        new_x = []
        for _image_feature_source in image_feature_source:
            if _image_feature_source not in x_feat_dict:
                raise ValueError(f'invalid image feature source: {_image_feature_source}')
            new_x.append(x_feat_dict[_image_feature_source])
        
        x = torch.cat(new_x, dim=1)
        
        # Step 5: image_projection
        x = x @ model.image_projection
        
        # Step 6: image_proj_norm
        x = model.image_proj_norm(x)
        
        # 방법 1: image_proj_norm 이후 compression
        if self.compression_point == CompressionPoint.AFTER_PROJ_NORM:
            # Output: (batch, seq_len, 768)
            # Compression will be applied here (if compressor is set)
            if self.compressor is not None:
                return self.compressor(x)
            return x
        
        raise RuntimeError(f"Should not reach here: compression_point={self.compression_point}")
    
    def get_output_dim(self) -> int:
        """Get output dimension."""
        return self.output_dim


class ReceiverV2(nn.Module):
    """
    Receiver with configurable decompression point.
    Matches the compression point of Transmitter.
    """
    
    def __init__(
        self,
        florence2_model: Florence2Model,
        compression_point: CompressionPoint = CompressionPoint.AFTER_VISION_TOWER,
        decompressor: Optional[nn.Module] = None,
        use_pooled_features: bool = False
    ):
        super().__init__()
        
        self.florence2_model = florence2_model
        self.compression_point = compression_point
        self.decompressor = decompressor  # Decompression 모듈
        self.use_pooled_features = use_pooled_features
    
    def forward(
        self,
        received_features: torch.Tensor,
        task_prompts: Union[List[str], torch.Tensor]
    ) -> torch.Tensor:
        """
        Process received features through remaining pipeline.
        
        Args:
            received_features: Compressed features from channel
            task_prompts: Task prompts
            
        Returns:
            Combined embeddings ready for language model
        """
        model = self.florence2_model.model
        processor = self.florence2_model.processor
        
        # ============================================================
        # Step 1: Decompression
        # ============================================================
        if self.decompressor is not None:
            vision_features = self.decompressor(received_features)
        else:
            vision_features = received_features
        
        batch_size = vision_features.shape[0]
        T = 1
        
        # ============================================================
        # 방법 2: vision_tower 이후 compression인 경우
        # 나머지 모든 단계 수행
        # ============================================================
        if self.compression_point == CompressionPoint.AFTER_VISION_TOWER:
            # Step 2: image_pos_embed
            if hasattr(model, 'image_pos_embed') and model.image_pos_embed is not None:
                x = vision_features.view(batch_size * T, -1, vision_features.shape[-1])
                num_tokens = x.shape[-2]
                h = w = int(num_tokens ** 0.5)
                
                if h * w == num_tokens:
                    x = x.view(batch_size * T, h, w, x.shape[-1])
                    pos_embed = model.image_pos_embed(x)
                    x = x + pos_embed
                    x = x.view(batch_size, T * h * w, x.shape[-1])
                    vision_features = x
            
            # Step 3: visual_temporal_embed
            if hasattr(model, 'visual_temporal_embed') and model.visual_temporal_embed is not None:
                x_reshaped = vision_features.view(batch_size, T, -1, vision_features.shape[-1])
                visual_temporal_emb = model.visual_temporal_embed(x_reshaped[:, :, 0])
                x_reshaped = x_reshaped + visual_temporal_emb.view(1, T, 1, vision_features.shape[-1])
                vision_features = x_reshaped.view(batch_size, T * x_reshaped.shape[2], vision_features.shape[-1])
            
            # Step 4: image_feature_source pooling
            x_reshaped = vision_features.view(batch_size, T, -1, vision_features.shape[-1])
            
            if hasattr(model, 'image_feature_source'):
                image_feature_source = model.image_feature_source
            else:
                image_feature_source = ['last_frame']
            
            x_feat_dict = {
                'spatial_avg_pool': x_reshaped.mean(dim=2),
                'temporal_avg_pool': x_reshaped.mean(dim=1),
                'last_frame': x_reshaped[:, -1]
            }
            
            new_x = []
            for _image_feature_source in image_feature_source:
                if _image_feature_source not in x_feat_dict:
                    raise ValueError(f'invalid image feature source: {_image_feature_source}')
                new_x.append(x_feat_dict[_image_feature_source])
            
            vision_features = torch.cat(new_x, dim=1)
            
            # Step 5: image_projection
            vision_features = vision_features @ model.image_projection
            
            # Step 6: image_proj_norm
            vision_features = model.image_proj_norm(vision_features)
        
        # ============================================================
        # 방법 1: image_proj_norm 이후 compression인 경우
        # 이미 모든 처리가 끝났으므로 바로 language_model 사용
        # ============================================================
        elif self.compression_point == CompressionPoint.AFTER_PROJ_NORM:
            # vision_features는 이미 image_proj_norm 출력
            pass
        else:
            raise ValueError(f"Unknown compression_point: {self.compression_point}")
        
        # ============================================================
        # Step 7: Language Model (공통)
        # ============================================================
        # Encode task prompts
        if isinstance(task_prompts, list):
            tokenizer = processor.tokenizer
            inputs = tokenizer(task_prompts, return_tensors="pt", padding=True)
            input_ids = inputs["input_ids"].to(
                device=vision_features.device,
                dtype=torch.long
            )
            embedding_layer = model.get_input_embeddings()
            text_embeddings = embedding_layer(input_ids)
        else:
            text_embeddings = task_prompts
        
        # Merge vision and text embeddings
        merged_embeds, attention_mask = model._merge_input_ids_with_image_features(
            vision_features, text_embeddings
        )
        
        return merged_embeds, attention_mask
    
    def generate(
        self,
        received_features: torch.Tensor,
        task_prompts: List[str],
        max_new_tokens: int = 1024,
        num_beams: int = 3,
        do_sample: bool = False,
    ) -> torch.Tensor:
        """Generate text using language model."""
        model = self.florence2_model.model
        language_model = model.language_model
        
        merged_embeds, attention_mask = self.forward(received_features, task_prompts)
        
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


# ====================================================================
# 사용 예시
# ====================================================================

def example_usage():
    """두 가지 방법 사용 예시"""
    
    # Florence-2 모델 로드
    from models.florence2_model import Florence2Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    florence2_model = Florence2Model(
        model_name="microsoft/Florence-2-base",
        device=device
    )
    
    # ================================================================
    # 방법 2: Vision Tower 이후 compression (추천)
    # ================================================================
    print("=" * 70)
    print("방법 2: Vision Tower 이후 compression")
    print("=" * 70)
    
    tx_method2 = TransmitterV2(
        florence2_model=florence2_model,
        compression_point=CompressionPoint.AFTER_VISION_TOWER,
        compressor=None,  # 실제 compression 모듈 추가 필요
    )
    
    rx_method2 = ReceiverV2(
        florence2_model=florence2_model,
        compression_point=CompressionPoint.AFTER_VISION_TOWER,
        decompressor=None,  # 실제 decompression 모듈 추가 필요
    )
    
    print(f"Tx output dimension: {tx_method2.get_output_dim()}")  # 1024
    print("Tx output: (batch, ~576, 1024)")
    print("→ Compression 여기서 적용")
    print("→ Rx에서 나머지 모든 처리 수행")
    
    # ================================================================
    # 방법 1: image_proj_norm 이후 compression
    # ================================================================
    print("\n" + "=" * 70)
    print("방법 1: image_proj_norm 이후 compression")
    print("=" * 70)
    
    tx_method1 = TransmitterV2(
        florence2_model=florence2_model,
        compression_point=CompressionPoint.AFTER_PROJ_NORM,
        compressor=None,
    )
    
    rx_method1 = ReceiverV2(
        florence2_model=florence2_model,
        compression_point=CompressionPoint.AFTER_PROJ_NORM,
        decompressor=None,
    )
    
    print(f"Tx output dimension: {tx_method1.get_output_dim()}")  # 768
    print("Tx output: (batch, ~577, 768)")
    print("→ Compression 여기서 적용")
    print("→ Rx에서 language_model만 수행")


if __name__ == "__main__":
    example_usage()
