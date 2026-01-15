"""
Receiver module.
Processes received signals by combining with task embedding and 
passing through Florence-2's transformer encoder/decoder.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Union, List

from models.florence2_model import Florence2Model


class Receiver(nn.Module):
    """
    Receiver for semantic communication.
    
    Combines received vision embedding with task embedding (from Florence-2)
    and processes through Florence-2's language model.
    
    Flow: received_vision_embedding + task_embedding -> merge -> language_model.generate
    
    Args:
        florence2_model: Florence-2 model instance
        use_pooled_features: Whether transmitter used pooled features
    """
    
    def __init__(
        self,
        florence2_model: Florence2Model,
        use_pooled_features: bool = False
    ):
        super().__init__()
        
        self.florence2_model = florence2_model
        self.use_pooled_features = use_pooled_features
        
        # Transmitter now outputs 768 dimension (after image_proj_norm)
        # This matches the language model hidden dimension
        self.vision_dim = 768  # Transmitter output dimension (image_proj_norm output)

        # Task embedding dimension (768 for Florence-2-base text embeddings)
        self.task_embedding_dim = 768
        
        # No projection needed since both vision and task embeddings are 768
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
    ) -> torch.Tensor:
        """
        Process received vision embedding with task prompts through
        Florence-2's transformer encoder and decoder.
        
        Flow:
        1. received_vision_embedding (from channel) + task_prompts
        2. -> transformer_encoder
        3. -> transformer_decoder
        4. -> output
        
        Args:
            received_vision_embedding: Received vision embedding from channel
                - If use_pooled_features=True: (batch_size, vision_dim)
                - If use_pooled_features=False: (batch_size, seq_len, vision_dim)
            task_prompts: Task prompts (list of strings) or task embeddings (tensor)
                If strings, will be encoded using Florence-2's processor/tokenizer
                If tensor, assumed to be pre-encoded task embeddings
        
        Returns:
            Combined encoder input embeddings (vision + text), shape (batch, seq_len_total, hidden)
        """
        # Encode task prompts using tokenizer + embedding layer
        processor = self.florence2_model.processor
        model = self.florence2_model.model

        if isinstance(task_prompts, list):
            tokenizer = processor.tokenizer
            inputs = tokenizer(
                task_prompts,
                return_tensors="pt",
                padding=True
            )
            input_ids = inputs["input_ids"].to(
                device=received_vision_embedding.device,
                dtype=torch.long
            )

            # Get text embeddings using the same embedding layer as Florence-2
            embedding_layer = model.get_input_embeddings()
            task_embeddings = embedding_layer(input_ids)
        else:
            task_embeddings = task_prompts

        # Combine vision embedding with task embeddings
        if self.use_pooled_features:
            vision_tokens = received_vision_embedding.unsqueeze(1)
        else:
            vision_tokens = received_vision_embedding

        # Ensure dtype consistency
        if vision_tokens.dtype != task_embeddings.dtype:
            vision_tokens = vision_tokens.to(dtype=task_embeddings.dtype)

        # Optional safety projection if dims mismatch
        vision_dim = vision_tokens.shape[-1]
        task_dim = task_embeddings.shape[-1]
        if vision_dim != task_dim:
            print(f"Warning: Dimension mismatch: vision_dim={vision_dim}, task_dim={task_dim}")
            if self.task_projection is None or self.task_projection.out_features != vision_dim:
                self.task_projection = nn.Linear(task_dim, vision_dim).to(
                    device=task_embeddings.device,
                    dtype=task_embeddings.dtype
                )
            task_embeddings = self.task_projection(task_embeddings)

        # Concatenate [vision_tokens, task_embeddings]
        combined_input = torch.cat([vision_tokens, task_embeddings], dim=1)
        return combined_input

    def generate(
        self,
        received_vision_embedding: torch.Tensor,
        task_prompts: List[str],
        max_new_tokens: int = 1024,
        num_beams: int = 3,
        do_sample: bool = False,
    ) -> torch.Tensor:
        """
        Generate text using the same flow as Florence-2's generate, but
        using precomputed image features from the transmitter instead of pixel_values.

        Steps:
        1. Encode task_prompts to input_ids
        2. Get text embeddings via model.get_input_embeddings()
        3. Merge image features (from Tx) with text embeddings using
           Florence-2's _merge_input_ids_with_image_features
        4. Call language_model.generate(input_ids=None, inputs_embeds=merged_embeds, ...)
        """
        processor = self.florence2_model.processor
        model = self.florence2_model.model  # Florence2ForConditionalGeneration
        language_model = model.language_model

        # 1. Encode task prompts to input_ids (same tokenizer as reference)
        tokenizer = processor.tokenizer
        inputs = tokenizer(
            task_prompts,
            return_tensors="pt",
            padding=True
        )
        input_ids = inputs["input_ids"].to(
            device=received_vision_embedding.device,
            dtype=torch.long
        )

        # 2. Get text embeddings via model.get_input_embeddings()
        embedding_layer = model.get_input_embeddings()
        text_embeddings = embedding_layer(input_ids)  # (batch, text_len, hidden)

        # 3. Merge image features (from Tx) with text embeddings using
        #    Florence-2's _merge_input_ids_with_image_features
        image_features = received_vision_embedding  # (batch, image_token_len, hidden)
        if image_features.dtype != text_embeddings.dtype:
            image_features = image_features.to(dtype=text_embeddings.dtype)

        merged_embeds, attention_mask = model._merge_input_ids_with_image_features(
            image_features, text_embeddings
        )

        # 4. Call language_model.generate with inputs_embeds (same as model.generate does)
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
