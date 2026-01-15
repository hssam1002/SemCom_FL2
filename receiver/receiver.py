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
    and processes through Florence-2's transformer encoder and decoder.
    
    Flow: received_vision_embedding + task_embedding -> transformer_encoder -> transformer_decoder
    
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
        
        # Get vision encoder output dimension
        self.vision_dim = florence2_model.get_vision_dim()
        
        # Task embedding dimension (typically 768 for Florence-2-base text embeddings)
        # We'll detect it dynamically, but default to 768
        self.task_embedding_dim = 768
        
        # Projection layer to match task embeddings to vision dimension if needed
        # This will be created dynamically when we first see the task embedding dimension
        self.task_projection = None
        
        # Access Florence-2's transformer encoder and decoder
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
                If strings, will be encoded using Florence-2's processor
                If tensor, assumed to be pre-encoded task embeddings
                
        Returns:
            Decoded output from transformer decoder
        """
        # Encode task prompts if they are strings
        if isinstance(task_prompts, list):
            # Use Florence-2's processor to encode task prompts
            task_embeddings = self.florence2_model.encode_task_prompt(task_prompts)
        else:
            # Assume task_prompts is already a tensor (task embeddings)
            task_embeddings = task_prompts
        
        # Combine vision embedding with task embeddings
        # Florence-2 expects: vision tokens + text tokens
        if self.use_pooled_features:
            # received_vision_embedding: (batch_size, vision_dim)
            # Need to expand to sequence format for transformer
            # Add sequence dimension: (batch_size, 1, vision_dim)
            vision_tokens = received_vision_embedding.unsqueeze(1)
        else:
            # received_vision_embedding: (batch_size, seq_len, vision_dim)
            vision_tokens = received_vision_embedding
        
        # Ensure dtype consistency between vision_tokens and task_embeddings
        # Match vision_tokens dtype to task_embeddings dtype (usually float16 for Florence-2)
        if vision_tokens.dtype != task_embeddings.dtype:
            vision_tokens = vision_tokens.to(dtype=task_embeddings.dtype)
        
        # Check and fix dimension mismatch between vision_tokens and task_embeddings
        # vision_tokens: (batch_size, seq_len, vision_dim)
        # task_embeddings: (batch_size, task_seq_len, task_embedding_dim)
        vision_dim = vision_tokens.shape[-1]
        task_dim = task_embeddings.shape[-1]
        
        if vision_dim != task_dim:
            # Create projection layer if it doesn't exist or dimensions changed
            if self.task_projection is None or self.task_projection.out_features != vision_dim:
                # Match dtype with task_embeddings
                task_dtype = task_embeddings.dtype
                self.task_projection = nn.Linear(task_dim, vision_dim).to(
                    device=task_embeddings.device,
                    dtype=task_dtype
                )
                self.task_embedding_dim = task_dim
            
            # Project task_embeddings to match vision_dim
            task_embeddings = self.task_projection(task_embeddings)
        
        # Combine vision tokens and task embeddings
        # Florence-2 concatenates them: [vision_tokens, task_embeddings]
        combined_input = torch.cat([vision_tokens, task_embeddings], dim=1)
        
        # Florence-2's language_model processes the combined input directly
        # The language_model is a decoder, so we need to use decoder_inputs_embeds
        if hasattr(self.florence2_model.model, 'language_model'):
            try:
                # Use language_model as decoder with combined embeddings
                # Since it's a decoder, use decoder_inputs_embeds
                language_output = self.florence2_model.model.language_model(
                    decoder_inputs_embeds=combined_input
                )
                
                # Extract output from language model
                if hasattr(language_output, 'last_hidden_state'):
                    output = language_output.last_hidden_state
                elif isinstance(language_output, tuple):
                    output = language_output[0]
                else:
                    output = language_output
            except Exception as e:
                # Try with inputs_embeds if decoder_inputs_embeds doesn't work
                try:
                    language_output = self.florence2_model.model.language_model(
                        inputs_embeds=combined_input
                    )
                    if hasattr(language_output, 'last_hidden_state'):
                        output = language_output.last_hidden_state
                    elif isinstance(language_output, tuple):
                        output = language_output[0]
                    else:
                        output = language_output
                except Exception as e2:
                    print(f"Warning: Error in language_model: {e2}")
                    # Fallback: return combined input
                    output = combined_input
        else:
            # Fallback: if no language_model, try encoder/decoder separately
            if self.transformer_encoder is not None:
                try:
                    encoder_output = self.transformer_encoder(
                        inputs_embeds=combined_input
                    )
                    if hasattr(encoder_output, 'last_hidden_state'):
                        encoder_hidden = encoder_output.last_hidden_state
                    elif isinstance(encoder_output, tuple):
                        encoder_hidden = encoder_output[0]
                    else:
                        encoder_hidden = encoder_output
                except Exception as e:
                    print(f"Warning: Error in transformer encoder: {e}")
                    encoder_hidden = combined_input
            else:
                encoder_hidden = combined_input
            
            # Use encoder output directly as fallback
            output = encoder_hidden
        
        return output
