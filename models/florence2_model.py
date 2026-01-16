"""
Florence-2 model wrapper.
Handles loading and using Florence-2's vision encoder.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
import warnings

try:
    from transformers import AutoProcessor, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    warnings.warn(
        "transformers library not available. "
        "Please install it: pip install transformers"
    )


def get_vision_encoder_output_dim(model_size: str = "base") -> int:
    """
    Get the output dimension of Florence-2's vision encoder (DaViT).
    
    Args:
        model_size: Model size ('base', 'large')
        
    Returns:
        Output dimension of vision encoder
        
    Note:
        - Florence-2-base uses DaViT-base with output dimension 768
        - Florence-2-large uses DaViT-large with output dimension 1024
    """
    dim_map = {
        'base': 768,
        'large': 1024,
        'florence-2-base': 768,
        'florence-2-large': 1024,
    }
    
    model_size_lower = model_size.lower()
    if model_size_lower in dim_map:
        return dim_map[model_size_lower]
    
    # Default to base
    return 768


class Florence2Model(nn.Module):
    """
    Florence-2 model wrapper for semantic communication.
    
    Args:
        model_name: HuggingFace model name (default: 'microsoft/Florence-2-base')
        device: Device to run the model on
    """
    
    def __init__(
        self,
        model_name: str = "microsoft/Florence-2-base",
        device: Optional[torch.device] = None
    ):
        super().__init__()
        
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers library is required. "
                "Install it with: pip install transformers"
            )
        
        self.model_name = model_name
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Determine torch dtype based on device
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        # Load Florence-2 model using AutoModelForCausalLM (as per HuggingFace documentation)
        # Use attn_implementation="eager" to avoid _supports_sdpa attribute error
        print(f"Loading Florence-2 model: {model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch_dtype,  # Use dtype instead of torch_dtype (deprecated)
            attn_implementation="eager",  # Fix for _supports_sdpa attribute error
            trust_remote_code=True
        ).to(self.device)
        
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        # Extract vision encoder
        # Florence-2 structure: model has vision_tower (vision encoder) and language_model (text encoder/decoder)
        
        if hasattr(self.model, 'vision_tower'):
            self.vision_encoder = self.model.vision_tower
        elif hasattr(self.model, 'florence') and hasattr(self.model.florence, 'vision_encoder'):
            self.vision_encoder = self.model.florence.vision_encoder
        else:
            raise AttributeError("Could not find vision encoder in model. Available attributes: " + 
                               str([attr for attr in dir(self.model) if not attr.startswith('_')]))
        
        self.vision_encoder.eval()  # Set to eval mode for inference
        
        # Extract transformer encoder and decoder (for receiver)
        # Florence-2 structure: language_model.model contains encoder and decoder
        # Note: language_model itself is the wrapper, encoder/decoder are in language_model.model
        if hasattr(self.model, 'language_model'):
            # Check if language_model has a 'model' attribute (which contains encoder/decoder)
            if hasattr(self.model.language_model, 'model'):
                # Florence-2 structure: language_model.model.encoder and language_model.model.decoder
                self.text_encoder = getattr(self.model.language_model.model, 'encoder', None)
                self.text_decoder = getattr(self.model.language_model.model, 'decoder', None)
            else:
                # Fallback: try direct access
                self.text_encoder = getattr(self.model.language_model, 'encoder', None)
                self.text_decoder = getattr(self.model.language_model, 'decoder', None)
            
            # If still no decoder found, use language_model itself (for generate method)
            if self.text_decoder is None:
                self.text_decoder = self.model.language_model
        elif hasattr(self.model, 'florence'):
            # Fallback to florence structure if it exists
            self.text_encoder = getattr(self.model.florence, 'text_encoder', None)
            self.text_decoder = getattr(self.model.florence, 'text_decoder', None)
        else:
            self.text_encoder = None
            self.text_decoder = None
        
        # Get output dimension
        # Florence-2-base uses DaViT-base: output dim = 768
        # Florence-2-large uses DaViT-large: output dim = 1024
        if 'large' in model_name.lower():
            self.vision_dim = 1024
        else:
            self.vision_dim = 768
        
        print(f"Vision encoder output dimension: {self.vision_dim}")
    
    def encode_image(
        self,
        images: torch.Tensor,
        return_pooled: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Encode images using Florence-2's vision encoder.
        
        Args:
            images: Input images tensor of shape (batch_size, 3, H, W)
            return_pooled: Whether to return pooled features (CLS token)
            
        Returns:
            - vision_features: Vision features from encoder
            - pooled_features: Pooled features (if return_pooled=True)
            
        Note:
            DaViT output shape: (batch_size, num_patches, vision_dim)
            - num_patches depends on image size and patch size
            - For 224x224 image with patch size 16: num_patches = (224/16)^2 = 196
        """
        with torch.no_grad():
            # Ensure input dtype matches model dtype
            model_dtype = next(self.vision_encoder.parameters()).dtype
            if images.dtype != model_dtype:
                images = images.to(dtype=model_dtype)
            
            # Forward through vision encoder using forward_features_unpool method
            # DaViT uses forward_features_unpool which returns (features, size) tuple
            
            if hasattr(self.vision_encoder, 'forward_features_unpool'):
                vision_outputs = self.vision_encoder.forward_features_unpool(images)
                # forward_features_unpool returns (features, size) tuple
                if isinstance(vision_outputs, tuple):
                    vision_features = vision_outputs[0]
                else:
                    vision_features = vision_outputs
            elif hasattr(self.vision_encoder, 'forward_features'):
                vision_features = self.vision_encoder.forward_features(images)
                if isinstance(vision_features, tuple):
                    vision_features = vision_features[0]
            else:
                # Fallback to regular forward
                vision_outputs = self.vision_encoder(images)
                # Extract features
                if hasattr(vision_outputs, 'last_hidden_state'):
                    vision_features = vision_outputs.last_hidden_state
                elif isinstance(vision_outputs, tuple):
                    vision_features = vision_outputs[0]
                else:
                    vision_features = vision_outputs
            
            # Get pooled features (CLS token) if available
            pooled_features = None
            if return_pooled:
                if hasattr(vision_outputs, 'pooler_output') and vision_outputs.pooler_output is not None:
                    pooled_features = vision_outputs.pooler_output
                else:
                    # Use first token (CLS token) as pooled feature
                    pooled_features = vision_features[:, 0, :]
            
            return vision_features, pooled_features
    
    def forward(
        self,
        images: torch.Tensor,
        task_prompts: Optional[list] = None
    ) -> torch.Tensor:
        """
        Forward pass through vision encoder.
        
        Args:
            images: Input images
            task_prompts: Optional task prompts (not used in encoder-only mode)
            
        Returns:
            Vision features
        """
        vision_features, _ = self.encode_image(images)
        return vision_features
    
    def get_vision_dim(self) -> int:
        """Get vision encoder output dimension."""
        return self.vision_dim
    
    def encode_task_prompt(
        self,
        task_prompts: list,
        return_tensors: str = "pt"
    ) -> torch.Tensor:
        """
        Encode task prompts using Florence-2's processor.
        
        Args:
            task_prompts: List of task prompt strings
            return_tensors: Return format ('pt' for PyTorch tensors)
            
        Returns:
            Task embeddings (text embeddings) from Florence-2 processor
        """
        # Use processor to encode task prompts
        # Note: processor may require images, so we'll use tokenizer directly if available
        if hasattr(self.processor, 'tokenizer'):
            # Use tokenizer directly for text-only encoding
            inputs = self.processor.tokenizer(
                task_prompts,
                return_tensors=return_tensors,
                padding=True
            )
        else:
            # Fallback: try processor with dummy image
            # This will create text embeddings that can be combined with vision embeddings
            inputs = self.processor(
                text=task_prompts,
                return_tensors=return_tensors,
                padding=True
            )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get text embeddings using language_model if available
        if hasattr(self.model, 'language_model'):
            with torch.no_grad():
                # Use language_model to encode text
                if hasattr(self.model.language_model, 'get_input_embeddings'):
                    # Get embedding layer
                    embedding_layer = self.model.language_model.get_input_embeddings()
                    input_ids = inputs.get('input_ids')
                    if input_ids is not None:
                        text_embeddings = embedding_layer(input_ids)
                    else:
                        raise ValueError("input_ids not found in processor output")
                else:
                    # Fallback: try to use text_encoder if available
                    if self.text_encoder is not None:
                        text_outputs = self.text_encoder(**inputs)
                        if hasattr(text_outputs, 'last_hidden_state'):
                            text_embeddings = text_outputs.last_hidden_state
                        elif isinstance(text_outputs, tuple):
                            text_embeddings = text_outputs[0]
                        else:
                            text_embeddings = text_outputs
                    else:
                        # Last fallback: use input_ids as placeholder
                        text_embeddings = inputs.get('input_ids', None)
                        if text_embeddings is None:
                            raise ValueError("Could not generate task embeddings")
        elif self.text_encoder is not None:
            with torch.no_grad():
                text_outputs = self.text_encoder(**inputs)
                if hasattr(text_outputs, 'last_hidden_state'):
                    text_embeddings = text_outputs.last_hidden_state
                elif isinstance(text_outputs, tuple):
                    text_embeddings = text_outputs[0]
                else:
                    text_embeddings = text_outputs
        else:
            # Fallback: use input_ids as placeholder
            text_embeddings = inputs.get('input_ids', None)
            if text_embeddings is None:
                raise ValueError("Could not generate task embeddings")
        
        return text_embeddings
    
    def get_transformer_encoder(self):
        """Get transformer encoder component."""
        return self.text_encoder
    
    def get_transformer_decoder(self):
        """Get transformer decoder component."""
        return self.text_decoder
    
    def generate(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        max_new_tokens: int = 1024,
        do_sample: bool = False,
        num_beams: int = 3,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate text using Florence-2 model.
        
        This method wraps the model's generate method with the fixes learned from
        test_florence2.py: use_cache=False to avoid past_key_values None error.
        
        Args:
            input_ids: Input token IDs
            pixel_values: Image pixel values
            max_new_tokens: Maximum number of new tokens to generate
            do_sample: Whether to use sampling
            num_beams: Number of beams for beam search
            **kwargs: Additional arguments for generate
            
        Returns:
            Generated token IDs
        """
        # Use use_cache=False to avoid past_key_values None error (learned from test_florence2.py)
        return self.model.generate(
            input_ids=input_ids,
            pixel_values=pixel_values,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            num_beams=num_beams,
            use_cache=False,  # Required to avoid past_key_values None error
            **kwargs
        )