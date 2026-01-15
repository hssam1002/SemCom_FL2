"""
Task Embedding module for shared task prompt/embedding.
Note: Task embedding is actually handled by Florence-2's processor internally.
This module is kept for compatibility and utility functions.
"""

import torch
import torch.nn as nn
from typing import Union, Optional


class TaskEmbedding(nn.Module):
    """
    Task embedding module that can handle both task prompts and task embeddings.
    
    Args:
        task_dim: Dimension of task embedding
        vocab_size: Vocabulary size for text prompts (if using text)
        max_length: Maximum length of task prompt
    """
    
    def __init__(
        self,
        task_dim: int = 768,
        vocab_size: Optional[int] = None,
        max_length: Optional[int] = None
    ):
        super().__init__()
        self.task_dim = task_dim
        
        # If vocab_size is provided, create embedding layer for text prompts
        if vocab_size is not None:
            self.text_embedding = nn.Embedding(vocab_size, task_dim)
            self.max_length = max_length
        else:
            self.text_embedding = None
    
    def forward(
        self,
        task_input: Union[torch.Tensor, str, list],
        return_tensor: bool = True
    ) -> torch.Tensor:
        """
        Process task input (prompt or embedding).
        
        Args:
            task_input: Can be:
                - torch.Tensor: Direct task embedding
                - str: Task prompt text (requires tokenization)
                - list: List of task prompt texts
            return_tensor: Whether to return tensor (default: True)
            
        Returns:
            Task embedding tensor of shape (batch_size, task_dim)
        """
        if isinstance(task_input, torch.Tensor):
            # Already a tensor, just return it
            if task_input.dim() == 1:
                task_input = task_input.unsqueeze(0)
            return task_input
        
        # For text prompts, would need tokenization
        # This is a placeholder - in practice, you'd use a tokenizer
        if isinstance(task_input, str) or isinstance(task_input, list):
            raise NotImplementedError(
                "Text prompt processing requires tokenization. "
                "Please provide pre-processed task embeddings as tensors."
            )
        
        return task_input
    
    def create_embedding(self, batch_size: int = 1) -> torch.Tensor:
        """
        Create a random task embedding (useful for testing).
        
        Args:
            batch_size: Batch size
            
        Returns:
            Random task embedding tensor
        """
        return torch.randn(batch_size, self.task_dim)
