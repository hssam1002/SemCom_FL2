"""
Shared modules for semantic communication system.
Contains task embedding and CSI (Channel State Information) handling.
"""

from .task_embedding import TaskEmbedding
from .csi import CSI

__all__ = ['TaskEmbedding', 'CSI']
