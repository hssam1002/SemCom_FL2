"""
Data loading modules for semantic communication training.

This module provides dataset classes for loading training data:
- COCOCaptionDataset: COCO caption dataset for captioning task
- COCOMultiTaskDataset: COCO multi-task dataset (caption, detection, segmentation, keypoints)
"""

from .coco_dataset import COCOCaptionDataset, download_coco_info
from .coco_multitask_dataset import COCOMultiTaskDataset, download_coco_multitask_info

__all__ = [
    'COCOCaptionDataset', 
    'download_coco_info',
    'COCOMultiTaskDataset',
    'download_coco_multitask_info'
]
