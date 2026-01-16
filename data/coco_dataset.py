"""
COCO Caption Dataset for Florence-2 training.
"""

import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
from typing import List, Dict, Optional, Tuple
import requests
from pathlib import Path


class COCOCaptionDataset(Dataset):
    """
    COCO Caption Dataset for Florence-2 semantic communication training.
    
    This dataset loads COCO images and captions for training the semantic communication
    pipeline (transmitter/receiver) with Florence-2.
    
    Args:
        data_root: Root directory containing COCO dataset
            Expected structure:
            data_root/
                annotations/
                    captions_train2017.json
                train2017/
                    *.jpg
        task_prompt: Task prompt for Florence-2 (default: "<CAPTION>")
        transform: Optional image transform
    """
    
    def __init__(
        self,
        data_root: str,
        task_prompt: str = "<CAPTION>",
        transform: Optional[callable] = None
    ):
        self.data_root = Path(data_root)
        self.task_prompt = task_prompt
        self.transform = transform
        
        # Load annotations
        annotations_path = self.data_root / "annotations" / "captions_train2017.json"
        if not annotations_path.exists():
            raise FileNotFoundError(
                f"COCO annotations not found at {annotations_path}. "
                f"Please download COCO dataset to {data_root}"
            )
        
        with open(annotations_path, 'r') as f:
            coco_data = json.load(f)
        
        # Build image id to filename mapping
        self.image_id_to_filename = {
            img['id']: img['file_name']
            for img in coco_data['images']
        }
        
        # Build list of (image_id, caption) pairs
        self.samples = [
            (ann['image_id'], ann['caption'])
            for ann in coco_data['annotations']
        ]
        
        self.images_dir = self.data_root / "train2017"
        
        if not self.images_dir.exists():
            raise FileNotFoundError(
                f"COCO images directory not found at {self.images_dir}. "
                f"Please download COCO images to {self.images_dir}"
            )
        
        print(f"Loaded {len(self.samples)} COCO caption samples from {data_root}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, any]:
        """
        Get a sample from the dataset.
        
        Returns:
            Dictionary containing:
            - image: PIL Image
            - caption: str (ground truth caption)
            - task_prompt: str (task prompt for Florence-2)
            - image_id: int (COCO image ID)
        """
        image_id, caption = self.samples[idx]
        
        # Load image
        filename = self.image_id_to_filename[image_id]
        image_path = self.images_dir / filename
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = Image.open(image_path).convert('RGB')
        
        # Apply transform if provided
        if self.transform is not None:
            image = self.transform(image)
        
        return {
            'image': image,
            'caption': caption,
            'task_prompt': self.task_prompt,
            'image_id': image_id,
            'filename': filename
        }


def download_coco_info():
    """
    Print information about downloading COCO dataset.
    """
    print("=" * 70)
    print("COCO Dataset Download Information")
    print("=" * 70)
    print("\n1. Download COCO 2017 Training Images:")
    print("   http://images.cocodataset.org/zips/train2017.zip")
    print("   Extract to: /data4/hongsik/data/COCO/train2017/")
    
    print("\n2. Download COCO 2017 Captions Annotations:")
    print("   http://images.cocodataset.org/annotations/annotations_trainval2017.zip")
    print("   Extract and place captions_train2017.json in:")
    print("   /data4/hongsik/data/COCO/annotations/captions_train2017.json")
    
    print("\n3. Expected directory structure:")
    print("   /data4/hongsik/data/COCO/")
    print("   ├── annotations/")
    print("   │   └── captions_train2017.json")
    print("   └── train2017/")
    print("       └── *.jpg (118,287 images)")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    download_coco_info()
