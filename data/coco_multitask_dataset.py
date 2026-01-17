"""
COCO Multi-Task Dataset for Florence-2 training.
Supports multiple tasks: Caption, Object Detection, Segmentation, etc.

All ground truth data is pre-formatted as text strings for efficient training.
"""

import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
from typing import List, Dict, Optional, Tuple, Union
from pathlib import Path
import numpy as np


class COCOMultiTaskDataset(Dataset):
    """
    COCO Multi-Task Dataset for Florence-2 semantic communication training.
    
    Supports multiple tasks:
    - Caption: Image captioning
    - Object Detection (OD): Bounding boxes and labels
    - Instance Segmentation: Segmentation masks
    - Keypoints: Person pose estimation (if available)
    
    Args:
        data_root: Root directory containing COCO dataset
            Expected structure:
            data_root/
                annotations/
                    captions_train2017.json
                    instances_train2017.json
                    person_keypoints_train2017.json (optional)
                train2017/
                    *.jpg
        tasks: List of tasks to include (e.g., ['caption', 'od', 'segmentation'])
        transform: Optional image transform
    """
    
    def __init__(
        self,
        data_root: str,
        tasks: List[str] = ['caption'],
        transform: Optional[callable] = None
    ):
        self.data_root = Path(data_root)
        self.tasks = [task.lower() for task in tasks]
        self.transform = transform
        
        # Validate tasks
        valid_tasks = ['caption', 'od', 'detection', 'segmentation', 'keypoints']
        for task in self.tasks:
            if task not in valid_tasks:
                raise ValueError(f"Invalid task: {task}. Valid tasks: {valid_tasks}")
        
        # Normalize task names
        if 'detection' in self.tasks:
            self.tasks = [t if t != 'detection' else 'od' for t in self.tasks]
        
        # Load annotations based on tasks
        self.annotations = {}
        self.image_id_to_filename = {}
        self.samples = []
        
        # Load captions if needed
        if 'caption' in self.tasks:
            self._load_captions()
        
        # Load object detection/segmentation if needed
        if 'od' in self.tasks or 'segmentation' in self.tasks:
            self._load_instances()
        
        # Load keypoints if needed
        if 'keypoints' in self.tasks:
            self._load_keypoints()
        
        self.images_dir = self.data_root / "train2017"
        
        if not self.images_dir.exists():
            raise FileNotFoundError(
                f"COCO images directory not found at {self.images_dir}. "
                f"Please download COCO images to {self.images_dir}"
            )
        
        print(f"Loaded COCO multi-task dataset:")
        print(f"  Tasks: {self.tasks}")
        print(f"  Total samples: {len(self.samples)}")
        for task, count in self.annotations.items():
            print(f"  {task}: {count} annotations")
    
    def _load_captions(self):
        """Load caption annotations."""
        annotations_path = self.data_root / "annotations" / "captions_train2017.json"
        if not annotations_path.exists():
            raise FileNotFoundError(
                f"COCO caption annotations not found at {annotations_path}"
            )
        
        with open(annotations_path, 'r') as f:
            coco_data = json.load(f)
        
        # Build image id to filename mapping
        for img in coco_data['images']:
            if img['id'] not in self.image_id_to_filename:
                self.image_id_to_filename[img['id']] = img['file_name']
        
        # Store caption annotations
        caption_samples = [
            (ann['image_id'], ann['caption'])
            for ann in coco_data['annotations']
        ]
        
        self.annotations['caption'] = len(caption_samples)
        self.samples.extend([('caption', img_id, caption) for img_id, caption in caption_samples])
    
    def _load_instances(self):
        """Load object detection and segmentation annotations."""
        annotations_path = self.data_root / "annotations" / "instances_train2017.json"
        if not annotations_path.exists():
            raise FileNotFoundError(
                f"COCO instances annotations not found at {annotations_path}"
            )
        
        with open(annotations_path, 'r') as f:
            coco_data = json.load(f)
        
        # Build image id to filename mapping
        for img in coco_data['images']:
            if img['id'] not in self.image_id_to_filename:
                self.image_id_to_filename[img['id']] = img['file_name']
        
        # Build category id to name mapping
        self.category_id_to_name = {
            cat['id']: cat['name']
            for cat in coco_data['categories']
        }
        
        # Group annotations by image_id
        image_annotations = {}
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in image_annotations:
                image_annotations[img_id] = []
            
            # Extract bbox and category
            bbox = ann['bbox']  # [x, y, width, height]
            category_id = ann['category_id']
            category_name = self.category_id_to_name[category_id]
            
            annotation_data = {
                'bbox': bbox,
                'category_id': category_id,
                'category_name': category_name,
                'area': ann.get('area', 0),
                'iscrowd': ann.get('iscrowd', 0)
            }
            
            # Add segmentation if available and task requires it
            if 'segmentation' in self.tasks and 'segmentation' in ann:
                annotation_data['segmentation'] = ann['segmentation']
            
            image_annotations[img_id].append(annotation_data)
        
        # Create samples with pre-formatted text ground truth
        # Pre-format ground truth as text for efficient training (no conversion during training)
        od_samples = []
        for img_id, anns in image_annotations.items():
            # Format OD ground truth to Florence-2 text format: "[x1 y1 x2 y2] label"
            formatted_detections = []
            for ann in anns:
                bbox = ann['bbox']  # [x, y, width, height] (COCO format)
                category_name = ann['category_name']
                
                # Convert [x, y, w, h] to [x1, y1, x2, y2] format (corners)
                x, y, w, h = bbox
                x1, y1 = x, y
                x2, y2 = x + w, y + h
                
                # Florence-2 format: "[<x1> <y1> <x2> <y2>] <label>"
                formatted_detections.append(f"[{x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f}] {category_name}")
            
            # Join all detections with newlines
            gt_text = "\n".join(formatted_detections) if formatted_detections else ""
            od_samples.append((img_id, gt_text))
        
        self.annotations['od'] = len(od_samples)
        self.samples.extend([('od', img_id, gt_text) for img_id, gt_text in od_samples])
    
    def _load_keypoints(self):
        """Load person keypoints annotations."""
        annotations_path = self.data_root / "annotations" / "person_keypoints_train2017.json"
        if not annotations_path.exists():
            print(f"Warning: Keypoints annotations not found at {annotations_path}. Skipping keypoints.")
            return
        
        with open(annotations_path, 'r') as f:
            coco_data = json.load(f)
        
        # Build image id to filename mapping
        for img in coco_data['images']:
            if img['id'] not in self.image_id_to_filename:
                self.image_id_to_filename[img['id']] = img['file_name']
        
        # Group keypoints by image_id
        image_keypoints = {}
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in image_keypoints:
                image_keypoints[img_id] = []
            
            if 'keypoints' in ann and ann.get('num_keypoints', 0) > 0:
                image_keypoints[img_id].append({
                    'keypoints': ann['keypoints'],
                    'num_keypoints': ann.get('num_keypoints', 0),
                    'bbox': ann.get('bbox', [])
                })
        
        # Create samples with pre-formatted text ground truth
        # Pre-format ground truth as text for efficient training (no conversion during training)
        keypoint_samples = []
        for img_id, keypoints_list in image_keypoints.items():
            if len(keypoints_list) == 0:
                continue
            
            # Format keypoints ground truth to Florence-2 text format: "[x1 y1 x2 y2] person"
            formatted_keypoints = []
            for kp_data in keypoints_list:
                if 'bbox' in kp_data and len(kp_data['bbox']) == 4:
                    bbox = kp_data['bbox']  # [x, y, width, height] (COCO format)
                    
                    # Convert [x, y, w, h] to [x1, y1, x2, y2] format (corners)
                    x, y, w, h = bbox
                    x1, y1 = x, y
                    x2, y2 = x + w, y + h
                    
                    # Florence-2 uses "<OD>" task for keypoints, format similar to OD
                    formatted_keypoints.append(f"[{x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f}] person")
            
            # Join all keypoints with newlines
            gt_text = "\n".join(formatted_keypoints) if formatted_keypoints else ""
            keypoint_samples.append((img_id, gt_text))
        
        self.annotations['keypoints'] = len(keypoint_samples)
        self.samples.extend([('keypoints', img_id, gt_text) for img_id, gt_text in keypoint_samples])
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, any]:
        """
        Get a sample from the dataset.
        
        Note: All ground truth data is pre-formatted as text strings during dataset loading
        for efficient training (no conversion needed during training).
        
        Returns:
            Dictionary containing:
            - image: PIL Image
            - task: str (task type: 'caption', 'od', 'keypoints', etc.)
            - task_prompt: str (Florence-2 task prompt)
            - ground_truth: str (pre-formatted text ground truth for the task)
            - image_id: int (COCO image ID)
        """
        task_type, image_id, ground_truth = self.samples[idx]
        # ground_truth is now always a string (pre-formatted during dataset loading)
        
        # Load image
        filename = self.image_id_to_filename[image_id]
        image_path = self.images_dir / filename
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = Image.open(image_path).convert('RGB')
        
        # Apply transform if provided
        if self.transform is not None:
            image = self.transform(image)
        
        # Map task type to Florence-2 task prompt
        # Florence-2 supports many tasks, but COCO dataset only provides annotations for some
        task_prompt_map = {
            # COCO dataset에서 직접 제공하는 annotations
            'caption': '<CAPTION>',
            'detailed_caption': '<DETAILED_CAPTION>',
            'od': '<OD>',
            'detection': '<OD>',
            'segmentation': '<OD>',  # Florence-2 uses OD for segmentation
            'keypoints': '<OD>',  # Person keypoints
            
            # COCO dataset에서 파생 가능한 tasks
            'dense_region_caption': '<DENSE_REGION_CAPTION>',  # OD 결과 기반
            'region_proposal': '<REGION_PROPOSAL>',  # OD 결과 기반
            
            # COCO dataset으로는 학습 불가능 (추가 annotation 필요)
            # 'grounding': '<GROUNDING>',  # Text input 필요
            # 'ocr': '<OCR>',  # COCO에 text annotation 없음
            # 'ocr_with_region': '<OCR_WITH_REGION>',  # COCO에 text annotation 없음
        }
        task_prompt = task_prompt_map.get(task_type, '<CAPTION>')
        
        return {
            'image': image,
            'task': task_type,
            'task_prompt': task_prompt,
            'ground_truth': ground_truth,
            'image_id': image_id,
            'filename': filename
        }


def download_coco_multitask_info():
    """
    Print information about downloading COCO dataset for multi-task training.
    """
    print("=" * 70)
    print("COCO Multi-Task Dataset Download Information")
    print("=" * 70)
    print("\n1. Download COCO 2017 Training Images:")
    print("   http://images.cocodataset.org/zips/train2017.zip")
    print("   Extract to: /data4/hongsik/data/COCO/train2017/")
    
    print("\n2. Download COCO 2017 Annotations:")
    print("   http://images.cocodataset.org/annotations/annotations_trainval2017.zip")
    print("   Extract and place the following files in:")
    print("   /data4/hongsik/data/COCO/annotations/")
    print("   - captions_train2017.json (for captioning)")
    print("   - instances_train2017.json (for detection/segmentation)")
    print("   - person_keypoints_train2017.json (for keypoints, optional)")
    
    print("\n3. Expected directory structure:")
    print("   /data4/hongsik/data/COCO/")
    print("   ├── annotations/")
    print("   │   ├── captions_train2017.json")
    print("   │   ├── instances_train2017.json")
    print("   │   └── person_keypoints_train2017.json (optional)")
    print("   └── train2017/")
    print("       └── *.jpg (118,287 images)")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    download_coco_multitask_info()
