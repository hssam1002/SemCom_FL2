"""
Training script for semantic communication using Florence-2 with COCO dataset.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
from PIL import Image
from typing import Dict, Optional, List, Union

from models.florence2_model import Florence2Model, get_vision_encoder_output_dim
from transmitter.transmitter import Transmitter
from channel.channel import Channel, create_channel
from receiver.receiver import Receiver
from data.coco_dataset import COCOCaptionDataset, download_coco_info
from data.coco_multitask_dataset import COCOMultiTaskDataset, download_coco_multitask_info


# Note: Ground truth formatting functions have been moved to coco_multitask_dataset.py
# All ground truth is now pre-formatted as text during dataset loading for efficiency.
# No conversion is needed during training - ground_truth is already a string.


def train_one_epoch(
    transmitter: Transmitter,
    receiver: Receiver,
    channel: Channel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    mode: str = 'vision_tower',
    use_channel: bool = True,
    accumulation_steps: int = 1,
    multitask: bool = False  # If True, each sample has its own task_prompt
) -> Dict[str, float]:
    """
    Train one epoch of the semantic communication pipeline.
    
    Note: Florence-2 pre-trained model is always frozen.
    Only future compression/decompression modules will be trainable.
    
    Args:
        transmitter: Transmitter model
        receiver: Receiver model
        channel: Channel model (for adding noise during training)
        dataloader: DataLoader for training data
        optimizer: Optimizer for trainable parameters
        device: Device to run training on
        epoch: Current epoch number
        mode: Processing mode ('vision_tower' or 'image_proj_norm')
        use_channel: Whether to use channel noise during training
        accumulation_steps: Number of gradient accumulation steps
        multitask: Whether to use multi-task training
    
    Returns:
        Dictionary with training metrics (loss, etc.)
    """
    # Florence-2 model should always be in eval mode (frozen)
    florence2_model = transmitter.florence2_model
    florence2_model.model.eval()
    
    transmitter.train()  # Transmitter may have trainable modules in future
    receiver.train()  # Receiver may have trainable modules in future
    
    # Get processor and model for text embedding generation
    processor = florence2_model.processor
    model = florence2_model.model
    embedding_layer = model.get_input_embeddings()
    
    # Create dummy image for text embedding generation (shared between Tx/Rx)
    dummy_image = Image.fromarray(np.zeros((768, 768, 3), dtype=np.uint8))
    
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(pbar):
        # Get batch data
        images = batch['image']  # List of PIL Images or torch.Tensor
        batch_size = len(images)
        
        # Get task prompts from batch (each sample has its own task_prompt)
        # COCOMultiTaskDataset provides 'task_prompt' for each sample
        if 'task_prompt' in batch:
            task_prompts = batch['task_prompt']  # List of task prompts (one per sample)
        else:
            # Fallback: should not happen with COCOMultiTaskDataset
            # But keep for compatibility with old COCOCaptionDataset
            task_prompts = ['<CAPTION>'] * batch_size
        
        # Get ground truth data based on task type
        if multitask and 'task' in batch:
            tasks = batch['task']
            ground_truths = batch['ground_truth']  # List of ground truth (varies by task)
        else:
            # Single task mode (caption)
            tasks = ['caption'] * batch_size
            ground_truths = batch.get('caption', [''] * batch_size)  # Fallback for compatibility
        
        # Generate text embeddings for task prompts (shared between Tx/Rx)
        # Florence-2 processor and embedding layer are frozen (requires_grad=False)
        # Use no_grad() for memory efficiency (prevents computation graph creation)
        with torch.no_grad():
            inputs = processor(
                text=task_prompts,
                images=[dummy_image] * batch_size,
                return_tensors="pt"
            )
            input_ids_task = inputs["input_ids"].to(device=device, dtype=torch.long)
            text_embeddings = embedding_layer(input_ids_task)
        
        # Transmitter: Process images
        # Note: Transmitter uses frozen Florence-2 vision_tower internally
        # All Florence-2 modules (vision_tower, etc.) are frozen (requires_grad=False)
        # Since Florence-2 modules are frozen, no_grad() is optional but helps with memory efficiency
        # Future: Compression module (trainable) will be added here:
        #   vision_tower (frozen) → compression_module (trainable) → output
        # When compression module is added, remove no_grad() and detach() to allow gradient flow
        tx_output = transmitter(images)  # (batch, seq_len, dim)
        # Note: No detach() needed - frozen modules (requires_grad=False) won't compute gradients anyway
        # When compression module is added, this will allow gradients to flow to it
        
        # Channel: Add noise (if enabled)
        if use_channel:
            received_signal = channel(tx_output)
        else:
            received_signal = tx_output
        
        # Receiver: Process received signal and merge with text embeddings
        # Note: Receiver uses frozen Florence-2 modules (image_pos_embed, image_proj_norm, etc.)
        # All Florence-2 modules are frozen (requires_grad=False)
        # Since Florence-2 modules are frozen, no_grad() is optional but helps with memory efficiency
        # Future: Decompression module (trainable) will be added here:
        #   received → decompression_module (trainable) → image_pos_embed (frozen) → ...
        # When decompression module is added, remove no_grad() to allow gradient flow
        merged_embeds, attention_mask = receiver(received_signal, text_embeddings)
        # Note: No detach() needed - frozen modules (requires_grad=False) won't compute gradients anyway
        # When decompression module is added, this will allow gradients to flow to it
        
        # Prepare ground truth for teacher forcing
        # All tasks use cross-entropy loss with text format ground truth
        # Note: Ground truth is already pre-formatted as text in COCOMultiTaskDataset
        #       No conversion needed during training (more efficient)
        gt_texts = []
        for gt in ground_truths:
            # Ground truth is already a string (pre-formatted during dataset loading)
            if isinstance(gt, str) and gt.strip():  # Only add non-empty ground truth
                gt_texts.append(gt)
        
        if len(gt_texts) == 0:
            # Skip batch if no valid samples
            continue
        
        # Process ground truth texts with processor to get input_ids
        # Note: We process with images to get correct tokenization
        # Florence-2 is frozen, so use torch.no_grad() for processing
        with torch.no_grad():
            gt_inputs = processor(
                text=gt_texts,
                images=images[:len(gt_texts)],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            gt_input_ids = gt_inputs["input_ids"].to(device=device, dtype=torch.long)
        
        # Update batch size for actual processed samples
        actual_batch_size = len(gt_texts)
        if actual_batch_size < batch_size:
            # Adjust embeddings for reduced batch size
            text_embeddings = text_embeddings[:actual_batch_size]
            merged_embeds = merged_embeds[:actual_batch_size]
            attention_mask = attention_mask[:actual_batch_size] if attention_mask is not None else None
        
        # Get ground truth embeddings (for teacher forcing)
        # Florence-2 embedding layer is frozen (requires_grad=False) - use no_grad() for memory efficiency
        with torch.no_grad():
            gt_embeddings = embedding_layer(gt_input_ids)
        
        # Extract vision features from merged_embeds (before task prompt)
        # Note: merged_embeds comes from receiver, which may have trainable modules in future
        #       So we keep it outside no_grad() to allow gradient flow
        vision_seq_len = merged_embeds.size(1) - text_embeddings.size(1)
        vision_features = merged_embeds[:, :vision_seq_len]
        
        # Merge vision features with ground truth embeddings
        # Full sequence: [vision_features, gt_embeddings]
        # Note: We exclude task_prompt for training (use ground truth directly)
        # Florence-2's _merge_input_ids_with_image_features is frozen (requires_grad=False)
        # Use no_grad() for memory efficiency
        # However, vision_features may come from trainable modules, so we need to be careful
        with torch.no_grad():
            merged_embeds_gt, attention_mask_gt = model._merge_input_ids_with_image_features(
                vision_features,
                gt_embeddings
            )
        
        # Create labels for language model
        # Labels: -100 for vision part, ground truth tokens for text part
        # For next token prediction, we shift labels by 1
        vision_seq_len = vision_features.size(1)
        gt_seq_len = gt_input_ids.size(1)
        
        labels = torch.full(
            (actual_batch_size, vision_seq_len + gt_seq_len),
            -100,
            device=device,
            dtype=torch.long
        )
        # Set ground truth labels (vision part stays -100, text part gets actual tokens)
        labels[:, vision_seq_len:] = gt_input_ids
        
        # Forward through language model
        # Language model is frozen (requires_grad=False)
        # IMPORTANT: We need logits for loss computation, and logits need to allow gradient flow
        #            to trainable modules upstream (compression/decompression).
        #            Even though language_model is frozen, we should NOT use no_grad() here
        #            because we need the computation graph for loss.backward() to flow gradients
        #            to trainable modules in transmitter/receiver.
        #            The frozen language_model won't compute gradients anyway (requires_grad=False),
        #            but the computation graph is needed for gradient flow to upstream modules.
        language_model = model.language_model
        
        # Remove no_grad() to allow gradient flow to trainable modules upstream
        # Frozen modules (requires_grad=False) won't compute gradients, but graph is needed
        outputs = language_model(
            inputs_embeds=merged_embeds_gt,
            attention_mask=attention_mask_gt
        )
        
        # Get logits and compute loss
        # Loss: Cross-entropy between language_model output (logits) and ground truth token IDs
        # This is the standard language model training loss (next token prediction)
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
        
        # Shift logits and labels for next token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Compute loss: Cross-entropy between language_model output (logits) and ground truth token IDs
        # This is the standard language model training loss (next token prediction).
        # Loss = CrossEntropy(logits, ground_truth_token_ids)
        # 
        # Flow: Receiver output (merged_embeds) -> Language model -> Logits -> Loss with GT token IDs
        # 
        # Note: All Florence-2 modules are frozen (requires_grad=False), so they won't compute gradients.
        # Currently, no trainable modules exist, so this loss is for monitoring only.
        # Future: When compression/decompression modules are added, they will receive gradients
        #         and can be trained end-to-end. The gradient flow will work automatically since
        #         we removed no_grad() from language_model forward.
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        # Normalize loss by accumulation steps
        loss = loss / accumulation_steps
        
        # Backward pass (only if there are trainable parameters)
        if optimizer is not None:
            loss.backward()
            
            # Update weights
            if (batch_idx + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
        else:
            # No trainable parameters, skip backward/optimizer step
            # Loss is computed for monitoring only
            pass
        
        # Update metrics
        total_loss += loss.item() * accumulation_steps
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item() * accumulation_steps:.4f}',
            'avg_loss': f'{total_loss / num_batches:.4f}'
        })
    
    # Final gradient step if needed
    if optimizer is not None and num_batches % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    
    return {
        'loss': avg_loss,
        'num_batches': num_batches
    }


def main(args):
    """
    Main training function.
    """
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize Florence-2 model
    print("\n=== Initializing Florence-2 Model ===")
    florence2_model = Florence2Model(
        model_name=args.model_name,
        device=device
    )
    
    # Print vision encoder output dimension based on mode
    vision_dim = get_vision_encoder_output_dim(
        model_size=args.model_name.split('/')[-1].split('-')[-1],  # Extract 'base' or 'large'
        mode=args.mode
    )
    print(f"Vision encoder output dimension (mode={args.mode}): {vision_dim}")
    
    # Freeze Florence-2 pre-trained model (always frozen)
    # Note: All Florence-2 modules (vision_tower, image_pos_embed, image_proj_norm, language_model, etc.)
    #       are frozen. Only future compression/decompression modules will be trainable.
    print("\n=== Freezing Florence-2 Pre-trained Model ===")
    for param in florence2_model.model.parameters():
        param.requires_grad = False
    florence2_model.model.eval()  # Set to eval mode
    print("✓ Florence-2 model frozen (pre-trained weights will not be updated)")
    print("  - vision_tower: frozen")
    print("  - image_pos_embed: frozen")
    print("  - visual_temporal_embed: frozen")
    print("  - image_projection: frozen")
    print("  - image_proj_norm: frozen")
    print("  - language_model: frozen")
    
    # Initialize Transmitter
    # Note: Transmitter uses frozen Florence-2 modules (vision_tower, etc.)
    #       Future compression module will be added here and will be trainable
    print("\n=== Initializing Transmitter ===")
    transmitter = Transmitter(
        florence2_model=florence2_model,
        mode=args.mode,
        task_embedding_dim=768
    ).to(device)
    print(f"Transmitter mode: {args.mode}")
    print("  Note: All Florence-2 modules used by Transmitter are frozen")
    print("  Future: Compression module (trainable) will be added after vision_tower output")
    
    # Initialize Channel
    print("\n=== Initializing Channel ===")
    channel = create_channel(
        channel_type=args.channel_type,
        snr_db=args.snr_db,
        device=device
    )
    print(f"Channel type: {args.channel_type}")
    print(f"SNR: {args.snr_db} dB")
    
    # Initialize Receiver
    # Note: Receiver uses frozen Florence-2 modules (image_pos_embed, image_proj_norm, etc.)
    #       Future decompression module will be added here and will be trainable
    print("\n=== Initializing Receiver ===")
    receiver = Receiver(
        florence2_model=florence2_model,
        mode=args.mode
    ).to(device)
    print(f"Receiver mode: {args.mode}")
    print("  Note: All Florence-2 modules used by Receiver are frozen")
    print("  Future: Decompression module (trainable) will be added before image_pos_embed")
    
    # Load dataset
    print("\n=== Loading Dataset ===")
    data_root = args.data_root if args.data_root else "/data4/hongsik/data/COCO"
    
    # Use tasks argument (default to ['caption'] if not specified)
    tasks = args.tasks if args.tasks is not None and len(args.tasks) > 0 else ['caption']
    multitask_mode = len(tasks) > 1
    
    # Load dataset (no try-except for better performance - let errors propagate)
    if multitask_mode or len(tasks) == 1:
        # Use COCOMultiTaskDataset for both single and multi-task
        # Each sample will have its own task_prompt based on task type
        dataset = COCOMultiTaskDataset(
            data_root=data_root,
            tasks=tasks,
            transform=None
        )
        if multitask_mode:
            print(f"Multi-task mode: tasks={tasks}")
        else:
            print(f"Single task mode: task={tasks[0]}")
        print("  Note: Each sample will use task-specific prompt automatically")
    else:
        # Fallback to single-task caption dataset
        dataset = COCOCaptionDataset(data_root=data_root, transform=None)
        print("Single task mode: caption (fallback)")
    
    if multitask_mode:
        download_coco_multitask_info()
    else:
        download_coco_info()
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=None  # Use default collate_fn
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Batch size: {args.batch_size}")
    print(f"Number of workers: {args.num_workers}")
    
    # Setup optimizer (currently no trainable parameters)
    # Tx/Rx의 새로 추가된 module만 trainable하게 설정되며, Florence-2에서 가져온 module들은 frozen된 상태를 유지지
    trainable_params = [p for p in list(transmitter.parameters()) + list(receiver.parameters()) if p.requires_grad]
    florence2_params = [p for p in florence2_model.model.parameters() if p.requires_grad]
    florence2_num_params = sum(p.numel() for p in florence2_model.model.parameters())
    
    print("\n=== Model Parameters ===")
    if len(trainable_params) == 0:
        print("  ✗ No trainable parameters found")
        print(f"  ✓ Florence-2 model is frozen ({florence2_num_params:,} parameters)")
        optimizer = None
    else:
        trainable_num_params = sum(p.numel() for p in trainable_params)
        print(f"  ✓ Trainable parameters: {trainable_num_params:,}")
        print(f"  ✓ Florence-2 model is frozen ({florence2_num_params:,} parameters)")
        optimizer = AdamW(
            trainable_params,
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
    
    if optimizer is None:
        print("  This is expected: Current Tx/Rx only use frozen Florence-2 modules.")
        print("  Future: Compression/decompression modules will be added and will be trainable.")
        print("  Training will run for loss monitoring (no parameter updates until compression modules are added).")
    
    # Setup learning rate scheduler
    if optimizer is not None:
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=args.num_epochs,
            eta_min=args.learning_rate * 0.01
        )
    else:
        scheduler = None
    
    # Training loop
    print("\n=== Starting Training ===")
    best_loss = float('inf')
    
    for epoch in range(1, args.num_epochs + 1):
        metrics = train_one_epoch(
            transmitter=transmitter,
            receiver=receiver,
            channel=channel,
            dataloader=dataloader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            mode=args.mode,
            use_channel=args.use_channel,
            accumulation_steps=args.accumulation_steps,
            multitask=multitask_mode
        )
        
        # Update learning rate
        if scheduler is not None:
            scheduler.step()
        
        # Print epoch summary
        print(f"\nEpoch {epoch}/{args.num_epochs}")
        print(f"  Loss: {metrics['loss']:.4f}")
        print(f"  Batches: {metrics['num_batches']}")
        if scheduler is not None:
            print(f"  Learning rate: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save checkpoint if best
        if metrics['loss'] < best_loss:
            best_loss = metrics['loss']
            if args.save_dir:
                save_dir = Path(args.save_dir)
                save_dir.mkdir(parents=True, exist_ok=True)
                checkpoint = {
                    'epoch': epoch,
                    'transmitter_state_dict': transmitter.state_dict(),
                    'receiver_state_dict': receiver.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
                    'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                    'loss': metrics['loss'],
                    'mode': args.mode,
                    'tasks': tasks
                }
                checkpoint_path = save_dir / f"checkpoint_epoch_{epoch}.pt"
                torch.save(checkpoint, checkpoint_path)
                print(f"  ✓ Saved checkpoint: {checkpoint_path}")
    
    print("\n=== Training Complete ===")
    print(f"Best loss: {best_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train semantic communication pipeline with Florence-2")
    
    # Data arguments
    parser.add_argument(
        '--data_root',
        type=str,
        default=None,
        help='Root directory of COCO dataset (default: /data4/hongsik/data/COCO)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=4,
        help='Batch size for training'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='Number of data loading workers'
    )
    
    # Training arguments
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=200,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=1e-3,
        help='Learning rate'
    )
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=1e-4,
        help='Weight decay'
    )
    parser.add_argument(
        '--accumulation_steps',
        type=int,
        default=1,
        help='Gradient accumulation steps'
    )
    
    # Channel arguments
    parser.add_argument(
        '--channel_type',
        type=str,
        default='awgn',
        choices=['noiseless', 'awgn'],
        help='Channel type'
    )
    parser.add_argument(
        '--snr_db',
        type=float,
        default=10.0,
        help='SNR in dB for AWGN channel'
    )
    parser.add_argument(
        '--use_channel',
        type=bool,
        default=True,
        help='Whether to use channel noise during training'
    )
    
    # Model arguments
    parser.add_argument(
        '--model_name',
        type=str,
        default='microsoft/Florence-2-base',
        help='Florence-2 model name'
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='vision_tower',
        choices=['vision_tower', 'image_proj_norm'],
        help='Processing mode'
    )
    
    # Task arguments
    parser.add_argument(
        '--tasks',
        nargs='+',
        default=['caption'],
        choices=['caption', 'od', 'detection', 'segmentation', 'keypoints'],
        help='List of tasks for training (e.g., --tasks caption for single task, --tasks caption od for multi-task). Each sample will use task-specific prompt automatically based on its task type.'
    )
    
    # Other arguments
    parser.add_argument(
        '--save_dir',
        type=str,
        default=None,
        help='Directory to save checkpoints'
    )
    parser.add_argument(
        '--cpu',
        action='store_true',
        help='Force CPU usage'
    )
    
    args = parser.parse_args()
    
    main(args)
