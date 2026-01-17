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
from typing import Dict, Optional

from models.florence2_model import Florence2Model, get_vision_encoder_output_dim
from transmitter.transmitter import Transmitter
from channel.channel import Channel, create_channel
from receiver.receiver import Receiver
from data.coco_dataset import COCOCaptionDataset, download_coco_info
from data.coco_multitask_dataset import COCOMultiTaskDataset, download_coco_multitask_info


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
    Train one epoch on the semantic communication pipeline.
    
    Note: Florence-2 pre-trained model is always frozen.
    Only Transmitter and Receiver are trained.
    
    Args:
        transmitter: Transmitter model
        receiver: Receiver model
        channel: Channel model (for adding noise during training)
        dataloader: DataLoader for COCO dataset
        optimizer: Optimizer
        device: Device to run on
        epoch: Current epoch number
        mode: Processing mode ('vision_tower' or 'image_proj_norm')
        use_channel: Whether to use channel (add noise) during training
        accumulation_steps: Gradient accumulation steps
    
    Returns:
        Dictionary with training metrics (loss, etc.)
    """
    transmitter.train()
    receiver.train()
    
    # Florence-2 model should always be in eval mode (frozen)
    florence2_model = transmitter.florence2_model
    florence2_model.model.eval()
    
    total_loss = 0.0
    num_batches = 0
    
    # Get processor and model for text embedding generation
    processor = florence2_model.processor
    model = florence2_model.model
    embedding_layer = model.get_input_embeddings()
    
    # Create dummy image for text embedding generation (shared between Tx/Rx)
    dummy_image = Image.fromarray(np.zeros((768, 768, 3), dtype=np.uint8))
    
    # Progress bar
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(pbar):
        images = batch['image']  # List of PIL Images
        
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
        # For caption task: process text directly
        # For OD task: convert bboxes+labels to Florence-2 format
        # Note: Currently only caption task is fully implemented for training
        #       Other tasks (OD, etc.) will need different loss calculation
        
        # Get ground truth text for each sample
        gt_texts = []
        for i, (task, gt) in enumerate(zip(tasks, ground_truths)):
            if task == 'caption':
                gt_texts.append(gt)
            elif task == 'od':
                # For OD task, we'll need to format it properly
                # For now, skip non-caption tasks in training
                # TODO: Implement OD loss calculation
                continue
            else:
                # Other tasks: skip for now
                # TODO: Implement task-specific loss calculation
                continue
        
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
            caption_input_ids = gt_inputs["input_ids"].to(device=device, dtype=torch.long)
        
        # Update batch size for actual processed samples
        actual_batch_size = len(gt_texts)
        if actual_batch_size < batch_size:
            # Adjust embeddings for reduced batch size
            text_embeddings = text_embeddings[:actual_batch_size]
            merged_embeds = merged_embeds[:actual_batch_size]
            attention_mask = attention_mask[:actual_batch_size] if attention_mask is not None else None
        
        # Get caption embeddings (for teacher forcing)
        # Florence-2 embedding layer is frozen, so use torch.no_grad()
        with torch.no_grad():
            caption_embeddings = embedding_layer(caption_input_ids)
        
        # Extract vision features from merged_embeds (before task prompt)
        vision_seq_len = merged_embeds.size(1) - text_embeddings.size(1)
        vision_features = merged_embeds[:, :vision_seq_len]
        
        # Merge vision features with caption embeddings
        # Full sequence: [vision_features, caption_embeddings]
        # Note: We exclude task_prompt for training (use caption directly)
        # Florence-2's _merge_input_ids_with_image_features is frozen (requires_grad=False)
        # no_grad() is optional but helps with memory efficiency
        merged_embeds_gt, attention_mask_gt = model._merge_input_ids_with_image_features(
            vision_features,
            caption_embeddings
        )
        
        # Create labels for language model
        # Labels: -100 for vision part, caption tokens for caption part
        # For next token prediction, we shift labels by 1
        vision_seq_len = vision_features.size(1)
        caption_seq_len = caption_input_ids.size(1)
        
        labels = torch.full(
            (actual_batch_size, vision_seq_len + caption_seq_len),
            -100,
            device=device,
            dtype=torch.long
        )
        # Set caption labels (vision part stays -100, caption part gets actual tokens)
        labels[:, vision_seq_len:] = caption_input_ids
        
        # Forward through language model
        # Language model is frozen (requires_grad=False)
        # Use no_grad() for memory efficiency (prevents computation graph creation for frozen LM)
        # Note: We still need logits for loss computation, but frozen modules won't compute gradients
        #       The loss.backward() will only compute gradients for trainable modules upstream
        language_model = model.language_model
        
        with torch.no_grad():
            outputs = language_model(
                inputs_embeds=merged_embeds_gt,
                attention_mask=attention_mask_gt
            )
        
        # Get logits and compute loss
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
        
        # Shift logits and labels for next token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Compute loss
        # Note: All Florence-2 modules are frozen (requires_grad=False), so they won't compute gradients.
        # Currently, no trainable modules exist, so this loss is for monitoring only.
        # Future: When compression/decompression modules are added, they will receive gradients
        #         and can be trained end-to-end. The gradient flow will work automatically since
        #         we removed no_grad() and detach() from the forward passes.
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
        effective_snr_db=args.snr_db
    )
    print(f"Channel type: {args.channel_type}, SNR: {args.snr_db} dB")
    
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
    
    # Load COCO dataset
    print("\n=== Loading COCO Dataset ===")
    # Use tasks argument (default to ['caption'] if not specified)
    tasks = args.tasks if args.tasks is not None and len(args.tasks) > 0 else ['caption']
    multitask_mode = len(tasks) > 1
    
    try:
        if multitask_mode or len(tasks) == 1:
            # Use COCOMultiTaskDataset for both single and multi-task
            # Each sample will have its own task_prompt based on task type
            dataset = COCOMultiTaskDataset(
                data_root=args.data_root,
                tasks=tasks,
                transform=None  # Processor handles preprocessing
            )
            if multitask_mode:
                print(f"Multi-task mode: tasks={tasks}")
            else:
                print(f"Single task mode: task={tasks[0]}")
            print("  Note: Each sample will use task-specific prompt automatically")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nPlease download COCO dataset first:")
        if multitask_mode:
            download_coco_multitask_info()
        else:
            download_coco_info()
        return
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Number of batches: {len(dataloader)}")
    
    # Setup optimizer
    # Only train transmitter and receiver (Florence-2 is frozen)
    # Tx/Rx의 새로 추가된 module만 trainable하게 설정되며, Florence-2에서 가져온 module들은 frozen된 상태를 유지지
    print("\n=== Setting up Optimizer ===")
    trainable_params = []
    
    # Collect trainable parameters from Transmitter
    tx_params = [p for p in transmitter.parameters() if p.requires_grad]
    trainable_params += tx_params
    tx_num_params = sum(p.numel() for p in tx_params)
    print(f"  Transmitter trainable parameters: {tx_num_params:,}")
    
    # Collect trainable parameters from Receiver
    rx_params = [p for p in receiver.parameters() if p.requires_grad]
    trainable_params += rx_params
    rx_num_params = sum(p.numel() for p in rx_params)
    print(f"  Receiver trainable parameters: {rx_num_params:,}")
    
    # Verify Florence-2 is frozen
    florence2_params = [p for p in florence2_model.model.parameters() if p.requires_grad]
    florence2_num_params = sum(p.numel() for p in florence2_model.model.parameters())
    if len(florence2_params) > 0:
        print(f"  ⚠ Warning: {len(florence2_params)} Florence-2 parameters are trainable (should be 0)")
    else:
        print(f"  ✓ Florence-2 model is frozen ({florence2_num_params:,} parameters)")
    
    total_trainable = sum(p.numel() for p in trainable_params)
    print(f"  Total trainable parameters: {total_trainable:,}")
    
    if total_trainable == 0:
        print("\n  ℹ Info: No trainable parameters found in Transmitter/Receiver.")
        print("  This is expected: Current Tx/Rx only use frozen Florence-2 modules.")
        print("  Future: Compression/decompression modules will be added and will be trainable.")
        print("  Training will run for loss monitoring (no parameter updates until compression modules are added).")
    
    optimizer = AdamW(
        trainable_params,
        lr=args.learning_rate,
        weight_decay = args.weight_decay
    ) if total_trainable > 0 else None
    
    # Setup scheduler (only if optimizer exists)
    scheduler = None
    if optimizer is not None:
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max = args.num_epochs,
            eta_min = args.learning_rate * 0.01
        )
    
    # Training loop
    print("\n=== Starting Training ===")
    for epoch in range(1, args.num_epochs + 1):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch}/{args.num_epochs}")
        print(f"{'='*70}")
        
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
        
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Loss: {metrics['loss']:.4f}")
        if optimizer is not None:
            print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        else:
            print(f"  Learning Rate: N/A (no trainable parameters)")
        
        # Save checkpoint
        if epoch % args.save_interval == 0:
            checkpoint_path = Path(args.output_dir) / f"checkpoint_epoch_{epoch}.pt"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            
            torch.save({
                'epoch': epoch,
                'transmitter_state_dict': transmitter.state_dict(),
                'receiver_state_dict': receiver.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': metrics['loss'],
                'mode': args.mode,
            }, checkpoint_path)
            print(f"  Checkpoint saved: {checkpoint_path}")
    
    print("\n=== Training Complete ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train semantic communication system with COCO dataset"
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
    
    # Data arguments
    parser.add_argument(
        '--data_root',
        type=str,
        default='/data4/hongsik/data/COCO',
        help='Root directory of COCO dataset'
    )
    parser.add_argument(
        '--tasks',
        type=str,
        nargs='+',
        default=['caption'],
        choices=['caption', 'od', 'detection', 'segmentation', 'keypoints'],
        help='List of tasks for training (e.g., --tasks caption for single task, --tasks caption od for multi-task). Each sample will use task-specific prompt automatically based on its task type.'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=4,
        help='Batch size'
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
        default = 200,
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
        default = 1e-4,
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
        choices=['noiseless', 'awgn', 'rayleigh'],
        help='Channel type for training'
    )
    parser.add_argument(
        '--snr_db',
        type=float,
        default=20.0,
        help='SNR in dB for channel'
    )
    parser.add_argument(
        '--use_channel',
        action='store_true',
        help='Use channel (add noise) during training'
    )
    
    # Output arguments
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./checkpoints',
        help='Directory to save checkpoints'
    )
    parser.add_argument(
        '--save_interval',
        type=int,
        default=1,
        help='Save checkpoint every N epochs'
    )
    
    # Device arguments
    parser.add_argument(
        '--cpu',
        action='store_true',
        help='Force CPU usage'
    )
    
    args = parser.parse_args()
    
    main(args)
