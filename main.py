"""
Main script for semantic communication system using Florence-2.
"""

import torch
import argparse
import requests
import math
from pathlib import Path
from PIL import Image

from models.florence2_model import Florence2Model, get_vision_encoder_output_dim
from transmitter.transmitter import Transmitter
from channel.channel import Channel, create_channel
from receiver.receiver import Receiver
from utils.image_utils import load_image


def main(args):
    """
    Main function for semantic communication simulation.
    
    Args:
        args: Command line arguments
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"Using device: {device}")
    
    # Get vision encoder output dimension based on mode
    model_size = args.model_name.split('/')[-1].split('-')[-1]  # Extract 'base' or 'large'
    vision_dim = get_vision_encoder_output_dim(model_size=model_size, mode=args.mode)
    print(f"Vision encoder output dimension (mode={args.mode}): {vision_dim}")
    
    # Initialize Florence-2 model
    print("\n=== Initializing Florence-2 Model ===")
    florence2_model = Florence2Model(
        model_name=args.model_name,
        device=device
    )
    
    # Note: CSI is created by Channel, not needed here separately
    
    # Create task prompts
    if args.task_prompt:
        task_prompts = [args.task_prompt] * args.batch_size
    else:
        # Default task prompt
        task_prompts = ["<CAPTION>"] * args.batch_size
    print(f"Task prompts: {task_prompts}")
    
    # Generate text embeddings at top level (shared between Tx/Rx)
    # Use all-zero dummy image - processor only needs image presence, not content
    print("\n=== Generating Text Embeddings (Shared) ===")
    import numpy as np
    from PIL import Image as PILImage
    dummy_image = PILImage.fromarray(np.zeros((768, 768, 3), dtype=np.uint8))
    
    with torch.no_grad():
        # Use processor with dummy image to get correct tokenization
        inputs = florence2_model.processor(
            text=task_prompts,
            images=[dummy_image] * len(task_prompts),
            return_tensors="pt"
        )
        input_ids = inputs["input_ids"].to(
            device=device,
            dtype=torch.long
        )
        embedding_layer = florence2_model.model.get_input_embeddings()
        text_embeddings = embedding_layer(input_ids)
    print(f"Text embeddings shape: {text_embeddings.shape}")
    print(f"Text embeddings will be shared with Tx and Rx")
    
    # Initialize Transmitter
    print("\n=== Initializing Transmitter ===")
    transmitter = Transmitter(
        florence2_model=florence2_model,
        mode=args.mode,
        task_embedding_dim=args.task_embedding_dim
    ).to(device)
    print(f"Transmitter mode: {args.mode}")
    print(f"Transmitter output dimension: {transmitter.get_output_dim()}")
    print(f"Transmitter output shape: {transmitter.get_output_shape(args.batch_size)}")
    
    # Initialize Channel
    print("\n=== Initializing Channel ===")
    channel = create_channel(
        channel_type=args.channel_type,
        effective_snr_db=args.snr_db
    )
    print(f"Channel type: {args.channel_type}, SNR: {args.snr_db} dB")
    
    # Initialize Receiver
    print("\n=== Initializing Receiver ===")
    receiver = Receiver(
        florence2_model=florence2_model,
        mode=args.mode
    ).to(device)
    
    # Load and preprocess image
    print("\n=== Processing Image ===")
    # Default test image URL (same as test_semcom.py)
    default_image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg?download=true"
    
    image = None  # PIL Image for processing
    image_tensor = None  # Tensor for fallback
    
    if args.image_path:
        image = load_image(args.image_path, target_size=(args.image_size, args.image_size))
        print(f"✓ Image loaded from path: {image.size}")
    else:
        # Use default test image URL
        try:
            print(f"Loading test image from URL: {default_image_url}")
            image = Image.open(requests.get(default_image_url, stream=True).raw)
            print(f"✓ Image loaded: {image.size}")
        except Exception as e:
            print(f"✗ Failed to load image from URL: {e}")
            print("Using dummy image instead...")
            # Create dummy image as fallback
            image_tensor = torch.randn(
                args.batch_size, 3, args.image_size, args.image_size
            ).to(device)
            print(f"Using dummy image: {image_tensor.shape}")
    
    # Transmitter processing
    # Transmitter: Image -> Vision Encoder -> Vision Embedding
    print("\n=== Transmitter Processing ===")
    with torch.no_grad():
        # Pass PIL Image directly (transmitter handles preprocessing)
        if image is not None:
            tx_output = transmitter(image)  # Vision embedding
        elif image_tensor is not None:
            tx_output = transmitter(image_tensor)  # Vision embedding (dummy image)
        else:
            raise ValueError("No image provided (neither path, URL, nor dummy image)")
    print(f"Transmitter output (vision embedding) shape: {tx_output.shape}")
    
    # Channel transmission
    print("\n=== Channel Transmission ===")
    with torch.no_grad():
        received_signal = channel(tx_output)
    print(f"Received signal shape: {received_signal.shape}")
    
    # Receiver processing
    # Receiver: Received Vision Embedding + Text Embeddings (shared) -> Merged Embeddings
    print("\n=== Receiver Processing ===")
    with torch.no_grad():
        # Pass text_embeddings (shared from top level) instead of task_prompts strings
        merged_embeds, attention_mask = receiver(received_signal, text_embeddings)
    print(f"Receiver merged embeddings shape: {merged_embeds.shape}")
    print(f"Receiver attention mask shape: {attention_mask.shape}")
    
    # Generate text output using receiver.generate() (if noiseless channel)
    receiver_result = None
    reference_result = None
    
    if args.channel_type == 'noiseless' and image is not None:
        print("\n=== Generating Text Output ===")
        
        # Receiver generation
        print("\n[Receiver] Generating text...")
        try:
            with torch.no_grad():
                generated_ids_rx = receiver.generate(
                    received_signal,
                    text_embeddings,  # Use shared text_embeddings from top level
                    max_new_tokens=1024,
                    num_beams=3,
                    do_sample=False,
                )
            
            # Decode and parse receiver output
            generated_text_receiver = florence2_model.processor.batch_decode(
                generated_ids_rx, skip_special_tokens=False
            )[0]
            receiver_result = florence2_model.processor.post_process_generation(
                generated_text_receiver,
                task=task_prompts[0],
                image_size=(image.width, image.height)
            )
            print(f"✓ Receiver result: {receiver_result}")
        except Exception as e:
            print(f"✗ Receiver generation failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Reference (direct model generation)
        print("\n[Reference] Generating text...")
        try:
            # Process image and prompt using processor
            inputs = {}
            for k, v in florence2_model.processor(
                text=task_prompts[0],
                images=image,
                return_tensors="pt"
            ).items():
                if isinstance(v, torch.Tensor):
                    if k == 'input_ids':
                        # input_ids must be Long (int64)
                        inputs[k] = v.to(device=device).long()
                    else:
                        # Other tensors (pixel_values) use model dtype
                        inputs[k] = v.to(device=device, dtype=florence2_model.model.dtype)
                else:
                    inputs[k] = v
            
            # Generate using the model
            with torch.no_grad():
                generated_ids = florence2_model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=1024,
                    do_sample=False,
                    num_beams=3,
                )
            
            # Decode and parse
            generated_text = florence2_model.processor.batch_decode(
                generated_ids, skip_special_tokens=False
            )[0]
            
            reference_result = florence2_model.processor.post_process_generation(
                generated_text,
                task=task_prompts[0],
                image_size=(image.width, image.height)
            )
            print(f"✓ Reference result: {reference_result}")
        except Exception as e:
            print(f"✗ Reference generation failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Compare results
        print("\n=== Comparison ===")
        if receiver_result is not None and reference_result is not None:
            if receiver_result == reference_result:
                print("✓ Results match! Semantic communication pipeline works correctly.")
            else:
                print("⚠ Results differ:")
                print(f"  Receiver: {receiver_result}")
                print(f"  Reference: {reference_result}")
        else:
            print("⚠ Could not compare results (generation failed)")
    
    # Calculate metrics
    print("\n=== Results ===")
    tx_power = torch.mean(tx_output ** 2).item()
    rx_power = torch.mean(received_signal ** 2).item()
    noise_power = rx_power - tx_power if args.channel_type != 'noiseless' else 0.0
    
    print(f"Transmitted signal power: {tx_power:.6f}")
    print(f"Received signal power: {rx_power:.6f}")
    if args.channel_type != 'noiseless':
        print(f"Noise power: {noise_power:.6f}")
        actual_snr = 10 * math.log10(tx_power / noise_power) if noise_power > 0 else float('inf')
        print(f"Actual SNR: {actual_snr:.2f} dB")
    
    print("\n=== Simulation Complete ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Semantic Communication System using Florence-2"
    )
    
    # Model arguments
    parser.add_argument(
        '--model_name',
        type=str,
        default='microsoft/Florence-2-base',
        help='Florence-2 model name from HuggingFace'
    )
    parser.add_argument(
        '--model_size',
        type=str,
        default='base',
        choices=['base', 'large'],
        help='Model size (for dimension lookup)'
    )
    
    # Mode arguments
    parser.add_argument(
        '--mode',
        type=str,
        default='vision_tower',
        choices=['vision_tower', 'image_proj_norm'],
        help='Processing mode: vision_tower (Mode 1) or image_proj_norm (Mode 2)'
    )
    
    # Task prompt arguments
    parser.add_argument(
        '--task_prompt',
        type=str,
        default=None,
        help='Task prompt string (e.g., "<CAPTION>")'
    )
    parser.add_argument(
        '--task_embedding_dim',
        type=int,
        default=768,
        help='Dimension of task embedding (for linear embedding option, if used)'
    )
    
    # Transmitter arguments
    
    # Channel arguments
    parser.add_argument(
        '--channel_type',
        type=str,
        default='awgn',
        choices=['noiseless', 'awgn', 'rayleigh'],
        help='Channel type'
    )
    parser.add_argument(
        '--snr_db',
        type=float,
        default=20.0,
        help='Effective SNR in dB'
    )
    
    # Input arguments
    parser.add_argument(
        '--image_path',
        type=str,
        default=None,
        help='Path to input image (if None, uses dummy image)'
    )
    parser.add_argument(
        '--image_size',
        type=int,
        default=224,
        help='Image size (H=W)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1,
        help='Batch size'
    )
    
    # Device arguments
    parser.add_argument(
        '--cpu',
        action='store_true',
        help='Force CPU usage'
    )
    
    args = parser.parse_args()
    
    main(args)
