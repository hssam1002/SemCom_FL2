"""
Main script for semantic communication system using Florence-2.
"""

import torch
import argparse
from pathlib import Path

from models.florence2_model import Florence2Model, get_vision_encoder_output_dim
from transmitter.transmitter import Transmitter
from channel.channel import Channel, create_channel
from receiver.receiver import Receiver
from shared.task_embedding import TaskEmbedding
from shared.csi import CSI
from utils.image_utils import load_image, preprocess_image


def main(args):
    """
    Main function for semantic communication simulation.
    
    Args:
        args: Command line arguments
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"Using device: {device}")
    
    # Get vision encoder output dimension
    vision_dim = get_vision_encoder_output_dim(args.model_size)
    print(f"Vision encoder output dimension: {vision_dim}")
    
    # Initialize Florence-2 model
    print("\n=== Initializing Florence-2 Model ===")
    florence2_model = Florence2Model(
        model_name=args.model_name,
        device=device
    )
    
    # Initialize shared components
    print("\n=== Initializing Shared Components ===")
    csi = CSI(
        effective_snr_db=args.snr_db,
        channel_type=args.channel_type
    )
    print(f"CSI: {csi}")
    
    # Create task prompts (Florence-2 will handle task embedding internally)
    if args.task_prompt:
        task_prompts = [args.task_prompt] * args.batch_size
    else:
        # Default task prompt
        task_prompts = ["What does the image describe?"] * args.batch_size
    print(f"Task prompts: {task_prompts}")
    
    # Initialize Transmitter
    print("\n=== Initializing Transmitter ===")
    transmitter = Transmitter(
        florence2_model=florence2_model,
        task_embedding_dim=args.task_embedding_dim,
        include_linear_embedding=args.include_linear_embedding,
        use_pooled_features=args.use_pooled_features
    ).to(device)
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
        use_pooled_features=args.use_pooled_features
    ).to(device)
    
    # Load and preprocess image
    print("\n=== Processing Image ===")
    if args.image_path:
        image = load_image(args.image_path, target_size=(args.image_size, args.image_size))
        image_tensor = preprocess_image(image, normalize=True, device=device)
    else:
        # Create dummy image for testing
        image_tensor = torch.randn(
            args.batch_size, 3, args.image_size, args.image_size
        ).to(device)
        print(f"Using dummy image: {image_tensor.shape}")
    
    # Transmitter processing
    # Transmitter: Image -> Vision Encoder -> Vision Embedding
    print("\n=== Transmitter Processing ===")
    with torch.no_grad():
        tx_output = transmitter(image_tensor)  # Vision embedding
    print(f"Transmitter output (vision embedding) shape: {tx_output.shape}")
    
    # Channel transmission
    print("\n=== Channel Transmission ===")
    with torch.no_grad():
        received_signal = channel(tx_output)
    print(f"Received signal shape: {received_signal.shape}")
    
    # Receiver processing
    # Receiver: Received Vision Embedding + Task Prompts -> Transformer Encoder -> Transformer Decoder
    print("\n=== Receiver Processing ===")
    with torch.no_grad():
        rx_output = receiver(received_signal, task_prompts)
    print(f"Receiver output shape: {rx_output.shape}")
    
    # Calculate metrics
    print("\n=== Results ===")
    tx_power = torch.mean(tx_output ** 2).item()
    rx_power = torch.mean(received_signal ** 2).item()
    noise_power = rx_power - tx_power if args.channel_type != 'noiseless' else 0.0
    
    print(f"Transmitted signal power: {tx_power:.6f}")
    print(f"Received signal power: {rx_power:.6f}")
    if args.channel_type != 'noiseless':
        print(f"Noise power: {noise_power:.6f}")
        actual_snr = 10 * torch.log10(tx_power / noise_power) if noise_power > 0 else float('inf')
        print(f"Actual SNR: {actual_snr:.2f} dB")
    
    print("\n=== Simulation Complete ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Semantic Communication System using Florence-2"
    )
    
    # Model arguments
    parser.add_argument(
        '--model-name',
        type=str,
        default='microsoft/Florence-2-base',
        help='Florence-2 model name from HuggingFace'
    )
    parser.add_argument(
        '--model-size',
        type=str,
        default='base',
        choices=['base', 'large'],
        help='Model size (for dimension lookup)'
    )
    
    # Task prompt arguments
    parser.add_argument(
        '--task-prompt',
        type=str,
        default=None,
        help='Task prompt string (e.g., "What does the image describe?")'
    )
    parser.add_argument(
        '--task-embedding-dim',
        type=int,
        default=768,
        help='Dimension of task embedding (for linear embedding option, if used)'
    )
    
    # Transmitter arguments
    parser.add_argument(
        '--include-linear-embedding',
        action='store_true',
        help='Include linear embedding to match task embedding dimension'
    )
    parser.add_argument(
        '--use-pooled-features',
        action='store_true',
        help='Use pooled features (CLS token) instead of full sequence'
    )
    
    # Channel arguments
    parser.add_argument(
        '--channel-type',
        type=str,
        default='awgn',
        choices=['noiseless', 'awgn', 'rayleigh'],
        help='Channel type'
    )
    parser.add_argument(
        '--snr-db',
        type=float,
        default=20.0,
        help='Effective SNR in dB'
    )
    
    # Input arguments
    parser.add_argument(
        '--image-path',
        type=str,
        default=None,
        help='Path to input image (if None, uses dummy image)'
    )
    parser.add_argument(
        '--image-size',
        type=int,
        default=224,
        help='Image size (H=W)'
    )
    parser.add_argument(
        '--batch-size',
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
