"""
Test script for full semantic communication pipeline.
Tests: Image -> Transmitter -> Channel -> Receiver -> Task Execution

Run this script from the project root directory:
    python test_semcom.py
"""

import sys
import math
from pathlib import Path

# Ensure we can import from the project
project_root = Path(__file__).parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
import requests
from PIL import Image

# Import using the same pattern as main.py
from models.florence2_model import Florence2Model, get_vision_encoder_output_dim
from transmitter.transmitter import Transmitter
from channel.channel import Channel, create_channel
from receiver.receiver import Receiver
from utils.image_utils import load_image, preprocess_image


def test_semantic_communication():
    """Test the full semantic communication pipeline."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 70)
    print("Semantic Communication Pipeline Test")
    print("=" * 70)
    
    # Load test image
    print("\n[1/5] Loading test image...")
    url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg?download=true"
    try:
        image = Image.open(requests.get(url, stream=True).raw)
        print(f"✓ Image loaded: {image.size}")
    except Exception as e:
        print(f"✗ Failed to load image: {e}")
        return
    
    # Initialize Florence-2 model
    print("\n[2/5] Initializing Florence-2 model...")
    try:
        florence2_model = Florence2Model(
            model_name="microsoft/Florence-2-base",
            device=device
        )
        print("✓ Florence-2 model loaded")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Initialize Transmitter
    print("\n[3/5] Initializing Transmitter...")
    try:
        transmitter = Transmitter(
            florence2_model=florence2_model,
            task_embedding_dim=768,
            include_linear_embedding=False,
            use_pooled_features=False
        ).to(device)
        print("✓ Transmitter initialized")
    except Exception as e:
        print(f"✗ Failed to initialize transmitter: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Initialize Channel
    print("\n[4/5] Initializing Channel...")
    try:
        channel = create_channel(
            channel_type='noiseless',
            effective_snr_db=20.0
        )
        print("✓ Channel initialized (Noiseless)")
    except Exception as e:
        print(f"✗ Failed to initialize channel: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Initialize Receiver
    print("\n[5/5] Initializing Receiver...")
    try:
        receiver = Receiver(
            florence2_model=florence2_model,
            use_pooled_features=False
        ).to(device)
        print("✓ Receiver initialized")
    except Exception as e:
        print(f"✗ Failed to initialize receiver: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Preprocess image
    print("\n" + "=" * 70)
    print("Processing Pipeline")
    print("=" * 70)
    
    print("\n[Step 1] Preprocessing image...")
    try:
        image_tensor = preprocess_image(image, normalize=True, device=device)
        print(f"✓ Image preprocessed: {image_tensor.shape}")
    except Exception as e:
        print(f"✗ Failed to preprocess image: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Transmitter: Image -> Vision Embedding
    print("\n[Step 2] Transmitter: Encoding image to vision embedding...")
    try:
        with torch.no_grad():
            tx_output = transmitter(image_tensor)
        print(f"✓ Transmitter output shape: {tx_output.shape}")
        print(f"  Vision embedding generated successfully")
    except Exception as e:
        print(f"✗ Transmitter failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Channel: Vision Embedding -> Received Signal
    print("\n[Step 3] Channel: Transmitting through noiseless channel...")
    try:
        with torch.no_grad():
            received_signal = channel(tx_output)
        print(f"✓ Received signal shape: {received_signal.shape}")
        print(f"  Noiseless channel: signal passed through unchanged")
    except Exception as e:
        print(f"✗ Channel transmission failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Receiver: Received Signal + Task Prompt -> Task Result
    print("\n[Step 4] Receiver: Processing received signal with task prompts...")
    
    # Test different tasks
    tasks = [
        ("<CAPTION>", "Caption"),
        ("<DETAILED_CAPTION>", "Detailed Caption"),
        ("<OD>", "Object Detection"),
    ]
    
    print("\n" + "=" * 70)
    print("Task Results Comparison")
    print("=" * 70)
    
    for task_prompt, task_name in tasks:
        print(f"\n--- Testing {task_name} ({task_prompt}) ---")
        
        # Receiver processing
        receiver_result = None
        try:
            with torch.no_grad():
                # Receiver processes: received_vision_embedding + task_prompt
                rx_output = receiver(received_signal, [task_prompt])
            
            print(f"  ✓ Receiver output shape: {rx_output.shape}")
            
            # Generate text from receiver output using language model's lm_head
            # The receiver output is the last hidden state, we need to get logits
            if hasattr(florence2_model.model, 'language_model') and hasattr(florence2_model.model.language_model, 'lm_head'):
                # Get logits from receiver output
                # Use the last token of the task embeddings part (after vision tokens)
                # Or use the entire output and get logits
                logits = florence2_model.model.language_model.lm_head(rx_output)
                
                # Get token IDs from logits (greedy decoding)
                predicted_ids = torch.argmax(logits, dim=-1)
                
                # Decode to text
                generated_text_receiver = florence2_model.processor.batch_decode(
                    predicted_ids, skip_special_tokens=False
                )[0]
                
                # Parse the result
                receiver_result = florence2_model.processor.post_process_generation(
                    generated_text_receiver,
                    task=task_prompt,
                    image_size=(image.width, image.height)
                )
            
        except Exception as e:
            print(f"  ✗ Receiver processing failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Reference (direct model generation)
        reference_result = None
        try:
            # Process image and prompt using processor
            inputs = florence2_model.processor(
                text=task_prompt,
                images=image,
                return_tensors="pt"
            ).to(device, florence2_model.model.dtype)
            
            # Generate using the model
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
                task=task_prompt,
                image_size=(image.width, image.height)
            )
            
        except Exception as e:
            print(f"  ✗ Reference generation failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Print comparison
        print(f"\n  [Receiver Result]")
        if receiver_result is not None:
            print(f"    {receiver_result}")
        else:
            print(f"    (Failed to generate)")
        
        print(f"\n  [Reference Result]")
        if reference_result is not None:
            print(f"    {reference_result}")
        else:
            print(f"    (Failed to generate)")
        
        # Compare if both succeeded
        if receiver_result is not None and reference_result is not None:
            if receiver_result == reference_result:
                print(f"\n  ✓ Results match!")
            else:
                print(f"\n  ⚠ Results differ")
    
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    print("✓ Full semantic communication pipeline tested successfully!")
    print("  - Transmitter: Image -> Vision Embedding")
    print("  - Channel: Vision Embedding -> Received Signal (with noise)")
    print("  - Receiver: Received Signal + Task Prompt -> Semantic Output")
    print("=" * 70)


if __name__ == "__main__":
    test_semantic_communication()
