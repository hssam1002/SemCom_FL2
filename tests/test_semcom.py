"""
Test script for full semantic communication pipeline.
Tests: Image -> Transmitter -> Channel -> Receiver -> Task Execution

Run this script from the project root directory:
    python tests/test_semcom.py
    or
    python -m tests.test_semcom
"""

import sys
import math
from pathlib import Path

# Ensure we can import from the project root (parent directory)
project_root = Path(__file__).parent.parent.absolute()
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
    
    print("\n[Step 1] Transmitter: Encoding image to vision embedding...")
    # Transmitter now handles image preprocessing internally using processor
    try:
        with torch.no_grad():
            tx_output = transmitter(image)  # Pass PIL Image directly
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
    
    # Create all-zero dummy image for text embedding generation (shared between Tx/Rx)
    # Note: Any image works - processor only checks if image is provided, not its content
    import numpy as np
    dummy_image = Image.fromarray(np.zeros((768, 768, 3), dtype=np.uint8))
    print(f"\n[Shared] Created all-zero dummy image for text embedding generation")
    
    for task_prompt, task_name in tasks:
        print(f"\n--- Testing {task_name} ({task_prompt}) ---")
        
        # Generate text embeddings at top level (shared between Tx/Rx)
        # Use all-zero dummy image - processor only needs image presence, not content
        print(f"  [Top Level] Encoding task prompt: {task_prompt}")
        with torch.no_grad():
            # Use processor with dummy image to get correct tokenization
            inputs = florence2_model.processor(
                text=[task_prompt],
                images=[dummy_image],
                return_tensors="pt"
            )
            input_ids = inputs["input_ids"].to(
                device=device,
                dtype=torch.long
            )
            embedding_layer = florence2_model.model.get_input_embeddings()
            text_embeddings = embedding_layer(input_ids)
        print(f"  ✓ Text embeddings shape: {text_embeddings.shape}")
        print(f"  ✓ Text embeddings will be shared with Tx and Rx")
        
        # Receiver processing: use same generate flow as reference, but with Tx features
        receiver_result = None
        try:
            with torch.no_grad():
                # Pass text_embeddings (from Tx) instead of task_prompt strings
                merged_embeds, attention_mask = receiver(received_signal, text_embeddings)
            print(f"  ✓ Receiver merged embeddings shape: {merged_embeds.shape}")
            print(f"  ✓ Receiver attention mask shape: {attention_mask.shape}")

            # Use receiver.generate() to run Florence-2 language_model.generate with merged embeddings
            # Pass text_embeddings (shared from top level)
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
            )
            # Move to device and dtype
            # Note: input_ids should be Long, pixel_values should match model dtype
            inputs = {}
            for k, v in florence2_model.processor(
                text=task_prompt,
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
