"""
Test script for Florence-2 following HuggingFace official examples.
Tests the model with the same car.jpg image and task prompts.
"""

import torch
import requests
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM


def test_florence2_official_way():
    """Test Florence-2 using the official HuggingFace example method."""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    print("=" * 60)
    print("Florence-2 Official Example Test")
    print("=" * 60)
    
    # Load model and processor
    # Note: HuggingFace docs use torch_dtype but it's deprecated, we use dtype with attn_implementation
    print("\n[1/3] Loading model and processor...")
    try:
        # Try with attn_implementation first (our fix)
        model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Florence-2-base",
            dtype=torch_dtype,
            attn_implementation="eager",  # Fix for _supports_sdpa attribute error
            trust_remote_code=True
        ).to(device)
    except Exception as e:
        print(f"Warning: Failed with attn_implementation, trying without: {e}")
        # Fallback to original HuggingFace way
        model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Florence-2-base",
            torch_dtype=torch_dtype,  # Original HuggingFace way
            trust_remote_code=True
        ).to(device)
    
    processor = AutoProcessor.from_pretrained(
        "microsoft/Florence-2-base",
        trust_remote_code=True
    )
    print("✓ Model and processor loaded")
    
    # Load test image
    print("\n[2/3] Loading test image...")
    url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg?download=true"
    image = Image.open(requests.get(url, stream=True).raw)
    print(f"✓ Image loaded: {image.size}")
    
    # Test different tasks
    print("\n[3/3] Testing different tasks...")
    
    tasks = [
        ("<CAPTION>", "Caption"),
        ("<DETAILED_CAPTION>", "Detailed Caption"),
        ("<OD>", "Object Detection"),
    ]
    
    for task_prompt, task_name in tasks:
        print(f"\n--- Testing {task_name} ({task_prompt}) ---")
        try:
            inputs = processor(text=task_prompt, images=image, return_tensors="pt").to(device, torch_dtype)
            
            # Generate with use_cache=False to avoid past_key_values issue
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                do_sample=False,
                num_beams=3,
                use_cache=False,  # Required to avoid past_key_values None error
            )
            
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            
            parsed_answer = processor.post_process_generation(
                generated_text, 
                task=task_prompt, 
                image_size=(image.width, image.height)
            )
            
            print(f"✓ {task_name} result:")
            print(parsed_answer)
            
        except Exception as e:
            print(f"✗ {task_name} failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)


if __name__ == "__main__":
    test_florence2_official_way()
