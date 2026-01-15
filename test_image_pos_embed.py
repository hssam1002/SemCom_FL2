"""
Test how image_pos_embed actually works.
"""

import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import requests

def test_image_pos_embed():
    """Test image_pos_embed behavior."""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model_name = "microsoft/Florence-2-base"
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch_dtype,
        attn_implementation="eager",
        trust_remote_code=True
    ).to(device)
    
    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    
    url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg?download=true"
    image = Image.open(requests.get(url, stream=True).raw)
    inputs = processor(text="<CAPTION>", images=image, return_tensors="pt").to(device, torch_dtype)
    
    # Get vision features after image_proj_norm
    vision_output = model.vision_tower.forward_features_unpool(inputs['pixel_values'])
    if isinstance(vision_output, tuple):
        vision_features = vision_output[0]
    else:
        vision_features = vision_output
    
    proj_features = torch.matmul(vision_features, model.image_projection)
    norm_features = model.image_proj_norm(proj_features)
    
    print(f"norm_features shape: {norm_features.shape}")
    
    # Test image_pos_embed
    print("\nTesting image_pos_embed...")
    try:
        pos_output = model.image_pos_embed(inputs['pixel_values'])
        print(f"image_pos_embed output shape: {pos_output.shape}")
        print(f"image_pos_embed output type: {type(pos_output)}")
        
        # Check if we need to add or if it's already combined
        if pos_output.shape == norm_features.shape:
            print("  ✓ Shapes match - can be added")
            combined = norm_features + pos_output
            print(f"  Combined shape: {combined.shape}")
        else:
            print(f"  ✗ Shapes don't match")
            print(f"    norm_features: {norm_features.shape}")
            print(f"    pos_output: {pos_output.shape}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_image_pos_embed()
