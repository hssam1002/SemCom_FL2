"""
Check script to verify Florence-2 model loading.
Tests if the model can be loaded correctly using AutoModelForCausalLM.
"""

import torch
import json
from transformers import AutoProcessor, AutoModelForCausalLM
from transformers import __version__ as transformers_version


def log_debug(location, message, data, hypothesis_id=None):
    """Log debug information."""
    # #region agent log
    try:
        log_entry = {
            "sessionId": "debug-session",
            "runId": "run1",
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(torch.cuda.current_device() if torch.cuda.is_available() else 0) if hasattr(torch.cuda, 'current_device') else 0
        }
        with open("/data4/hongsik/SemCom_FL2/.cursor/debug.log", "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception:
        pass
    # #endregion


def check_florence2_model():
    """Check if Florence-2 model loads correctly."""
    model_name = "microsoft/Florence-2-base"
    
    print("=" * 60)
    print("Florence-2 Model Loading Check")
    print("=" * 60)
    
    # Log environment info
    log_debug("check.py:18", "Environment check", {
        "transformers_version": transformers_version,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available()
    }, "B")
    
    # Set device and dtype
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    print(f"\nDevice: {device}")
    print(f"Torch dtype: {torch_dtype}")
    print(f"Model name: {model_name}")
    print(f"Transformers version: {transformers_version}")
    
    log_debug("check.py:29", "Before model load", {
        "model_name": model_name,
        "device": device,
        "torch_dtype": str(torch_dtype),
        "using_torch_dtype_param": True
    }, "A")
    
    try:
        # Load model - Hypothesis A: try dtype instead of torch_dtype
        print("\n[1/3] Loading model...")
        log_debug("check.py:35", "Attempting model load with torch_dtype", {
            "param_name": "torch_dtype",
            "value": str(torch_dtype)
        }, "A")
        
        # Try with dtype parameter (Hypothesis A)
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=torch_dtype,  # Hypothesis A: use dtype instead
                trust_remote_code=True
            ).to(device)
            log_debug("check.py:45", "Model load success with dtype", {}, "A")
        except Exception as e1:
            log_debug("check.py:48", "Model load failed with dtype", {
                "error": str(e1),
                "error_type": type(e1).__name__
            }, "A")
            
            # Try with attn_implementation (Hypothesis C)
            log_debug("check.py:53", "Attempting model load with attn_implementation", {
                "attn_implementation": "eager"
            }, "C")
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    dtype=torch_dtype,
                    attn_implementation="eager",  # Hypothesis C
                    trust_remote_code=True
                ).to(device)
                log_debug("check.py:63", "Model load success with attn_implementation", {}, "C")
            except Exception as e2:
                log_debug("check.py:66", "Model load failed with attn_implementation", {
                    "error": str(e2),
                    "error_type": type(e2).__name__
                }, "C")
                
                # Try without dtype (Hypothesis D)
                log_debug("check.py:71", "Attempting model load without dtype", {}, "D")
                try:
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        trust_remote_code=True
                    ).to(device)
                    log_debug("check.py:77", "Model load success without dtype", {}, "D")
                except Exception as e3:
                    log_debug("check.py:80", "Model load failed without dtype", {
                        "error": str(e3),
                        "error_type": type(e3).__name__
                    }, "D")
                    raise e3
        print("✓ Model loaded successfully!")
        log_debug("check.py:85", "Model loaded", {
            "model_type": str(type(model)),
            "has_florence": hasattr(model, 'florence')
        }, "B")
        
        # Load processor
        print("\n[2/3] Loading processor...")
        processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        print("✓ Processor loaded successfully!")
        log_debug("check.py:95", "Processor loaded", {
            "processor_type": str(type(processor))
        }, "B")
        
        # Check model structure
        print("\n[3/3] Checking model structure...")
        print(f"Model type: {type(model)}")
        print(f"Model config: {model.config.model_type}")
        
        # Check if florence attribute exists
        if hasattr(model, 'florence'):
            print("✓ Model has 'florence' attribute")
            florence = model.florence
            log_debug("check.py:107", "Florence attribute check", {
                "has_vision_encoder": hasattr(florence, 'vision_encoder'),
                "has_text_encoder": hasattr(florence, 'text_encoder'),
                "has_text_decoder": hasattr(florence, 'text_decoder')
            }, "B")
            
            # Check vision encoder
            if hasattr(florence, 'vision_encoder'):
                print("✓ Vision encoder found")
                vision_encoder = florence.vision_encoder
                print(f"  Vision encoder type: {type(vision_encoder)}")
            else:
                print("✗ Vision encoder not found")
            
            # Check text encoder
            if hasattr(florence, 'text_encoder'):
                print("✓ Text encoder found")
                text_encoder = florence.text_encoder
                print(f"  Text encoder type: {type(text_encoder)}")
            else:
                print("✗ Text encoder not found")
            
            # Check text decoder
            if hasattr(florence, 'text_decoder'):
                print("✓ Text decoder found")
                text_decoder = florence.text_decoder
                print(f"  Text decoder type: {type(text_decoder)}")
            else:
                print("✗ Text decoder not found")
        else:
            print("✗ Model does not have 'florence' attribute")
            available_attrs = [attr for attr in dir(model) if not attr.startswith('_')]
            print("Available attributes:", available_attrs)
            log_debug("check.py:140", "Model attributes", {
                "available_attrs": available_attrs
            }, "B")
        
        # Print model config summary
        print("\n" + "=" * 60)
        print("Model Config Summary")
        print("=" * 60)
        config = model.config
        print(f"Model type: {getattr(config, 'model_type', 'N/A')}")
        print(f"Hidden size: {getattr(config, 'hidden_size', 'N/A')}")
        print(f"Num layers: {getattr(config, 'num_hidden_layers', 'N/A')}")
        print(f"Num attention heads: {getattr(config, 'num_attention_heads', 'N/A')}")
        
        # Check processor
        print("\n" + "=" * 60)
        print("Processor Check")
        print("=" * 60)
        print(f"Processor type: {type(processor)}")
        if hasattr(processor, 'tokenizer'):
            print("✓ Tokenizer found")
        if hasattr(processor, 'image_processor'):
            print("✓ Image processor found")
        
        print("\n" + "=" * 60)
        print("✓ All checks passed! Model is ready to use.")
        print("=" * 60)
        
        return model, processor
        
    except Exception as e:
        print(f"\n✗ Error loading model: {e}")
        import traceback
        error_trace = traceback.format_exc()
        log_debug("check.py:150", "Model load error", {
            "error": str(e),
            "error_type": type(e).__name__,
            "traceback": error_trace[:500]  # First 500 chars
        }, "B")
        traceback.print_exc()
        return None, None


if __name__ == "__main__":
    model, processor = check_florence2_model()
    
    if model is not None and processor is not None:
        print("\nModel and processor loaded successfully!")
        print("You can now use them in your semantic communication system.")
    else:
        print("\nFailed to load model. Please check the error messages above.")