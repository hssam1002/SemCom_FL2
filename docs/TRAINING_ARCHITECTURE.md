# Training Architecture

## Frozen vs Trainable Components

### Always Frozen (Florence-2 Pre-trained)
- **vision_tower**: Vision encoder (DaViT)
- **image_pos_embed**: 2D positional embedding
- **visual_temporal_embed**: Temporal embedding
- **image_feature_source**: Pooling strategies
- **image_projection**: Linear projection (1024 → 768)
- **image_proj_norm**: Layer normalization
- **language_model**: Text generation model
- **processor**: Image/text preprocessing
- **tokenizer**: Text tokenization

### Future Trainable (To be added)
- **Compression Module** (in Transmitter)
  - Location: After vision_tower output
  - Input: Vision tower features (1024-dim)
  - Output: Compressed features
  - Purpose: Compress vision features for transmission

- **Decompression Module** (in Receiver)
  - Location: Before image_pos_embed
  - Input: Received compressed features
  - Output: Decompressed features (matching vision_tower output shape)
  - Purpose: Recover vision features from compressed representation

## Current Training Flow

```
Image → Transmitter (vision_tower frozen) → Channel → Receiver (Florence-2 modules frozen) → Language Model (frozen) → Loss
```

**Current Status:**
- All Florence-2 modules: **FROZEN**
- Trainable parameters: **0** (expected)
- Loss computation: **For monitoring only**

## Future Training Flow (with Compression)

```
Image → Transmitter:
  vision_tower (frozen) → compression_module (trainable) → Channel
  
Channel → Receiver:
  decompression_module (trainable) → image_pos_embed (frozen) → ... → language_model (frozen) → Loss
```

**Future Status:**
- All Florence-2 modules: **FROZEN**
- Compression module: **TRAINABLE**
- Decompression module: **TRAINABLE**
- End-to-end training: **Enabled**

## Implementation Notes

1. **Florence-2 Freezing**: All parameters in `florence2_model.model` are set to `requires_grad=False`
2. **Forward Pass**: 
   - Frozen modules (requires_grad=False) don't need `torch.no_grad()` or `detach()` to prevent gradients
   - However, `no_grad()` can be used for memory efficiency (prevents computation graph creation)
   - Current implementation: `no_grad()` and `detach()` are removed to prepare for future trainable modules
3. **Gradient Flow**: 
   - Frozen modules (requires_grad=False) automatically won't compute gradients
   - When compression modules are added, gradients will flow to them automatically
4. **Optimizer**: Only trainable parameters (compression modules) will be included in optimizer

## requires_grad=False vs torch.no_grad()

- **requires_grad=False**: Parameter is frozen, but computation graph is still created (can use more memory)
- **torch.no_grad()**: Prevents computation graph creation entirely (more memory efficient)
- **Both prevent gradients**, but `no_grad()` is more efficient for frozen modules
- **For future trainable modules**: Remove `no_grad()` and `detach()` to allow gradient flow

## Adding Compression Modules

When ready to add compression modules:

1. **In Transmitter**:
   ```python
   # After vision_tower output
   vision_features = model.vision_tower(...)  # frozen
   compressed_features = self.compression_module(vision_features)  # trainable
   return compressed_features
   ```

2. **In Receiver**:
   ```python
   # Before image_pos_embed
   decompressed_features = self.decompression_module(received_signal)  # trainable
   vision_features = decompressed_features
   # Then continue with frozen Florence-2 modules...
   ```

3. **Remove detach()**: Once compression modules are added, remove `detach()` calls to allow gradient flow
