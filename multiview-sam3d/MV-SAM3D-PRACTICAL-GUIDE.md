# MV-SAM3D: Multi-View 3D Reconstruction - Practical Guide

**A Comprehensive Guide to Understanding and Using MV-SAM3D**

Last Updated: 2026-01-13

---

## Table of Contents

1. [What is MV-SAM3D?](#what-is-mv-sam3d)
2. [How It Works: The Big Picture](#how-it-works-the-big-picture)
3. [Key Innovation: Multi-View Fusion](#key-innovation-multi-view-fusion)
4. [Technical Architecture](#technical-architecture)
5. [Differences from SAM-3D-Objects](#differences-from-sam-3d-objects)
6. [Practical Usage](#practical-usage)
7. [Understanding the Parameters](#understanding-the-parameters)
8. [Common Use Cases](#common-use-cases)
9. [Performance Considerations](#performance-considerations)
10. [Troubleshooting](#troubleshooting)

---

## What is MV-SAM3D?

**MV-SAM3D** is a multi-view extension of Meta's SAM 3D Objects that reconstructs 3D models from **multiple photographs** of the same object taken from different angles.

### The Problem It Solves

**SAM 3D Objects (Original):**
- Takes **1 photo** → Generates 3D model
- Limited by single viewpoint information
- May miss occluded or hidden parts

**MV-SAM3D (Enhanced):**
- Takes **multiple photos** → Generates better 3D model
- Combines information from all viewpoints intelligently
- Handles occlusions and difficult viewing angles
- Produces more accurate geometry and texture

### Real-World Example

Imagine photographing a coffee mug:
- **View 1**: Front view (sees logo, but not handle clearly)
- **View 2**: Side view (sees handle, but logo is angled)
- **View 3**: Back view (sees handle attachment, misses logo)

MV-SAM3D combines all three views, weighing each view based on:
- **Confidence**: How certain the model is about its predictions
- **Visibility**: Which parts are visible vs. occluded in each view
- **Quality**: Which view has the best information for each part

---

## How It Works: The Big Picture

### The Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│ INPUT: Multiple Photos + Masks                                 │
│ • View 0: image_0.jpg + mask_0.png                             │
│ • View 1: image_1.jpg + mask_1.png                             │
│ • View 2: image_2.jpg + mask_2.png                             │
│ ... (up to 8+ views)                                            │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 1: Shape Generation (50 steps)                           │
│ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  │
│                                                                 │
│ For each view:                                                  │
│   Process through shape generator → Get sparse voxel features   │
│                                                                 │
│ 🔥 FUSION: Entropy-Based Weighting                             │
│   • Analyze attention patterns from each view                   │
│   • Low entropy = focused attention = high confidence           │
│   • High entropy = scattered attention = low confidence         │
│   • Weight each view's contribution per spatial location        │
│                                                                 │
│ Output: Fused 3D shape structure (geometry skeleton)            │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 2: Texture Generation (25 steps)                         │
│ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  │
│                                                                 │
│ For each view:                                                  │
│   Process through texture generator → Get dense feature latents │
│                                                                 │
│ 🔥 FUSION: Entropy + Visibility Weighting                      │
│   • Entropy weighting (like Stage 1)                           │
│   • Visibility weighting (ray tracing for occlusions)          │
│   • Mixed mode (combine both strategies)                       │
│                                                                 │
│ Output: Fused 3D model with rich texture details               │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ DECODING: Convert to 3D Formats                                │
│ • Gaussian Splat → .ply file                                   │
│ • Triangle Mesh → .glb file                                    │
└─────────────────────────────────────────────────────────────────┘
```

### Where Fusion Happens (Critical Understanding)

**Fusion occurs in LATENT SPACE during the generative process, NOT after:**

- ❌ **NOT**: Generate 8 separate 3D meshes, then merge them
- ✅ **YES**: At each denoising step, weighted-average the latent features

This is why the approach works so well—fusion happens where the model "thinks," not where the model "outputs."

---

## Key Innovation: Multi-View Fusion

### 1. Entropy-Based Weighting

**Concept**: Use the model's internal attention patterns as confidence indicators.

**How It Works**:
```python
# Simplified conceptual flow
for each_view in views:
    attention_map = model.get_cross_attention(view)
    entropy = compute_entropy(attention_map)  # High entropy = uncertain
    confidence = 1 / (entropy + epsilon)      # Low entropy = confident
    weight[view] = softmax(confidence * alpha)  # Convert to weights

fused_latent = weighted_average(all_latents, weights)
```

**Intuition**:
- **Focused attention** (low entropy) → Model is confident → Higher weight
- **Scattered attention** (high entropy) → Model is uncertain → Lower weight

**Example**:
- View 1: Clear frontal view → Low entropy → Weight = 0.6
- View 2: Partially occluded → Medium entropy → Weight = 0.3
- View 3: Heavy occlusion → High entropy → Weight = 0.1

### 2. Visibility-Based Weighting

**Concept**: Use geometric occlusion detection to identify which parts are visible.

**How It Works**:
1. Depth Anything 3 (DA3) provides camera poses and depth maps
2. Ray tracing (DDA algorithm) detects self-occlusion
3. Visible regions get weight = 1, occluded regions get weight = 0

**Example**:
- View 1: Back of mug is self-occluded → Visibility = 0 for back surface
- View 2: Side view sees handle → Visibility = 1 for handle
- View 3: Front view misses handle → Visibility = 0 for handle

### 3. Mixed Weighting

Combines both entropy and visibility:

```
final_weight = α * entropy_weight + (1-α) * visibility_weight
```

**Use Cases**:
- **Entropy only**: No depth information available, general scenarios
- **Visibility only**: Strong geometric occlusion, reliable depth maps
- **Mixed**: Best of both worlds, requires DA3 integration

### 4. Per-Latent Spatial Weighting

**Key Innovation**: Weights are computed **per spatial location**, not per entire view.

```
Traditional (view-level):
  View 0: weight = 0.5 (entire view)
  View 1: weight = 0.3 (entire view)
  View 2: weight = 0.2 (entire view)

MV-SAM3D (per-latent):
  Location [x=10, y=5, z=3]:
    View 0: weight = 0.8 (confident here)
    View 1: weight = 0.1 (uncertain here)
    View 2: weight = 0.1 (occluded here)

  Location [x=20, y=10, z=15]:
    View 0: weight = 0.2 (occluded here)
    View 1: weight = 0.7 (visible here)
    View 2: weight = 0.1 (uncertain here)
```

This allows **spatially-adaptive fusion**: different parts of the object use different views.

---

## Technical Architecture

### Diffusion Model: Rectified Flow

MV-SAM3D uses **Rectified Flow** (Flow Matching), not standard DDPM diffusion.

#### What is Rectified Flow?

**Standard Diffusion (DDPM/Stable Diffusion 1.x)**:
- Forward: Gradually add Gaussian noise (stochastic)
- Reverse: Gradually denoise (curved trajectory)
- Equation: `x_t = √(α_t)·x_0 + √(1-α_t)·noise`

**Rectified Flow (MV-SAM3D/Stable Diffusion 3)**:
- Forward: Straight-line interpolation (deterministic)
- Reverse: Follow velocity field (straight trajectory)
- Equation: `x_t = (1-t)·noise + t·data`

#### Key Differences

| Aspect | DDPM | Rectified Flow |
|--------|------|----------------|
| Path | Curved (SDE) | Straight (ODE) |
| Training | Predict noise `ε` | Predict velocity `v` |
| Sampling | 100-1000 steps | 25-50 steps |
| Speed | Slower | Faster ⚡ |

#### Training Objective

```
Loss = MSE(predicted_velocity, true_velocity)
     = ||v_θ(x_t, t, conditions) - (x_1 - x_0)||²
```

**Not FID minimization!** FID is an evaluation metric, not the training loss.

#### Inference Process

```python
# Start from pure Gaussian noise
x_0 = torch.randn(shape)

# Integrate ODE from t=0 to t=1
for step in range(num_steps):
    t = step / num_steps
    velocity = model(x_t, t, image, mask)
    x_t = x_t + velocity * dt  # Euler integration

# x_t at t=1 is the generated latent
```

### Two-Stage Architecture

**Stage 1: Sparse Structure (SS)**
- **Purpose**: Generate 3D geometry skeleton
- **Latent**: Sparse voxel grid (~20,000 occupied voxels)
- **Steps**: 50 (default)
- **Fusion**: Entropy-weighted averaging
- **Output**: Coarse 3D shape

**Stage 2: Structured Latent (SLAT)**
- **Purpose**: Add fine details and texture
- **Latent**: Dense structured features
- **Steps**: 25 (default)
- **Fusion**: Entropy/Visibility/Mixed weighting
- **Output**: Detailed 3D model with texture

### New Modules in MV-SAM3D

| Module | Lines | Purpose |
|--------|-------|---------|
| `latent_weighting.py` | ~906 | Per-latent confidence weighting system |
| `cross_attention_logger.py` | ~412 | Capture attention maps during inference |
| `coordinate_transforms.py` | ~677 | Handle multiple coordinate systems |
| `multi_view_utils.py` | ~208 | Multi-view fusion orchestration |
| `multi_view_weighted.py` | ~1438 | Two-pass weighted generation pipeline |
| `run_inference_weighted.py` | ~3135 | Main CLI with 20+ parameters |

**Total New Code**: ~5800 lines

---

## Differences from SAM-3D-Objects

### Functional Differences

| Feature | SAM-3D-Objects | MV-SAM3D |
|---------|----------------|----------|
| **Input Views** | Single image | Multiple images (2-8+) |
| **Fusion Strategy** | N/A | Adaptive weighted fusion |
| **Weighting Methods** | N/A | Entropy, visibility, mixed |
| **Attention Analysis** | ❌ | ✅ Cross-attention logging |
| **DA3 Integration** | ❌ | ✅ Scene merging support |
| **Coordinate Systems** | Basic | Comprehensive (4 systems) |
| **CLI Parameters** | ~10 | ~30+ |
| **Computational Cost** | 1x | ~2-3x (warmup + main pass) |

### Architectural Differences

**No Additional CUDA/GPU Dependencies**:
- Uses same PyTorch, Kaolin, gsplat, flash_attn as original
- All fusion logic in pure Python/PyTorch
- No custom CUDA kernels needed

**Backward Compatible**:
- Can run in single-view mode (identical to SAM-3D-Objects)
- Same model weights and checkpoints
- Same decoder outputs (.ply, .glb)

---

## Practical Usage

### Installation

```bash
cd /content/sam3d-ready
bash setup_env_mv_sam3d.sh
```

This installs:
- Micromamba environment manager
- PyTorch 2.5.1 with CUDA 12.1
- MV-SAM3D package from `/content/MV-SAM3D`
- All dependencies (Kaolin, gsplat, flash_attn, etc.)

### Activate Environment

```bash
# Using micromamba
/content/sam3d-ready/bin/micromamba run -n sam3d-unified python --version

# Or set up shell integration
eval "$(/content/sam3d-ready/bin/micromamba shell hook -s bash)"
micromamba activate sam3d-unified
```

### Basic Multi-View Inference

```bash
python /content/MV-SAM3D/run_inference_weighted.py \
    --input_path ./data/my_object \
    --mask_prompt my_object \
    --image_names 0,1,2,3,4,5,6,7 \
    --model_tag hf \
    --seed 42
```

**Expected Directory Structure**:
```
data/my_object/
├── 0.jpg                 # View 0 image
├── 1.jpg                 # View 1 image
├── 2.jpg                 # View 2 image
├── ...
└── masks/
    └── my_object/
        ├── 0.png         # View 0 mask
        ├── 1.png         # View 1 mask
        ├── 2.png         # View 2 mask
        └── ...
```

**Output Files**:
```
output/
├── result.glb            # Main 3D mesh (textured)
├── result.ply            # Gaussian splat point cloud
├── params.npz            # Pose, scale parameters
└── inference.log         # Detailed logs
```

### Advanced Usage with DA3 Integration

```bash
# Step 1: Process images through Depth Anything 3
python /content/MV-SAM3D/scripts/run_da3.py \
    --input_path ./data/my_object \
    --output_path ./da3_outputs/my_object

# Step 2: Run MV-SAM3D with visibility weighting
python /content/MV-SAM3D/run_inference_weighted.py \
    --input_path ./data/my_object \
    --mask_prompt my_object \
    --image_names 0,1,2,3,4,5,6,7 \
    --da3_output ./da3_outputs/my_object/da3_output.npz \
    --stage2_weight_source visibility \
    --stage2_visibility_alpha 60.0 \
    --merge_da3_glb \
    --overlay_pointmap
```

**Additional Outputs with DA3**:
```
output/
├── result.glb
├── result_merged_scene.glb   # Object merged with DA3 scene
├── result_overlay.glb        # Object overlaid on View 0 point cloud
└── weights/                  # Weight visualizations (if enabled)
    ├── stage1_entropy_heatmap.png
    ├── stage2_visibility_heatmap.png
    └── weight_distribution_3d.ply
```

---

## Understanding the Parameters

### Essential Parameters

#### Input Configuration

```bash
--input_path ./data/example
```
Path to directory containing images and masks.

```bash
--mask_prompt stuffed_toy
```
Mask folder name (if multiple objects have different mask folders).

```bash
--image_names 0,1,2,3,4,5,6,7
```
Which views to use (comma-separated, 0-indexed).

#### Model Configuration

```bash
--model_tag hf
```
Which checkpoint to use (usually `hf` for HuggingFace models).

```bash
--seed 42
```
Random seed for reproducibility.

### Stage 1 (Shape) Parameters

#### Enable/Disable Weighting

```bash
--no_stage1_weighting
```
Disable entropy weighting, use simple averaging instead.

**When to use**: Testing, debugging, or if weighting causes artifacts.

#### Entropy Layer

```bash
--stage1_entropy_layer 9
```
Which transformer layer to extract attention from (default: 9).

**Tuning**: Layers 6-12 usually work best. Layer 9 shows highest cross-view entropy differences.

#### Entropy Temperature

```bash
--stage1_entropy_alpha 30.0
```
Controls weight sharpness (Gibbs temperature).

**Effect**:
- `alpha = 10`: Very smooth, almost uniform (conservative)
- `alpha = 30`: Moderate contrast (default, recommended)
- `alpha = 60`: Sharp, winner-take-all (aggressive)
- `alpha = 100`: Extreme selectivity (use with caution)

**Tuning Guide**:
- Start with 30.0
- If results are blurry/averaged → Increase to 50-60
- If results have artifacts/holes → Decrease to 15-20

### Stage 2 (Texture) Parameters

#### Enable/Disable Weighting

```bash
--no_stage2_weighting
```
Disable all weighting for Stage 2.

#### Weight Source

```bash
--stage2_weight_source entropy       # Default, no DA3 needed
--stage2_weight_source visibility    # Requires DA3
--stage2_weight_source mixed         # Requires DA3
```

**Choosing the Right Source**:

| Scenario | Recommended |
|----------|-------------|
| No depth information | `entropy` |
| Heavy occlusions | `visibility` |
| Complex scene | `mixed` |
| General use | `entropy` (default) |

#### Entropy/Visibility Temperature

```bash
--stage2_entropy_alpha 30.0
--stage2_visibility_alpha 30.0
```
Same as Stage 1, but for texture.

#### Mixed Mode Configuration

```bash
--stage2_weight_combine_mode average      # or multiply
--stage2_visibility_weight_ratio 0.5      # 0.0 = entropy only, 1.0 = visibility only
```

**Example Combinations**:
- `ratio=0.5`: Equal contribution from entropy and visibility
- `ratio=0.7`: Favor visibility 70%, entropy 30%
- `ratio=0.3`: Favor entropy 70%, visibility 30%

#### Self-Occlusion Tolerance

```bash
--self_occlusion_tolerance 4.0
```
Voxel-unit tolerance for occlusion detection.

**Effect**:
- Higher values: More lenient, fewer false occlusions
- Lower values: Stricter detection

### Visualization Parameters

```bash
--visualize_weights
```
Save weight heatmaps and 3D visualizations.

**Output**: `weights/` folder with PNG heatmaps and PLY files.

```bash
--compute_latent_visibility
```
Visualize per-view visibility (green=visible, red=occluded).

```bash
--save_attention
```
Save all attention weights for analysis.

```bash
--attention_layers 4,5,6
```
Specify which layers to save.

### DA3 Integration Parameters

```bash
--da3_output ./da3_outputs/example/da3_output.npz
```
Path to DA3 output (required for visibility weighting).

```bash
--merge_da3_glb
```
Merge SAM3D object with DA3 scene into single GLB.

```bash
--overlay_pointmap
```
Overlay result on View 0 point cloud for pose verification.

---

## Common Use Cases

### Use Case 1: Simple Multi-View Reconstruction

**Scenario**: You have 4-8 photos of an object, no special requirements.

```bash
python run_inference_weighted.py \
    --input_path ./data/toy \
    --mask_prompt toy \
    --image_names 0,1,2,3,4,5,6,7 \
    --seed 42
```

**Expected Result**: High-quality 3D model better than single-view.

### Use Case 2: Heavily Occluded Object

**Scenario**: Object has many self-occlusions (e.g., chair legs, bicycle spokes).

```bash
# Step 1: Run DA3 for depth and pose
python scripts/run_da3.py \
    --input_path ./data/chair \
    --output_path ./da3_outputs/chair

# Step 2: Use visibility weighting
python run_inference_weighted.py \
    --input_path ./data/chair \
    --mask_prompt chair \
    --image_names 0,1,2,3,4,5,6,7 \
    --da3_output ./da3_outputs/chair/da3_output.npz \
    --stage2_weight_source visibility \
    --stage2_visibility_alpha 60.0
```

**Expected Result**: Better handling of occluded regions, cleaner geometry.

### Use Case 3: Scene + Object Integration

**Scenario**: Reconstruct object and place it in the scene context.

```bash
python run_inference_weighted.py \
    --input_path ./data/room \
    --mask_prompt lamp \
    --image_names 0,1,2,3,4 \
    --da3_output ./da3_outputs/room/da3_output.npz \
    --merge_da3_glb \
    --overlay_pointmap \
    --stage2_weight_source mixed
```

**Output**: `result_merged_scene.glb` with object integrated into scene.

### Use Case 4: Debugging/Analysis

**Scenario**: Results look wrong, need to understand what's happening.

```bash
python run_inference_weighted.py \
    --input_path ./data/debug \
    --mask_prompt object \
    --image_names 0,1,2 \
    --visualize_weights \
    --save_attention \
    --attention_layers 6,9,12 \
    --compute_latent_visibility \
    --stage1_entropy_alpha 30.0 \
    --stage2_entropy_alpha 30.0
```

**Output**: Extensive visualizations in `weights/` folder showing:
- Per-view entropy distributions
- Weight heatmaps for each view
- 3D weight visualization on latent grid
- Visibility maps (if DA3 used)

### Use Case 5: Conservative vs. Aggressive Fusion

**Conservative** (when views are very different):
```bash
--stage1_entropy_alpha 15.0 \
--stage2_entropy_alpha 15.0
```
Result: Smoother blending, less artifacts, may be slightly blurry.

**Aggressive** (when one view is clearly best):
```bash
--stage1_entropy_alpha 80.0 \
--stage2_entropy_alpha 80.0
```
Result: Sharp selection, winner-take-all behavior, risk of artifacts.

**Balanced** (recommended starting point):
```bash
--stage1_entropy_alpha 30.0 \
--stage2_entropy_alpha 30.0
```

---

## Performance Considerations

### Computational Cost

**Single-View (SAM-3D-Objects)**:
- Stage 1: 50 steps × 1 view = 50 forward passes
- Stage 2: 25 steps × 1 view = 25 forward passes
- **Total**: ~75 forward passes

**Multi-View (MV-SAM3D)** with 4 views:
- Warmup pass: (50 + 25) × 4 = 300 forward passes
- Main pass: (50 + 25) × 4 = 300 forward passes
- **Total**: ~600 forward passes

**Speedup Factor**: ~8x slower for 4 views (2x for warmup, 4x for views).

### Memory Requirements

**GPU Memory Scaling**:
- Single view: ~8-12 GB VRAM
- 4 views: ~16-24 GB VRAM
- 8 views: ~32-40 GB VRAM

**Recommendations**:
- RTX 3090 / A100 (24GB): Up to 6 views
- RTX 4090 (24GB): Up to 6 views
- H100 (80GB): Up to 12+ views

**Memory Optimization**:
- Reduce `--stage1_steps` and `--stage2_steps`
- Use fewer views (3-4 often sufficient)
- Process views in batches (requires code modification)

### Quality vs. Speed Trade-offs

**Fast** (lower quality):
```bash
--stage1_steps 25 \
--stage2_steps 15 \
--image_names 0,1,2
```
**Time**: ~5-10 minutes on A100
**Quality**: Good for preview/testing

**Balanced** (recommended):
```bash
--stage1_steps 50 \
--stage2_steps 25 \
--image_names 0,1,2,3,4
```
**Time**: ~15-20 minutes on A100
**Quality**: High quality, good balance

**Maximum** (best quality):
```bash
--stage1_steps 100 \
--stage2_steps 50 \
--image_names 0,1,2,3,4,5,6,7
```
**Time**: ~40-60 minutes on A100
**Quality**: Best possible, diminishing returns

---

## Troubleshooting

### Problem: Out of Memory (OOM)

**Symptoms**: CUDA out of memory error during inference.

**Solutions**:
1. Reduce number of views:
   ```bash
   --image_names 0,1,2,3  # Instead of 0-7
   ```

2. Reduce inference steps:
   ```bash
   --stage1_steps 25 \
   --stage2_steps 15
   ```

3. Use smaller batch size (requires code modification)

4. Use gradient checkpointing (requires code modification)

### Problem: Results Look Blurry/Averaged

**Symptoms**: 3D model lacks sharp details, looks like average of all views.

**Cause**: Alpha values too low, weights are too uniform.

**Solution**: Increase alpha to make weighting more selective:
```bash
--stage1_entropy_alpha 60.0 \
--stage2_entropy_alpha 60.0
```

### Problem: Results Have Holes/Artifacts

**Symptoms**: Missing geometry, disconnected parts, strange artifacts.

**Cause**: Alpha values too high, over-aggressive weighting.

**Solution**: Decrease alpha for smoother fusion:
```bash
--stage1_entropy_alpha 15.0 \
--stage2_entropy_alpha 15.0
```

### Problem: Misaligned Views

**Symptoms**: Final model looks distorted or impossible.

**Cause**: Camera poses are inconsistent between views.

**Solution**:
1. Use DA3 to estimate poses automatically:
   ```bash
   python scripts/run_da3.py --input_path ./data/object
   ```

2. Check pose visualization:
   ```bash
   --overlay_pointmap
   ```

3. Manually verify images are of the same object

### Problem: Some Views Ignored

**Symptoms**: Certain views don't seem to contribute to final result.

**Cause**: Those views have very high entropy (model is uncertain).

**Diagnosis**: Enable visualization:
```bash
--visualize_weights
```
Check `weights/entropy_per_view.png` to see which views have high entropy.

**Solutions**:
1. If view is actually bad quality → Remove it from `--image_names`
2. If view is good quality → Decrease alpha to be more inclusive

### Problem: Texture Misalignment

**Symptoms**: Texture looks correct but offset from geometry.

**Cause**: Stage 2 weighting strategy mismatch.

**Solutions**:
1. Try different weight sources:
   ```bash
   --stage2_weight_source entropy     # Try this first
   --stage2_weight_source visibility  # If you have DA3
   --stage2_weight_source mixed       # Combination
   ```

2. Adjust visibility tolerance:
   ```bash
   --self_occlusion_tolerance 8.0  # More lenient
   ```

### Problem: Slow Inference

**Symptoms**: Taking > 1 hour for 4 views.

**Diagnostics**:
1. Check GPU utilization: `nvidia-smi`
2. Check if using CPU fallback: Look for warnings in logs

**Solutions**:
1. Ensure CUDA is properly installed
2. Verify flash attention is working:
   ```bash
   python -c "import flash_attn; print(flash_attn.__version__)"
   ```
3. Disable attention logging if not needed:
   ```bash
   # Remove --save_attention flag
   ```

---

## Advanced Topics

### Custom Weighting Strategies

The `latent_weighting.py` module is designed for extensibility. You can add new confidence factors by extending the `ConfidenceFactors` dataclass and implementing custom computation logic.

### Coordinate System Reference

MV-SAM3D handles 4 coordinate systems:

1. **SAM3D Canonical (Z-up)**: GLB output format
2. **PyTorch3D Camera (Y-up)**: SAM3D pose parameters
3. **OpenCV Camera (Y-down)**: DA3 pointmaps and extrinsics
4. **glTF Space (Y-up, Z-out)**: Final visualization format

Transforms are in `sam3d_objects/utils/coordinate_transforms.py`.

### Two-Pass Strategy Details

**Why Two Passes?**
- Pass 1 (Warmup): Simple averaging to collect attention maps
- Pass 2 (Main): Full generation with computed weights

**Benefit**: All denoising steps benefit from optimal weighting, not just later steps.

**Cost**: 2x computational overhead.

**Alternative**: Single-pass with pre-computed weights (requires code modification).

---

## Summary: When to Use MV-SAM3D

### Use MV-SAM3D When:

✅ You have multiple photos of the same object
✅ Object has occlusions or complex geometry
✅ Single-view results are unsatisfactory
✅ You need higher quality reconstructions
✅ You have sufficient GPU memory (16GB+)
✅ You can afford 2-8x longer inference time

### Stick with SAM-3D-Objects When:

❌ You only have one photo
❌ Object is simple (e.g., box, sphere)
❌ Speed is critical
❌ Limited GPU memory (< 12GB)
❌ Single-view results are already good

---

## Resources

### Code Locations

- **MV-SAM3D Source**: `/content/MV-SAM3D`
- **Original SAM-3D-Objects**: `/content/sam-3d-objects`
- **Setup Scripts**: `/content/sam3d-ready`

### Key Files

- Main inference script: `MV-SAM3D/run_inference_weighted.py`
- Weighting logic: `MV-SAM3D/sam3d_objects/utils/latent_weighting.py`
- Fusion pipeline: `MV-SAM3D/sam3d_objects/pipeline/multi_view_weighted.py`
- DA3 integration: `MV-SAM3D/scripts/run_da3.py`

### Documentation

- Parameter reference: `MV-SAM3D/README_PARAMETERS.md`
- Original SAM3D paper: https://arxiv.org/abs/2511.16624
- Setup guide: `/content/sam3d-ready/README.md`

---

## Glossary

**SS (Sparse Structure)**: Coarse 3D geometry representation using sparse voxels

**SLAT (Structured Latent)**: Dense feature representation for high-quality output

**Rectified Flow**: ODE-based generative model with straight-line paths

**Entropy**: Measure of uncertainty in attention patterns (high = uncertain)

**Visibility**: Binary indicator of whether a surface point is visible from a view

**Per-Latent**: Operating at individual spatial location granularity

**DA3 (Depth Anything 3)**: Monocular depth estimation model for camera poses

**DDA**: Digital Differential Analyzer (ray tracing algorithm)

**CFG (Classifier-Free Guidance)**: Technique to improve generation quality

**ODE**: Ordinary Differential Equation (deterministic flow)

**SDE**: Stochastic Differential Equation (random walk)

---

## FAQ

**Q: Do I need Depth Anything 3 (DA3)?**
A: No, it's optional. Entropy-based weighting works without DA3. DA3 is only needed for visibility-based weighting.

**Q: How many views should I use?**
A: 3-5 views usually sufficient. More views = better quality but slower inference.

**Q: Can I use it with video frames?**
A: Yes, extract frames and treat them as multi-view images. Be careful with motion blur.

**Q: Does it work with different camera intrinsics?**
A: Yes, but DA3 integration helps if cameras differ significantly.

**Q: Can I train my own model?**
A: Training code not included. MV-SAM3D works with pre-trained SAM3D checkpoints.

**Q: What if views have different lighting?**
A: Model is somewhat robust to lighting changes. Extreme differences may cause issues.

**Q: How do I choose alpha values?**
A: Start with 30.0. Increase for sharper weighting, decrease for smoother blending.

**Q: Can I process multiple objects in one scene?**
A: Yes, provide separate masks for each object. Process them sequentially.

---

**Document Version**: 1.0
**Last Updated**: 2026-01-13
**Author**: Comprehensive analysis of MV-SAM3D codebase and architecture
