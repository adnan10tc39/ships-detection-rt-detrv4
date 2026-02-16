# Multi-Object Detection Inference Pipeline

This document describes the complete inference pipeline for multi-object detection using RT-DETRv4 with open-vocabulary capabilities.

## Overview

The inference pipeline processes images or videos to detect multiple objects across multiple classes. It supports:
- **Multi-object detection**: Multiple instances of the same class
- **Multi-class detection**: Different classes detected simultaneously
- **Open-vocabulary**: Add/remove classes without retraining
- **Optional NMS**: Filter overlapping detections

## Pipeline Flow

See the detailed flow diagram: `../../diagrams/inference_pipeline.mmd`

### Step-by-Step Process

1. **Input Preparation**
   - Load image/video
   - Resize to 640x640
   - Convert to tensor format

2. **Text Prompt Preparation**
   - Load class names (from command line or file)
   - Encode using CLIP text encoder
   - Generate text embeddings [C, 512]

3. **Visual Feature Extraction**
   - HGNetv2 backbone extracts multi-scale features
   - Hybrid Encoder (FPN + PAN + AIFI) enhances features

4. **Object Query Generation**
   - DFINE Transformer Decoder generates 300 object queries
   - Each query produces:
     - Visual embedding [512-dim]
     - Bounding box [4-dim: cx, cy, w, h]

5. **Similarity Computation**
   - Compute cosine similarity between visual and text embeddings
   - Apply learnable logit scale
   - Generate classification scores [300, C]

6. **Class Assignment**
   - Apply sigmoid activation
   - Select top-300 detections
   - Extract query index, class index, and confidence score

7. **Bounding Box Post-Processing**
   - Convert from normalized cxcywh to absolute xyxy
   - Scale to original image size

8. **Filtering and NMS (Optional)**
   - Apply confidence threshold
   - Non-Maximum Suppression to remove overlapping boxes

9. **Output Format**
   - Dictionary with labels, boxes, and scores
   - Map class IDs to class names

10. **Visualization**
    - Draw bounding boxes with class labels
    - Save annotated image/video

## Usage

### Basic Image Inference

```bash
python tools/inference/torch_inf.py \
    -c configs/rtv4/rtv4_hgnetv2_m_ships_open_vocab.yml \
    -r outputs/checkpoint.pth \
    -i path/to/image.jpg \
    --prompts "bulk cargo carrier,container ship,fishing boat,general cargo ship,ore carrier,passenger ship"
```

### With NMS Filtering

```bash
python tools/inference/torch_inf.py \
    -c configs/rtv4/rtv4_hgnetv2_m_ships_open_vocab.yml \
    -r outputs/checkpoint.pth \
    -i path/to/image.jpg \
    --prompts "bulk cargo carrier,container ship,fishing boat" \
    --nms \
    --nms-iou 0.5 \
    --confidence 0.4
```

### Using Prompts File

Create a text file `prompts.txt`:
```
bulk cargo carrier
container ship
fishing boat
general cargo ship
ore carrier
passenger ship
```

Then run:
```bash
python tools/inference/torch_inf.py \
    -c configs/rtv4/rtv4_hgnetv2_m_ships_open_vocab.yml \
    -r outputs/checkpoint.pth \
    -i path/to/image.jpg \
    --prompts-file prompts.txt \
    --output result.jpg
```

### Video Inference

```bash
python tools/inference/torch_inf.py \
    -c configs/rtv4/rtv4_hgnetv2_m_ships_open_vocab.yml \
    -r outputs/checkpoint.pth \
    -i path/to/video.mp4 \
    --prompts-file prompts.txt \
    --nms \
    --output result.mp4
```

## Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `-c, --config` | str | Required | Path to model config YAML file |
| `-r, --resume` | str | Required | Path to model checkpoint (.pth) |
| `-i, --input` | str | Required | Path to input image or video |
| `-d, --device` | str | `cuda`/`cpu` | Device to run inference on |
| `--prompts` | str | None | Comma-separated prompt list |
| `--prompts-file` | str | None | Path to txt/json prompts file |
| `--confidence` | float | 0.4 | Confidence threshold (0.0-1.0) |
| `--nms` | flag | False | Apply Non-Maximum Suppression |
| `--nms-iou` | float | 0.5 | IoU threshold for NMS (0.0-1.0) |
| `--output` | str | None | Output path (default: torch_results.jpg/mp4) |

## Multi-Object Detection Examples

### Example 1: Multiple Objects of Same Class

**Input**: Image with 3 fishing boats

**Prompts**: `"fishing boat"`

**Output**:
- Query 5 → Class 0 (fishing boat), Score: 0.92
- Query 12 → Class 0 (fishing boat), Score: 0.88
- Query 23 → Class 0 (fishing boat), Score: 0.85

**Result**: 3 separate bounding boxes, all labeled as "fishing boat"

### Example 2: Multi-Class Detection

**Input**: Image with 2 container ships and 1 bulk cargo carrier

**Prompts**: `"bulk cargo carrier,container ship"`

**Output**:
- Query 5 → Class 1 (container ship), Score: 0.90
- Query 12 → Class 1 (container ship), Score: 0.87
- Query 23 → Class 0 (bulk cargo carrier), Score: 0.91

**Result**: 3 bounding boxes with different class labels

### Example 3: Complex Scene

**Input**: Image with multiple ships of different types

**Prompts**: `"bulk cargo carrier,container ship,fishing boat,general cargo ship,ore carrier,passenger ship"`

**Output**: Multiple detections across all classes, each with its own bounding box and confidence score.

## How It Works

### Multi-Object Detection Within Same Class

1. The model generates 300 independent object queries
2. Each query can match any class via similarity computation
3. Multiple queries can match the same class
4. Each matching query produces a separate bounding box
5. Result: Multiple detections of the same class

### Multi-Class Detection

1. Text prompts define all classes to detect
2. Similarity matrix: [300 queries × C classes]
3. Each query selects the best matching class
4. Different queries can match different classes
5. Result: Multiple classes detected simultaneously

### Post-Processing

1. **Confidence Filtering**: Remove detections below threshold
2. **NMS (Optional)**: Remove overlapping boxes of the same class
3. **Visualization**: Draw boxes with class-specific colors

## Output Format

The inference script outputs:
- **Annotated image/video**: With bounding boxes and labels
- **Console statistics**: Per-class detection counts
- **Detection data**: Labels, boxes, and scores tensors

Example console output:
```
[Detection Statistics for Image 1]
Total detections: 5
  bulk cargo carrier: 1 objects
  container ship: 2 objects
  fishing boat: 2 objects
```

## Tips

1. **Confidence Threshold**: Lower values (0.3-0.4) detect more objects but may include false positives
2. **NMS**: Use `--nms` when you have overlapping detections of the same object
3. **NMS IoU**: Lower values (0.3-0.4) are more aggressive in removing overlaps
4. **Prompts**: Use descriptive class names that match training data
5. **Device**: Use `--device cuda` for faster inference if GPU available

## Troubleshooting

- **No detections**: Lower confidence threshold or check if prompts match training classes
- **Too many detections**: Increase confidence threshold or enable NMS
- **Overlapping boxes**: Enable NMS with appropriate IoU threshold
- **Wrong classes**: Verify prompts match the classes used during training

