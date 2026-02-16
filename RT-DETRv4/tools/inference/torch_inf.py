"""
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.

Enhanced Inference Script for Multi-Object Detection
Supports:
- Multi-object detection within same class
- Multi-class detection
- Optional NMS filtering
- Enhanced visualization
- Statistics output
"""

import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.ops as ops

import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from collections import defaultdict

import sys
import os
import cv2  # Added for video processing

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from engine.core import YAMLConfig


# Color palette for different classes
COLORS = [
    (255, 0, 0),      # Red
    (0, 255, 0),      # Green
    (0, 0, 255),      # Blue
    (255, 255, 0),    # Yellow
    (255, 0, 255),    # Magenta
    (0, 255, 255),    # Cyan
    (128, 0, 128),    # Purple
    (255, 165, 0),    # Orange
]


def apply_nms(boxes, scores, labels, iou_threshold=0.5, score_threshold=0.4):
    """
    Apply Non-Maximum Suppression to filter overlapping detections.
    
    Args:
        boxes: Tensor of shape [N, 4] in xyxy format
        scores: Tensor of shape [N] with confidence scores
        labels: Tensor of shape [N] with class labels
        iou_threshold: IoU threshold for NMS
        score_threshold: Minimum confidence score to keep
    
    Returns:
        Filtered boxes, scores, and labels
    """
    if len(boxes) == 0:
        return boxes, scores, labels
    
    # Filter by score threshold first
    mask = scores > score_threshold
    if not mask.any():
        return boxes[0:0], scores[0:0], labels[0:0]
    
    boxes = boxes[mask]
    scores = scores[mask]
    labels = labels[mask]
    
    # Apply NMS per class
    keep_indices = ops.batched_nms(boxes, scores, labels, iou_threshold)
    
    return boxes[keep_indices], scores[keep_indices], labels[keep_indices]


def draw(images, labels, boxes, scores, thrh=0.4, label_names=None, 
         apply_nms_flag=False, nms_iou=0.5, output_path='torch_results.jpg'):
    """
    Draw bounding boxes on images with enhanced visualization.
    
    Args:
        images: List of PIL Images
        labels: Tensor of shape [B, N] with class labels
        boxes: Tensor of shape [B, N, 4] with bounding boxes in xyxy format
        scores: Tensor of shape [B, N] with confidence scores
        thrh: Confidence threshold
        label_names: List of class names
        apply_nms_flag: Whether to apply NMS
        nms_iou: IoU threshold for NMS
        output_path: Path to save output image
    """
    for i, im in enumerate(images):
        draw_obj = ImageDraw.Draw(im)
        
        # Try to load a font, fallback to default if not available
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        except:
            try:
                font = ImageFont.truetype("arial.ttf", 16)
            except:
                font = ImageFont.load_default()

        scr = scores[i]
        lab = labels[i]
        box = boxes[i]
        
        # Filter by confidence threshold
        mask = scr > thrh
        if not mask.any():
            print(f"[Warning] No detections above threshold {thrh} for image {i}")
            im.save(output_path)
            return
        
        scr_filtered = scr[mask]
        lab_filtered = lab[mask]
        box_filtered = box[mask]
        
        # Apply NMS if requested
        if apply_nms_flag:
            box_filtered, scr_filtered, lab_filtered = apply_nms(
                box_filtered, scr_filtered, lab_filtered, 
                iou_threshold=nms_iou, score_threshold=thrh
            )
        
        # Print statistics
        print(f"\n[Detection Statistics for Image {i+1}]")
        print(f"Total detections: {len(box_filtered)}")
        
        # Count per class
        class_counts = defaultdict(int)
        for label_id in lab_filtered:
            class_id = int(label_id.item())
            class_counts[class_id] += 1
        
        for class_id, count in sorted(class_counts.items()):
            class_name = label_names[class_id] if label_names and class_id < len(label_names) else f"Class {class_id}"
            print(f"  {class_name}: {count} objects")
        
        # Draw bounding boxes
        for j, b in enumerate(box_filtered):
            label_id = int(lab_filtered[j].item())
            score_val = float(scr_filtered[j].item())
            
            # Get color for this class
            color = COLORS[label_id % len(COLORS)]
            
            # Draw rectangle
            draw_obj.rectangle(list(b), outline=color, width=3)
            
            # Prepare label text
            label_text = label_names[label_id] if label_names and label_id < len(label_names) else f"Class {label_id}"
            text = f"{label_text} {score_val:.2f}"
            
            # Draw text background
            text_bbox = draw_obj.textbbox((b[0], b[1]), text, font=font)
            text_bg = [text_bbox[0] - 2, text_bbox[1] - 2, text_bbox[2] + 2, text_bbox[3] + 2]
            draw_obj.rectangle(text_bg, fill=color)
            
            # Draw text
            draw_obj.text((b[0], b[1]), text, fill=(255, 255, 255), font=font)

        im.save(output_path)
        print(f"[Saved] Output image: {output_path}")


def process_image(model, device, file_path, label_names=None, text_prompts=None,
                  confidence_threshold=0.4, apply_nms=False, nms_iou=0.5, output_path=None):
    """
    Process a single image for multi-object detection.
    
    Args:
        model: Deployed model
        device: Device to run inference on
        file_path: Path to input image
        label_names: List of class names
        text_prompts: Text prompts for open-vocabulary detection
        confidence_threshold: Minimum confidence score to keep
        apply_nms: Whether to apply Non-Maximum Suppression
        nms_iou: IoU threshold for NMS
        output_path: Path to save output image (default: torch_results.jpg)
    """
    print(f"\n[Processing Image] {file_path}")
    
    im_pil = Image.open(file_path).convert('RGB')
    w, h = im_pil.size
    orig_size = torch.tensor([[w, h]]).to(device)
    
    print(f"Original size: {w} x {h}")

    transforms = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
    ])
    im_data = transforms(im_pil).unsqueeze(0).to(device)

    print("[Running Inference]...")
    with torch.no_grad():
        output = model(im_data, orig_size, text_prompts=text_prompts)
    
    labels, boxes, scores = output
    
    if output_path is None:
        output_path = 'torch_results.jpg'
    
    draw([im_pil], labels, boxes, scores, 
         thrh=confidence_threshold, 
         label_names=label_names,
         apply_nms_flag=apply_nms,
         nms_iou=nms_iou,
         output_path=output_path)


def process_video(model, device, file_path, label_names=None, text_prompts=None,
                  confidence_threshold=0.4, apply_nms=False, nms_iou=0.5, output_path=None):
    """
    Process a video file for multi-object detection.
    
    Args:
        model: Deployed model
        device: Device to run inference on
        file_path: Path to input video
        label_names: List of class names
        text_prompts: Text prompts for open-vocabulary detection
        confidence_threshold: Minimum confidence score to keep
        apply_nms: Whether to apply Non-Maximum Suppression
        nms_iou: IoU threshold for NMS
        output_path: Path to save output video (default: torch_results.mp4)
    """
    print(f"\n[Processing Video] {file_path}")
    
    cap = cv2.VideoCapture(file_path)

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video properties: {orig_w}x{orig_h}, {fps} FPS, {total_frames} frames")

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    if output_path is None:
        output_path = 'torch_results.mp4'
    out = cv2.VideoWriter(output_path, fourcc, fps, (orig_w, orig_h))

    transforms = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
    ])

    frame_count = 0
    print("Processing video frames...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to PIL image
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        w, h = frame_pil.size
        orig_size = torch.tensor([[w, h]]).to(device)

        im_data = transforms(frame_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(im_data, orig_size, text_prompts=text_prompts)
        labels, boxes, scores = output

        # Draw detections on the frame (temporary path for each frame)
        temp_path = f'temp_frame_{frame_count}.jpg'
        draw([frame_pil], labels, boxes, scores, 
             thrh=confidence_threshold,
             label_names=label_names,
             apply_nms_flag=apply_nms,
             nms_iou=nms_iou,
             output_path=temp_path)

        # Convert back to OpenCV image
        frame_with_detections = cv2.imread(temp_path)
        if frame_with_detections is not None:
            frame_with_detections = cv2.resize(frame_with_detections, (orig_w, orig_h))
            out.write(frame_with_detections)
        
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)

        frame_count += 1

        if frame_count % 10 == 0:
            print(f"Processed {frame_count}/{total_frames} frames...")

    cap.release()
    out.release()
    print(f"Video processing complete. Result saved as '{output_path}'.")


def main(args):
    """Main function"""
    cfg = YAMLConfig(args.config, resume=args.resume)

    if 'HGNetv2' in cfg.yaml_cfg:
        cfg.yaml_cfg['HGNetv2']['pretrained'] = False

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']
    else:
        raise AttributeError('Only support resume to load model.state_dict by now.')

    # Load train mode state and convert to deploy mode
    cfg.model.load_state_dict(state)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes, text_prompts=None):
            outputs = self.model(images, text_prompts=text_prompts)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    device = args.device
    model = Model().to(device)

    # Load text prompts
    text_prompts = None
    if args.prompts:
        text_prompts = [p.strip() for p in args.prompts.split(',') if p.strip()]
    if args.prompts_file:
        with open(args.prompts_file, 'r', encoding='utf-8') as f:
            if args.prompts_file.lower().endswith('.json'):
                text_prompts = json.load(f)
            else:
                text_prompts = [line.strip() for line in f if line.strip()]

    label_names = text_prompts
    
    if label_names:
        print(f"\n[Loaded {len(label_names)} class prompts]")
        for i, name in enumerate(label_names):
            print(f"  {i}: {name}")

    # Check if the input file is an image or a video
    file_path = args.input
    if os.path.splitext(file_path)[-1].lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
        # Process as image
        process_image(
            model, device, file_path, 
            label_names=label_names, 
            text_prompts=text_prompts,
            confidence_threshold=getattr(args, 'confidence', 0.4),
            apply_nms=getattr(args, 'nms', False),
            nms_iou=getattr(args, 'nms_iou', 0.5),
            output_path=getattr(args, 'output', None)
        )
        print("Image processing complete.")
    else:
        # Process as video
        process_video(
            model, device, file_path, 
            label_names=label_names, 
            text_prompts=text_prompts,
            confidence_threshold=getattr(args, 'confidence', 0.4),
            apply_nms=getattr(args, 'nms', False),
            nms_iou=getattr(args, 'nms_iou', 0.5),
            output_path=getattr(args, 'output', None)
        )


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Multi-Object Detection Inference for RT-DETRv4',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-c', '--config', type=str, required=True,
                        help='Path to model config YAML file')
    parser.add_argument('-r', '--resume', type=str, required=True,
                        help='Path to model checkpoint (.pth file)')
    parser.add_argument('-i', '--input', type=str, required=True,
                        help='Path to input image or video file')
    parser.add_argument('-d', '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to run inference on (cpu/cuda)')
    parser.add_argument('--prompts', type=str, default=None,
                        help='Comma-separated prompt list (e.g., "bulk cargo carrier,container ship")')
    parser.add_argument('--prompts-file', type=str, default=None,
                        help='Path to txt/json file containing prompts (one per line or JSON array)')
    parser.add_argument('--confidence', type=float, default=0.4,
                        help='Confidence threshold for detections (0.0-1.0)')
    parser.add_argument('--nms', action='store_true',
                        help='Apply Non-Maximum Suppression to filter overlapping boxes')
    parser.add_argument('--nms-iou', type=float, default=0.5,
                        help='IoU threshold for NMS (0.0-1.0)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for result image/video (default: torch_results.jpg/mp4)')
    args = parser.parse_args()
    main(args)
