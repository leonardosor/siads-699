import os
import sys
from pathlib import Path
import numpy as np
import torch
from ultralytics import YOLO
from tqdm import tqdm

# Add repo root to path
SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parent.parent
sys.path.append(str(REPO_ROOT))

DATA_DIR = REPO_ROOT / "data" / "input" / "ground-truth"
MODEL_PATH = REPO_ROOT / "models" / "production" / "best.pt"

def xywhn2xyxyn(x, w=640, h=640):
    """Convert normalized xywh to normalized xyxy"""
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    # x_center, y_center, width, height -> x1, y1, x2, y2
    # x1 = xc - w/2
    # y1 = yc - h/2
    # x2 = xc + w/2
    # y2 = yc + h/2
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y

def compute_iou(box1, box2):
    """
    Compute IoU between two boxes [x1, y1, x2, y2]
    """
    # Intersection coordinates
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # Intersection area
    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    # Union area
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection

    return intersection / union if union > 0 else 0.0

def calculate_image_metrics(pred_boxes, gt_boxes, iou_threshold=0.5):
    """
    Calculate Precision, Recall, F1 for a single image
    """
    if len(gt_boxes) == 0:
        return (0.0, 0.0, 0.0) if len(pred_boxes) > 0 else (1.0, 1.0, 1.0)
    
    if len(pred_boxes) == 0:
        return (0.0, 0.0, 0.0)

    # Match predictions to ground truth
    matches = []
    gt_matched = set()
    
    # Sort predictions by confidence if available (not passed here, assuming all valid)
    
    tp = 0
    fp = 0
    
    # Simple greedy matching
    for pred in pred_boxes:
        best_iou = 0
        best_gt_idx = -1
        
        for i, gt in enumerate(gt_boxes):
            if i in gt_matched:
                continue
            iou = compute_iou(pred, gt)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = i
        
        if best_iou >= iou_threshold:
            tp += 1
            gt_matched.add(best_gt_idx)
        else:
            fp += 1
            
    fn = len(gt_boxes) - len(gt_matched)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1

def main():
    print("="*80)
    print("CALCULATING DATASET VARIABILITY (OMEGA)")
    print("="*80)
    
    if not MODEL_PATH.exists():
        print(f"Error: Model not found at {MODEL_PATH}")
        return

    print(f"Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    
    image_files = sorted(list(DATA_DIR.glob("*.jpg")) + list(DATA_DIR.glob("*.png")))
    # Filter for original images only (no _aug)
    original_images = [img for img in image_files if "_aug" not in img.stem]
    
    print(f"Found {len(original_images)} original images in {DATA_DIR}")
    
    if len(original_images) == 0:
        print("No images found.")
        return

    f1_scores = []
    
    print("Running inference and calculating per-image metrics...")
    
    # Debug counters
    missing_labels = 0
    empty_labels = 0
    
    for img_path in tqdm(original_images):
        # Load GT - Check both same dir and 'labels' subdir
        label_path = img_path.with_suffix(".txt")
        if not label_path.exists():
            label_path = img_path.parent / "labels" / img_path.with_suffix(".txt").name
            
        gt_boxes = []
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        # class, x, y, w, h
                        cls = int(parts[0])
                        x, y, w, h = map(float, parts[1:5])
                        # Convert to x1, y1, x2, y2
                        x1 = x - w/2
                        y1 = y - h/2
                        x2 = x + w/2
                        y2 = y + h/2
                        gt_boxes.append([x1, y1, x2, y2])
        else:
            missing_labels += 1
            
        if len(gt_boxes) == 0 and label_path.exists():
            empty_labels += 1
        
        # Run inference
        results = model(img_path, verbose=False)
        
        pred_boxes = []
        for r in results:
            boxes = r.boxes.xyxyn.cpu().numpy() # Normalized xyxy
            for box in boxes:
                pred_boxes.append(box[:4]) # x1, y1, x2, y2
        
        # Calculate metrics
        p, r, f1 = calculate_image_metrics(pred_boxes, gt_boxes)
        f1_scores.append(f1)
        
    if missing_labels > 0:
        print(f"\nWarning: {missing_labels} images had no corresponding label file.")
    if empty_labels > 0:
        print(f"Warning: {empty_labels} label files were empty.")
        
    # Calculate Statistics
    f1_scores = np.array(f1_scores)
    mean_acc = np.mean(f1_scores)
    var_acc = np.var(f1_scores, ddof=1) # Sample variance
    
    print("\n" + "-"*80)
    print("RESULTS")
    print("-" * 80)
    print(f"Number of images: {len(f1_scores)}")
    print(f"Mean Accuracy (F1): {mean_acc:.4f}")
    print(f"Variance of Accuracy: {var_acc:.6f}")
    
    # Calculate Omega
    # Omega = (mean * (1 - mean) / variance) - 1
    if var_acc > 0:
        omega = (mean_acc * (1 - mean_acc) / var_acc) - 1
        print(f"Estimated Omega (Precision): {omega:.2f}")
        
        if omega < 10:
            variability = "High"
        elif omega < 50:
            variability = "Medium"
        else:
            variability = "Low"
            
        print(f"Variability Level: {variability}")
    else:
        print("Variance is 0. Cannot calculate Omega (Infinite precision).")
        print("This means all images have exactly the same accuracy (likely 1.0 or 0.0).")
        omega = float('inf')

    print("\nUse this Omega value in the validate_ci.py script.")

if __name__ == "__main__":
    main()
