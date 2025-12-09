#!/usr/bin/env python3
"""
Bootstrap Confidence Intervals for mAP50 Model Comparison

This script computes bootstrap confidence intervals for comparing
object detection model performance using paired resampling.

Based on: notebooks/bootstrap_confidence_intervals.ipynb
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO


def setup_device():
    """Setup and verify GPU device."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    return device


def load_ground_truth(data_yaml: str, split: str = 'test') -> Dict:
    """Load ground truth labels for test set."""
    import yaml

    with open(data_yaml, 'r') as f:
        data_config = yaml.safe_load(f)

    # Get the root path and test path from YAML
    root_path = Path(data_config.get('path', ''))
    test_rel_path = data_config.get(split, data_config.get('val'))

    # Construct full path to images directory
    if root_path.is_absolute():
        images_path = root_path / test_rel_path
    else:
        yaml_dir = Path(data_yaml).parent.resolve()
        images_path = yaml_dir / root_path / test_rel_path

    images_path = images_path.resolve()

    # Replace 'images' with 'labels' in the path
    labels_path = Path(str(images_path).replace('images', 'labels'))

    print(f"Images path: {images_path}")
    print(f"Labels path: {labels_path}")

    if not labels_path.exists():
        if 'images' in images_path.parts:
            parts = list(images_path.parts)
            for i, part in enumerate(parts):
                if part == 'images':
                    parts[i] = 'labels'
                    break
            labels_path = Path(*parts)
            print(f"Trying alternative labels path: {labels_path}")

    if not labels_path.exists():
        raise FileNotFoundError(
            f"Labels directory not found!\n"
            f"Tried: {labels_path}\n"
            f"Please verify your dataset structure."
        )

    # Load all label files
    ground_truth = {}
    label_files = list(labels_path.glob('*.txt'))

    print(f"Found {len(label_files)} label files")

    if len(label_files) == 0:
        raise FileNotFoundError(
            f"No .txt label files found in {labels_path}\n"
            f"Please check your dataset structure."
        )

    for label_file in label_files:
        with open(label_file, 'r') as f:
            lines = f.readlines()

        boxes = []
        classes = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls = int(parts[0])
                x, y, w, h = map(float, parts[1:5])
                boxes.append([x, y, w, h])
                classes.append(cls)

        ground_truth[label_file.stem] = {
            'boxes': np.array(boxes) if boxes else np.array([]).reshape(0, 4),
            'classes': np.array(classes) if classes else np.array([])
        }

    return ground_truth


def get_predictions_per_image(model, data_yaml: str, split: str = 'test') -> List[Dict]:
    """Run inference and collect per-image predictions."""
    import yaml

    with open(data_yaml, 'r') as f:
        data_config = yaml.safe_load(f)

    # Get test path
    root_path = Path(data_config.get('path', ''))
    test_rel_path = data_config.get(split, data_config.get('val'))

    if root_path.is_absolute():
        test_path = root_path / test_rel_path
    else:
        yaml_dir = Path(data_yaml).parent.resolve()
        test_path = yaml_dir / root_path / test_rel_path

    # Run predictions
    results_list = model.predict(
        source=str(test_path),
        imgsz=640,
        conf=0.001,
        iou=0.6,
        verbose=False
    )

    predictions = []
    for r in results_list:
        boxes = r.boxes.xyxy.cpu().numpy() if len(r.boxes) > 0 else np.array([]).reshape(0, 4)
        scores = r.boxes.conf.cpu().numpy() if len(r.boxes) > 0 else np.array([])
        classes = r.boxes.cls.cpu().numpy() if len(r.boxes) > 0 else np.array([])

        predictions.append({
            'boxes': boxes,
            'scores': scores,
            'classes': classes,
            'path': r.path
        })

    return predictions


def preload_image_dimensions(predictions: List[Dict]) -> Dict[str, Tuple[int, int]]:
    """Pre-load all image dimensions to avoid I/O during bootstrap."""
    img_dims = {}
    print("Pre-loading image dimensions...")
    for pred in tqdm(predictions):
        img_path = pred['path']
        if img_path not in img_dims:
            try:
                img = Image.open(img_path)
                img_dims[img_path] = img.size  # (width, height)
            except Exception as e:
                print(f"Warning: Could not load {img_path}: {e}")
                img_dims[img_path] = (640, 640)

    return img_dims


def compute_iou_xyxy_gpu(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Compute IoU between two sets of boxes using GPU."""
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    union = area1[:, None] + area2 - inter
    iou = inter / union.clamp(min=1e-6)

    return iou


def compute_ap_per_class(tp, conf, pred_cls, target_cls, eps=1e-16):
    """Compute Average Precision per class."""
    i = np.argsort(-conf)
    tp = tp[i]
    conf = conf[i]
    pred_cls = pred_cls[i]

    unique_classes = np.unique(target_cls)
    n_classes = unique_classes.shape[0]

    ap = np.zeros(n_classes)

    for ci, c in enumerate(unique_classes):
        i_class = pred_cls == c
        n_gt = (target_cls == c).sum()
        n_p = i_class.sum()

        if n_p == 0 or n_gt == 0:
            continue

        tp_class = tp[i_class]
        fp_class = 1 - tp_class

        tp_cumsum = np.cumsum(tp_class)
        fp_cumsum = np.cumsum(fp_class)

        recall = tp_cumsum / (n_gt + eps)
        precision = tp_cumsum / (tp_cumsum + fp_cumsum + eps)

        ap[ci] = compute_ap_from_pr(recall, precision)

    return ap


def compute_ap_from_pr(recall, precision):
    """Compute Average Precision from precision-recall curve."""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    x = np.linspace(0, 1, 101)
    ap = np.trapz(np.interp(x, mrec, mpre), x)

    return ap


def compute_map50_resampled_gpu(predictions, ground_truth, indices, img_dims, device='cuda'):
    """GPU-accelerated mAP50 computation on resampled predictions."""
    iou_threshold = 0.5

    all_tp = []
    all_conf = []
    all_pred_cls = []
    all_target_cls = []

    for idx in indices:
        pred = predictions[idx]
        img_path = Path(pred['path'])
        img_stem = img_path.stem

        if img_stem not in ground_truth:
            if len(pred['scores']) > 0:
                all_tp.extend([0] * len(pred['scores']))
                all_conf.extend(pred['scores'])
                all_pred_cls.extend(pred['classes'])
            continue

        gt = ground_truth[img_stem]
        gt_boxes = gt['boxes']
        gt_classes = gt['classes']

        if len(gt_boxes) == 0:
            if len(pred['scores']) > 0:
                all_tp.extend([0] * len(pred['scores']))
                all_conf.extend(pred['scores'])
                all_pred_cls.extend(pred['classes'])
            continue

        pred_boxes = pred['boxes']
        pred_scores = pred['scores']
        pred_classes = pred['classes']

        if len(pred_boxes) == 0:
            all_target_cls.extend(gt_classes)
            continue

        img_width, img_height = img_dims.get(str(img_path), (640, 640))

        gt_boxes_xywh = torch.tensor(gt_boxes, dtype=torch.float32, device=device)
        gt_classes_t = torch.tensor(gt_classes, dtype=torch.long, device=device)
        pred_boxes_t = torch.tensor(pred_boxes, dtype=torch.float32, device=device)
        pred_classes_t = torch.tensor(pred_classes, dtype=torch.long, device=device)

        pred_boxes_norm = pred_boxes_t.clone()
        pred_boxes_norm[:, [0, 2]] /= img_width
        pred_boxes_norm[:, [1, 3]] /= img_height

        gt_boxes_xyxy = torch.zeros_like(gt_boxes_xywh)
        gt_boxes_xyxy[:, 0] = gt_boxes_xywh[:, 0] - gt_boxes_xywh[:, 2] / 2
        gt_boxes_xyxy[:, 1] = gt_boxes_xywh[:, 1] - gt_boxes_xywh[:, 3] / 2
        gt_boxes_xyxy[:, 2] = gt_boxes_xywh[:, 0] + gt_boxes_xywh[:, 2] / 2
        gt_boxes_xyxy[:, 3] = gt_boxes_xywh[:, 1] + gt_boxes_xywh[:, 3] / 2

        iou_matrix = compute_iou_xyxy_gpu(pred_boxes_norm, gt_boxes_xyxy)

        gt_matched = torch.zeros(len(gt_boxes), dtype=torch.bool, device=device)

        for pi in range(len(pred_boxes)):
            pred_cls = pred_classes_t[pi].item()

            valid_gt = gt_classes_t == pred_cls
            valid_gt = valid_gt & ~gt_matched

            if not valid_gt.any():
                all_tp.append(0)
            else:
                ious = iou_matrix[pi]
                ious[~valid_gt] = 0

                best_iou, best_gt_idx = ious.max(dim=0)

                if best_iou >= iou_threshold:
                    gt_matched[best_gt_idx] = True
                    all_tp.append(1)
                else:
                    all_tp.append(0)

            all_conf.append(pred_scores[pi])
            all_pred_cls.append(pred_cls)

        all_target_cls.extend(gt_classes.tolist())

    if len(all_tp) == 0 or len(all_target_cls) == 0:
        return 0.0

    all_tp = np.array(all_tp)
    all_conf = np.array(all_conf)
    all_pred_cls = np.array(all_pred_cls)
    all_target_cls = np.array(all_target_cls)

    ap_per_cls = compute_ap_per_class(all_tp, all_conf, all_pred_cls, all_target_cls)
    map50 = ap_per_cls.mean() if len(ap_per_cls) > 0 else 0.0

    return float(map50)


def bootstrap_map50_gpu(
    baseline_predictions,
    finetuned_predictions,
    ground_truth,
    baseline_map50,
    finetuned_map50,
    n_bootstrap=10000,
    confidence_level=0.95,
    random_seed=42,
    device=None
):
    """GPU-accelerated bootstrap confidence intervals for mAP50."""
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if not torch.cuda.is_available() and device == 'cuda':
        print("WARNING: CUDA not available, falling back to CPU")
        device = 'cpu'

    print(f"\nUsing device: {device.upper()}")

    print("\nPreparing data for GPU acceleration...")
    img_dims_baseline = preload_image_dimensions(baseline_predictions)
    img_dims_finetuned = preload_image_dimensions(finetuned_predictions)

    n_images = len(baseline_predictions)
    image_indices = np.arange(n_images)

    baseline_map50_bootstrap = np.zeros(n_bootstrap)
    finetuned_map50_bootstrap = np.zeros(n_bootstrap)
    delta_map50_bootstrap = np.zeros(n_bootstrap)

    print(f"\nRunning {n_bootstrap} bootstrap iterations on {device.upper()}...")
    print("Using GPU-accelerated IoU calculations for maximum speed.\n")

    for b in tqdm(range(n_bootstrap)):
        bootstrap_indices = np.random.choice(image_indices, size=n_images, replace=True)

        baseline_map50_bootstrap[b] = compute_map50_resampled_gpu(
            baseline_predictions, ground_truth, bootstrap_indices, img_dims_baseline, device
        )
        finetuned_map50_bootstrap[b] = compute_map50_resampled_gpu(
            finetuned_predictions, ground_truth, bootstrap_indices, img_dims_finetuned, device
        )
        delta_map50_bootstrap[b] = finetuned_map50_bootstrap[b] - baseline_map50_bootstrap[b]

    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    results = {
        'baseline': {
            'point_estimate': baseline_map50,
            'bootstrap_mean': float(np.mean(baseline_map50_bootstrap)),
            'bootstrap_distribution': baseline_map50_bootstrap.tolist(),
            'ci_lower': float(np.percentile(baseline_map50_bootstrap, lower_percentile)),
            'ci_upper': float(np.percentile(baseline_map50_bootstrap, upper_percentile)),
            'std_error': float(np.std(baseline_map50_bootstrap))
        },
        'finetuned': {
            'point_estimate': finetuned_map50,
            'bootstrap_mean': float(np.mean(finetuned_map50_bootstrap)),
            'bootstrap_distribution': finetuned_map50_bootstrap.tolist(),
            'ci_lower': float(np.percentile(finetuned_map50_bootstrap, lower_percentile)),
            'ci_upper': float(np.percentile(finetuned_map50_bootstrap, upper_percentile)),
            'std_error': float(np.std(finetuned_map50_bootstrap))
        },
        'improvement': {
            'point_estimate': finetuned_map50 - baseline_map50,
            'bootstrap_mean': float(np.mean(delta_map50_bootstrap)),
            'bootstrap_distribution': delta_map50_bootstrap.tolist(),
            'ci_lower': float(np.percentile(delta_map50_bootstrap, lower_percentile)),
            'ci_upper': float(np.percentile(delta_map50_bootstrap, upper_percentile)),
            'std_error': float(np.std(delta_map50_bootstrap)),
            'p_value': float(np.mean(delta_map50_bootstrap <= 0))
        },
        'config': {
            'n_bootstrap': n_bootstrap,
            'n_images': n_images,
            'confidence_level': confidence_level,
            'device': device
        }
    }

    return results


def print_results(results):
    """Print formatted bootstrap confidence interval results."""
    print("\n" + "="*80)
    print("BOOTSTRAP CONFIDENCE INTERVAL RESULTS")
    print("="*80)

    print(f"\nConfiguration:")
    print(f"  Number of bootstrap iterations: {results['config']['n_bootstrap']}")
    print(f"  Number of test images: {results['config']['n_images']}")
    print(f"  Confidence level: {results['config']['confidence_level']*100}%")
    print(f"  Device: {results['config']['device']}")

    print(f"\n" + "-"*80)
    print("BASELINE MODEL")
    print("-"*80)
    b = results['baseline']
    print(f"  Point Estimate:      {b['point_estimate']:.4f} ({b['point_estimate']*100:.2f}%)")
    print(f"  Bootstrap Mean:      {b['bootstrap_mean']:.4f} ({b['bootstrap_mean']*100:.2f}%)")
    print(f"  95% CI:              [{b['ci_lower']:.4f}, {b['ci_upper']:.4f}]")
    print(f"  95% CI (percent):    [{b['ci_lower']*100:.2f}%, {b['ci_upper']*100:.2f}%]")
    print(f"  Standard Error:      {b['std_error']:.4f}")

    print(f"\n" + "-"*80)
    print("FINE-TUNED MODEL")
    print("-"*80)
    f = results['finetuned']
    print(f"  Point Estimate:      {f['point_estimate']:.4f} ({f['point_estimate']*100:.2f}%)")
    print(f"  Bootstrap Mean:      {f['bootstrap_mean']:.4f} ({f['bootstrap_mean']*100:.2f}%)")
    print(f"  95% CI:              [{f['ci_lower']:.4f}, {f['ci_upper']:.4f}]")
    print(f"  95% CI (percent):    [{f['ci_lower']*100:.2f}%, {f['ci_upper']*100:.2f}%]")
    print(f"  Standard Error:      {f['std_error']:.4f}")

    print(f"\n" + "-"*80)
    print("IMPROVEMENT (Fine-tuned - Baseline)")
    print("-"*80)
    d = results['improvement']
    print(f"  Point Estimate:      {d['point_estimate']:.4f} ({d['point_estimate']*100:.2f}%)")
    print(f"  Bootstrap Mean:      {d['bootstrap_mean']:.4f} ({d['bootstrap_mean']*100:.2f}%)")
    print(f"  95% CI:              [{d['ci_lower']:.4f}, {d['ci_upper']:.4f}]")
    print(f"  95% CI (percent):    [{d['ci_lower']*100:.2f}%, {d['ci_upper']*100:.2f}%]")
    print(f"  Standard Error:      {d['std_error']:.4f}")
    print(f"  P-value:             {d['p_value']:.4f}")

    print(f"\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)

    if d['ci_lower'] > 0:
        print(f"\n✓ The 95% confidence interval for improvement EXCLUDES zero.")
        print(f"  This provides strong evidence that fine-tuning improved performance.")
    elif d['ci_upper'] < 0:
        print(f"\n✗ The 95% confidence interval suggests fine-tuning may have decreased performance.")
    else:
        print(f"\n⚠ The 95% confidence interval INCLUDES zero.")
        print(f"  This suggests the observed improvement may not be statistically significant.")

    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(
        description='Compute bootstrap confidence intervals for mAP50 comparison'
    )
    parser.add_argument(
        '--baseline-model',
        type=str,
        required=True,
        help='Path to baseline model weights'
    )
    parser.add_argument(
        '--finetuned-model',
        type=str,
        required=True,
        help='Path to fine-tuned model weights'
    )
    parser.add_argument(
        '--data-config',
        type=str,
        required=True,
        help='Path to YOLO data configuration YAML'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/output',
        help='Directory to save results (default: data/output)'
    )
    parser.add_argument(
        '--n-bootstrap',
        type=int,
        default=10000,
        help='Number of bootstrap iterations (default: 10000)'
    )
    parser.add_argument(
        '--confidence-level',
        type=float,
        default=0.95,
        help='Confidence level (default: 0.95)'
    )
    parser.add_argument(
        '--random-seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        choices=['cuda', 'cpu', None],
        help='Device to use (default: auto-detect)'
    )

    args = parser.parse_args()

    # Setup
    print("="*80)
    print("Bootstrap Confidence Intervals for mAP50 Comparison")
    print("="*80)
    device = setup_device()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load models and get point estimates
    print("\n" + "="*80)
    print("Loading models and computing point estimates...")
    print("="*80)

    print("\nLoading baseline model...")
    baseline_model = YOLO(args.baseline_model)
    baseline_results = baseline_model.val(
        data=args.data_config,
        split='test',
        batch=16,
        imgsz=640,
        verbose=False
    )
    baseline_map50 = float(baseline_results.box.map50)

    print("Loading fine-tuned model...")
    finetuned_model = YOLO(args.finetuned_model)
    finetuned_results = finetuned_model.val(
        data=args.data_config,
        split='test',
        batch=16,
        imgsz=640,
        verbose=False
    )
    finetuned_map50 = float(finetuned_results.box.map50)

    print(f"\nObserved mAP50:")
    print(f"  Baseline:    {baseline_map50:.4f} ({baseline_map50*100:.2f}%)")
    print(f"  Fine-tuned:  {finetuned_map50:.4f} ({finetuned_map50*100:.2f}%)")
    print(f"  Improvement: {finetuned_map50 - baseline_map50:.4f} ({(finetuned_map50 - baseline_map50)*100:.2f}%)")

    # Load ground truth
    print("\n" + "="*80)
    print("Loading ground truth and predictions...")
    print("="*80)
    ground_truth = load_ground_truth(args.data_config, split='test')
    print(f"✓ Loaded ground truth for {len(ground_truth)} images")
    print(f"Total ground truth boxes: {sum(len(gt['boxes']) for gt in ground_truth.values())}")

    # Load predictions
    print("\nLoading baseline predictions...")
    baseline_preds = get_predictions_per_image(baseline_model, args.data_config, split='test')
    print(f"✓ Loaded {len(baseline_preds)} baseline predictions")

    print("Loading fine-tuned predictions...")
    finetuned_preds = get_predictions_per_image(finetuned_model, args.data_config, split='test')
    print(f"✓ Loaded {len(finetuned_preds)} fine-tuned predictions")

    # Run bootstrap
    print("\n" + "="*80)
    print("Running Bootstrap Analysis")
    print("="*80)
    results = bootstrap_map50_gpu(
        baseline_predictions=baseline_preds,
        finetuned_predictions=finetuned_preds,
        ground_truth=ground_truth,
        baseline_map50=baseline_map50,
        finetuned_map50=finetuned_map50,
        n_bootstrap=args.n_bootstrap,
        confidence_level=args.confidence_level,
        random_seed=args.random_seed,
        device=args.device
    )

    # Print results
    print_results(results)

    # Save results
    output_json = output_dir / 'bootstrap_confidence_intervals.json'

    # Remove bootstrap distributions from JSON (too large)
    results_save = {
        key: {k: v for k, v in value.items() if k != 'bootstrap_distribution'}
        if isinstance(value, dict) else value
        for key, value in results.items()
    }

    with open(output_json, 'w') as f:
        json.dump(results_save, f, indent=2)

    print(f"\nResults saved to: {output_json}")

    # Save full distributions as numpy
    output_npz = output_dir / 'bootstrap_distributions.npz'
    np.savez(
        output_npz,
        baseline=np.array(results['baseline']['bootstrap_distribution']),
        finetuned=np.array(results['finetuned']['bootstrap_distribution']),
        improvement=np.array(results['improvement']['bootstrap_distribution'])
    )

    print(f"Bootstrap distributions saved to: {output_npz}")
    print("\n" + "="*80)
    print("Bootstrap analysis complete!")
    print("="*80)


if __name__ == '__main__':
    main()
