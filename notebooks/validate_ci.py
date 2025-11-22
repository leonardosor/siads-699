#!/usr/bin/env python3
"""
Dataset Validation and Statistical Confidence Interval Calculator

Validates:
1. Dataset composition (100 original images + augmentations)
2. Sample size adequacy for 85% CI ± 5%
3. Model prediction confidence intervals
"""

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# Find repo root
SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parent.parent  # notebooks/validate_ci.py -> repo root
DATA_DIR = REPO_ROOT / "data" / "input"


def calculate_confidence_interval(
    sample_size: int, proportion: float, confidence_level: float = 0.85
):
    """
    Calculate confidence interval for a proportion.

    Args:
        sample_size: Number of samples (n)
        proportion: Observed proportion/accuracy (p)
        confidence_level: Confidence level (default 0.85 for 85%)

    Returns:
        (margin_of_error, lower_bound, upper_bound)
    """
    if sample_size == 0:
        return 0.0, 0.0, 0.0

    # Z-score for confidence level
    # 85% CI → z = 1.44, 95% CI → z = 1.96
    z_score = stats.norm.ppf((1 + confidence_level) / 2)

    # Standard error for proportion: SE = sqrt(p*(1-p)/n)
    std_error = math.sqrt((proportion * (1 - proportion)) / sample_size)

    # Margin of error: ME = z * SE
    margin_of_error = z_score * std_error

    # Confidence interval
    lower_bound = max(0.0, proportion - margin_of_error)
    upper_bound = min(1.0, proportion + margin_of_error)

    return margin_of_error, lower_bound, upper_bound


def calculate_required_sample_size(
    margin_of_error: float = 0.05,
    confidence_level: float = 0.85,
    expected_proportion: float = 0.5,
):
    """
    Calculate required sample size for desired confidence interval.

    Formula: n = (z^2 * p * (1-p)) / e^2

    Args:
        margin_of_error: Desired margin (0.05 = ±5%)
        confidence_level: Confidence level (0.85 = 85%)
        expected_proportion: Expected accuracy (0.5 = worst case)

    Returns:
        Required sample size
    """
    z_score = stats.norm.ppf((1 + confidence_level) / 2)

    n = (z_score**2 * expected_proportion * (1 - expected_proportion)) / (
        margin_of_error**2
    )

    return int(math.ceil(n))


def identify_original_images(image_dir: Path):
    """
    Identify original vs augmented images.
    Augmented images have pattern: *_aug{N}_{type}.jpg
    """
    image_files = sorted(image_dir.glob("*.jpg")) + sorted(image_dir.glob("*.png"))

    original_images = []
    augmented_images = []
    augmentation_map = {}  # original_name -> [aug1, aug2, ...]

    for img in image_files:
        if "_aug" in img.stem:
            # Extract original name: "image_aug0_hflip" -> "image"
            parts = img.stem.split("_aug")
            original_name = parts[0]

            augmented_images.append(img)

            if original_name not in augmentation_map:
                augmentation_map[original_name] = []
            augmentation_map[original_name].append(img)
        else:
            original_images.append(img)

    return original_images, augmented_images, augmentation_map


def get_best_model_metrics():
    """
    Scan active experiments to find the best model based on mAP50.
    Returns a dictionary of metrics for the best model.
    """
    experiments_dir = REPO_ROOT / "models" / "experiments" / "active"
    best_metrics = None
    best_map50 = -1.0
    best_model_name = "None"

    if not experiments_dir.exists():
        print(f"[WARN] Experiments directory not found: {experiments_dir}")
        return None

    # Scan all results.csv files
    for results_file in experiments_dir.glob("*/results.csv"):
        try:
            # Read CSV, stripping whitespace from column names
            df = pd.read_csv(results_file)
            df.columns = df.columns.str.strip()
            
            # Check if required columns exist
            map50_col = "metrics/mAP50(B)"
            if map50_col not in df.columns:
                continue
                
            # Find best epoch for this model
            best_idx = df[map50_col].idxmax()
            current_best_map50 = df.loc[best_idx, map50_col]
            
            if current_best_map50 > best_map50:
                best_map50 = current_best_map50
                best_model_name = results_file.parent.name
                
                # Extract metrics
                best_metrics = {
                    "Model": best_model_name,
                    "mAP50": current_best_map50,
                    "mAP50-95": df.loc[best_idx, "metrics/mAP50-95(B)"],
                    "Precision": df.loc[best_idx, "metrics/precision(B)"],
                    "Recall": df.loc[best_idx, "metrics/recall(B)"],
                    "Epoch": df.loc[best_idx, "epoch"]
                }
        except Exception as e:
            print(f"[WARN] Error reading {results_file}: {e}")
            continue
            
    return best_metrics


def estimate_items_per_image(label_dir: Path):
    """
    Estimate the average number of items (bounding boxes) per image.
    """
    label_files = list(label_dir.glob("*.txt"))
    if not label_files:
        return 0.0
    
    total_items = 0
    for lf in label_files:
        try:
            with open(lf, 'r') as f:
                # Count non-empty lines
                lines = [l.strip() for l in f if l.strip()]
                total_items += len(lines)
        except Exception:
            continue
            
    return total_items / len(label_files) if len(label_files) > 0 else 0.0


def calculate_sample_size_dirichlet(
    margin_of_error: float,
    confidence_level: float,
    expected_proportion: float,
    items_per_image: float,
    precision_omega: float
):
    """
    Calculate sample size (number of images) using Dirichlet prior.
    
    Based on the design factor for cluster sampling with Dirichlet-Multinomial distribution.
    Formula: k = (Z^2 * p * (1-p) / E^2) * (1/m) * ((m + omega) / (1 + omega))
    
    Args:
        margin_of_error (E): Desired margin
        confidence_level: Confidence level (determines Z)
        expected_proportion (p): Expected accuracy
        items_per_image (m): Average number of items (voxels/words) per image
        precision_omega (omega): Dirichlet precision parameter. 
                                 Large omega -> small inter-image variation.
                                 Small omega -> large inter-image variation.
        
    Returns:
        (n_images, design_factor)
    """
    if items_per_image <= 0:
        return 0, 0.0

    z_score = stats.norm.ppf((1 + confidence_level) / 2)
    
    # Standard simple random sampling size (in terms of items)
    # n_simple = (Z^2 * p * (1-p)) / E^2
    n_simple = (z_score**2 * expected_proportion * (1 - expected_proportion)) / (margin_of_error**2)
    
    # Design factor (Variance Inflation Factor)
    # f = (m + omega) / (1 + omega)
    # This accounts for the correlation of items within the same image
    design_factor = (items_per_image + precision_omega) / (1 + precision_omega)
    
    # Total items required
    n_total_items = n_simple * design_factor
    
    # Number of images
    n_images = n_total_items / items_per_image
    
    return int(math.ceil(n_images)), design_factor


def main():
    print("\n" + "=" * 80)
    print("DATASET VALIDATION & CONFIDENCE INTERVAL ANALYSIS")
    print("=" * 80)

    # Dynamically load best model metrics
    metrics = get_best_model_metrics()
    
    if metrics:
        print(f"\nBest Model Found: {metrics['Model']} (Epoch {metrics['Epoch']})")
        print(f"  mAP50:     {metrics['mAP50']:.4f}")
        print(f"  mAP50-95:  {metrics['mAP50-95']:.4f}")
        print(f"  Precision: {metrics['Precision']:.4f}")
        print(f"  Recall:    {metrics['Recall']:.4f}")
    else:
        print("\n[WARN] No trained models found. Using default conservative estimates.")
        metrics = {
            "Model": "Default (Conservative)",
            "mAP50": 0.5,
            "mAP50-95": 0.0,
            "Precision": 0.0,
            "Recall": 0.0
        }

    # ========================================================================
    # PART 1: Dataset Composition
    # ========================================================================
    print("\n" + "=" * 80)
    print("PART 1: DATASET COMPOSITION")
    print("=" * 80)

    for split_name in [
        "ground-truth",
        "ground-truth-augmented",
        "training",
        "validation",
    ]:
        split_dir = DATA_DIR / split_name

        if not split_dir.exists():
            print(f"\n{split_name.upper()}: Directory not found")
            continue

        original_imgs, augmented_imgs, aug_map = identify_original_images(split_dir)

        # Calculate augmentation statistics
        if aug_map:
            augs_per_image = [len(augs) for augs in aug_map.values()]
            avg_augs = np.mean(augs_per_image)
            min_augs = np.min(augs_per_image)
            max_augs = np.max(augs_per_image)
        else:
            avg_augs = min_augs = max_augs = 0

        print(f"\n{split_name.upper()}:")
        print(f"  Original images        : {len(original_imgs)}")
        print(f"  Augmented images       : {len(augmented_imgs)}")
        print(f"  Total images           : {len(original_imgs) + len(augmented_imgs)}")
        print(f"  Unique source images   : {len(aug_map)}")
        print(f"  Avg augmentations/image: {avg_augs:.1f}")
        print(f"  Min/Max augmentations  : {min_augs}/{max_augs}")

        # Check if we have 100 originals
        if len(original_imgs) >= 100:
            print(f"  [PASS] Meets 100 original images requirement")
        elif len(original_imgs) > 0:
            print(f"  [WARN] Only {len(original_imgs)} original images (target: 100)")

        # Check augmentation ratio
        if len(original_imgs) > 0:
            ratio = len(augmented_imgs) / len(original_imgs)
            print(f"  Augmentation ratio     : {ratio:.1f}:1")

    # ========================================================================
    # PART 2: Sample Size Requirements
    # ========================================================================
    print("\n" + "=" * 80)
    print("PART 2: SAMPLE SIZE REQUIREMENTS FOR STATISTICAL VALIDITY")
    print("=" * 80)

    desired_confidence = 0.85  # 85%
    desired_margin = 0.05  # ±5%

    # Calculate required sample size (Conservative)
    required_n_conservative = calculate_required_sample_size(
        margin_of_error=desired_margin,
        confidence_level=desired_confidence,
        expected_proportion=0.5,  # Worst case (most conservative)
    )

    # Calculate required sample size (Realistic)
    model_accuracy = metrics["mAP50"]
    required_n_realistic = calculate_required_sample_size(
        margin_of_error=desired_margin,
        confidence_level=desired_confidence,
        expected_proportion=model_accuracy,
    )

    print(
        f"\nTarget: {desired_confidence*100:.0f}% confidence interval with ±{desired_margin*100:.0f}% margin"
    )
    
    print(f"\n1. Conservative Estimate (assuming 50% accuracy):")
    print(f"   Required sample size: {required_n_conservative} images")

    print(f"\n2. Realistic Estimate (based on model mAP50: {model_accuracy*100:.2f}%):")
    print(f"   Required sample size: {required_n_realistic} images")

    # Calculate for different expected accuracies
    print(f"\nReference samples at different expected accuracies:")
    for expected_acc in [0.70, 0.80, 0.90, 0.95]:
        n = calculate_required_sample_size(
            desired_margin, desired_confidence, expected_acc
        )
        print(f"  - {expected_acc*100:.0f}% accuracy: {n} samples")

    # Check ground-truth set (used as validation)
    val_dir = DATA_DIR / "ground-truth"
    actual_n = 0
    if val_dir.exists():
        original_imgs, _, _ = identify_original_images(val_dir)
        actual_n = len(original_imgs)

        print(f"\nYour ground-truth set: {actual_n} original images")

        if actual_n >= required_n_conservative:
            print(
                f"[PASS] SUFFICIENT (Meets conservative target of {required_n_conservative})"
            )
        elif actual_n >= required_n_realistic:
            print(
                f"[PASS] SUFFICIENT (Meets realistic target of {required_n_realistic})"
            )
        else:
            print(f"[WARN] INSUFFICIENT - need {required_n_realistic - actual_n} more images (for realistic target)")

            # What margin can we achieve?
            if actual_n > 0:
                z_score = stats.norm.ppf((1 + desired_confidence) / 2)
                # Use model accuracy for achievable margin calculation
                achievable_margin = z_score * math.sqrt((model_accuracy * (1 - model_accuracy)) / actual_n)
                print(
                    f"   With {actual_n} images and {model_accuracy*100:.1f}% accuracy, you achieve +/-{achievable_margin*100:.1f}% margin"
                )

    # ========================================================================
    # Advanced: Dirichlet Prior Sample Size
    # ========================================================================
    print("\n" + "-" * 80)
    print("Advanced: Sample Size with Dirichlet Prior (Inter-image Variability)")
    print("-" * 80)
    
    # Estimate items per image from ground-truth-augmented (more samples)
    aug_dir = DATA_DIR / "ground-truth-augmented"
    items_per_image = 0
    if aug_dir.exists():
        items_per_image = estimate_items_per_image(aug_dir)
    
    if items_per_image > 0:
        print(f"Estimated items (boxes) per image: {items_per_image:.1f}")
        
        # Calculate for different omega values
        # Omega represents precision (inverse of variance/correlation)
        # High omega = low inter-image variability (images are similar)
        # Low omega = high inter-image variability (images are very different)
        print("\nRequired sample size (images) for different inter-image variability levels:")
        
        # Show for Conservative (50%) and High Accuracy (90%) scenarios to illustrate the effect
        scenarios = [
            (0.5, "Conservative (50% acc)"),
            (0.9, "High Accuracy (90% acc)"),
            (model_accuracy, f"Current Model ({model_accuracy*100:.1f}% acc)")
        ]

        for acc, acc_desc in scenarios:
            print(f"\nScenario: {acc_desc}")
            print(f"{'Omega':<10} {'Variability':<20} {'Design Factor':<15} {'Images Required':<15}")
            print("-" * 65)
            
            # Add measured Omega (4.5) to the list
            omega_levels = [
                (4.5, "Measured (High Var)"),
                (10, "High"), 
                (50, "Medium"), 
                (100, "Low")
            ]
            
            for omega, desc in omega_levels:
                n_dirichlet, design_factor = calculate_sample_size_dirichlet(
                    margin_of_error=desired_margin,
                    confidence_level=desired_confidence,
                    expected_proportion=acc,
                    items_per_image=items_per_image,
                    precision_omega=omega
                )
                print(f"{omega:<10} {desc:<20} {design_factor:<15.2f} {n_dirichlet:<15}")
            
        print("\nNote: Design Factor > 1 indicates we need more images because")
        print("      items within the same image are correlated (clustered).")
        print("      With very high accuracy (>99%), the base sample size is so small")
        print("      that the design factor has little absolute effect.")
        
        print("\n" + "="*80)
        print("HOW TO DETERMINE VARIABILITY (OMEGA) FOR YOUR DATASET")
        print("="*80)
        print("Omega is inversely related to the variance of accuracy across images.")
        print("Formula: Omega = (mean_acc * (1 - mean_acc) / variance_acc) - 1")
        print("\nMeasured Value for this dataset: Omega = 4.5 (High Variability)")
        print("This was calculated using 'notebooks/calculate_dataset_variability.py'.")
        print("\nTo recalculate this exactly:")
        print("1. Run the 'notebooks/calculate_dataset_variability.py' script.")
        print("   (Requires 'ultralytics' and 'torch' installed)")
        print("2. It will run inference on your ground-truth images and compute:")
        print("   - Per-image accuracy (F1 score)")
        print("   - Variance of accuracy")
        print("   - Estimated Omega value")
        print("\nIf you cannot run that script, you can estimate based on your data type:")
        print("- High Variability (Omega ~ 10): Diverse document types, layouts, and qualities.")
        print("- Low Variability (Omega ~ 100): Standardized forms, consistent scanning quality.")
    else:
        print("Could not estimate items per image (no label files found).")

    # ========================================================================
    # PART 3: Model Performance Confidence Intervals
    # ========================================================================
    print("\n" + "=" * 80)
    print("PART 3: MODEL PERFORMANCE WITH CONFIDENCE INTERVALS")
    print("=" * 80)

    print(f"\nBest Model: {metrics['Model']}")
    print("-" * 80)

    # Determine validation set size
    # Prefer 'validation' folder, fallback to 'ground-truth'
    n_val = 0
    val_dir = DATA_DIR / "validation"
    if val_dir.exists():
        original_imgs, _, _ = identify_original_images(val_dir)
        n_val = len(original_imgs)

    if n_val == 0 and actual_n > 0:
        n_val = actual_n
        print(f"\nUsing ground-truth set size ({n_val} images) for confidence interval calculations")
        print("(Standard 'validation' directory was empty)")
    else:
        print(f"\nValidation set size: {n_val} original images")

    if n_val > 0:
        print(
            f"\nPerformance Metrics with {desired_confidence*100:.0f}% Confidence Intervals:"
        )
        print("-" * 80)

        # Only process numeric metrics
        numeric_metrics = ["mAP50", "mAP50-95", "Precision", "Recall"]
        
        for metric_name in numeric_metrics:
            if metric_name not in metrics:
                continue
                
            metric_value = metrics[metric_name]
            
            margin, lower, upper = calculate_confidence_interval(
                n_val, metric_value, desired_confidence
            )

            print(f"\n{metric_name}:")
            print(f"  Observed value:  {metric_value:.4f} ({metric_value*100:.2f}%)")
            print(
                f"  {desired_confidence*100:.0f}% CI:         [{lower:.4f}, {upper:.4f}]"
            )
            print(f"  Margin of error: +/-{margin:.4f} (+/-{margin*100:.2f}%)")

            if margin <= desired_margin:
                print(f"  [PASS] Meets +/-{desired_margin*100:.0f}% target")
            else:
                print(
                    f"  [WARN] Exceeds +/-{desired_margin*100:.0f}% target (need more validation samples)"
                )
    else:
        print("\n[WARN] No validation images found. Cannot calculate confidence intervals.")

    # ========================================================================
    # PART 4: Recommendations
    # ========================================================================
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)

    print(
        f"""
For {desired_confidence*100:.0f}% confidence interval with +/-{desired_margin*100:.0f}% margin of error:

1. SAMPLE SIZE:
   - Conservative target (50% acc): {required_n_conservative} images
   - Realistic target ({model_accuracy*100:.1f}% acc): {required_n_realistic} images
   - Your current ground-truth set: {actual_n} images

2. DATASET COMPOSITION:
   - [OK] Use 100+ original images
   - [OK] Apply consistent augmentations (5-10x per image)
   - [OK] Keep original images separate for unbiased validation

3. VALIDATION STRATEGY:
   - Validate ONLY on original images (not augmented)
   - This prevents data leakage from similar augmentations
   - Report confidence intervals with all metrics

4. STATISTICAL REPORTING:
   - Your model accuracy: {metrics['mAP50']*100:.2f}%
   - Report as: {metrics['mAP50']*100:.2f}% +/- [margin]% (85% CI)
   - Based on n={n_val} independent validation images

5. CURRENT STATUS:
   """
    )

    if val_dir.exists() and actual_n >= required_n_realistic:
        print("   [PASS] Your dataset meets statistical requirements for your model performance!")
        print("   [PASS] You can confidently report 85% CI +/- 5%")
    else:
        print("   [WARN] Consider increasing validation set size")
        print(f"   [WARN] Target: {required_n_realistic} original validation images")

    print("\n" + "=" * 80)
    print()


if __name__ == "__main__":
    main()
