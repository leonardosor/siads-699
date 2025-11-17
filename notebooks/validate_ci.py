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


def main():
    print("\n" + "=" * 80)
    print("DATASET VALIDATION & CONFIDENCE INTERVAL ANALYSIS")
    print("=" * 80)

    # ========================================================================
    # PART 1: Dataset Composition
    # ========================================================================
    print("\n" + "=" * 80)
    print("PART 1: DATASET COMPOSITION")
    print("=" * 80)

    for split_name in ["training", "validation"]:
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
            print(f"  ✓ Meets 100 original images requirement")
        elif len(original_imgs) > 0:
            print(f"  ⚠️  Only {len(original_imgs)} original images (target: 100)")

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

    # Calculate required sample size
    required_n = calculate_required_sample_size(
        margin_of_error=desired_margin,
        confidence_level=desired_confidence,
        expected_proportion=0.5,  # Worst case (most conservative)
    )

    print(
        f"\nTarget: {desired_confidence*100:.0f}% confidence interval with ±{desired_margin*100:.0f}% margin"
    )
    print(f"\nRequired sample size: {required_n} images")
    print(f"  (Conservative estimate assuming 50% accuracy)")

    # Calculate for different expected accuracies
    print(f"\nRequired samples at different expected accuracies:")
    for expected_acc in [0.70, 0.80, 0.90, 0.95]:
        n = calculate_required_sample_size(
            desired_margin, desired_confidence, expected_acc
        )
        print(f"  - {expected_acc*100:.0f}% accuracy: {n} samples")

    # Check current validation set
    val_dir = DATA_DIR / "validation"
    if val_dir.exists():
        original_imgs, _, _ = identify_original_images(val_dir)
        actual_n = len(original_imgs)

        print(f"\nYour validation set: {actual_n} original images")

        if actual_n >= required_n:
            print(
                f"✓ SUFFICIENT for {desired_confidence*100:.0f}% CI ± {desired_margin*100:.0f}%"
            )
        else:
            print(f"⚠️  INSUFFICIENT - need {required_n - actual_n} more images")

            # What margin can we achieve?
            z_score = stats.norm.ppf((1 + desired_confidence) / 2)
            achievable_margin = z_score * math.sqrt(0.25 / actual_n)
            print(
                f"   With {actual_n} images, you can achieve ±{achievable_margin*100:.1f}% margin"
            )

    # ========================================================================
    # PART 3: Model Performance Confidence Intervals
    # ========================================================================
    print("\n" + "=" * 80)
    print("PART 3: MODEL PERFORMANCE WITH CONFIDENCE INTERVALS")
    print("=" * 80)

    # Example: Based on your best model from notebook
    # Best model: finance-parser-20251115_234720
    # mAP50: 0.9949, Precision: 0.9991, Recall: 0.9935

    print("\nBest Model: finance-parser-20251115_234720")
    print("-" * 80)

    # Assuming validation on original images
    val_dir = DATA_DIR / "validation"
    if val_dir.exists():
        original_imgs, _, _ = identify_original_images(val_dir)
        n_val = len(original_imgs)

        print(f"\nValidation set size: {n_val} original images")

        # Reported metrics from your notebook
        metrics = {
            "mAP50": 0.9949,
            "mAP50-95": 0.9465,
            "Precision": 0.9991,
            "Recall": 0.9935,
        }

        print(
            f"\nPerformance Metrics with {desired_confidence*100:.0f}% Confidence Intervals:"
        )
        print("-" * 80)

        for metric_name, metric_value in metrics.items():
            margin, lower, upper = calculate_confidence_interval(
                n_val, metric_value, desired_confidence
            )

            print(f"\n{metric_name}:")
            print(f"  Observed value:  {metric_value:.4f} ({metric_value*100:.2f}%)")
            print(
                f"  {desired_confidence*100:.0f}% CI:         [{lower:.4f}, {upper:.4f}]"
            )
            print(f"  Margin of error: ±{margin:.4f} (±{margin*100:.2f}%)")

            if margin <= desired_margin:
                print(f"  ✓ Meets ±{desired_margin*100:.0f}% target")
            else:
                print(
                    f"  ⚠️  Exceeds ±{desired_margin*100:.0f}% target (need more validation samples)"
                )

    # ========================================================================
    # PART 4: Recommendations
    # ========================================================================
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)

    print(
        f"""
For {desired_confidence*100:.0f}% confidence interval with ±{desired_margin*100:.0f}% margin of error:

1. SAMPLE SIZE:
   - Minimum required: {required_n} original images for validation
   - Your current validation set should have at least this many

2. DATASET COMPOSITION:
   - ✓ Use 100+ original images
   - ✓ Apply consistent augmentations (5-10x per image)
   - ✓ Keep original images separate for unbiased validation

3. VALIDATION STRATEGY:
   - Validate ONLY on original images (not augmented)
   - This prevents data leakage from similar augmentations
   - Report confidence intervals with all metrics

4. STATISTICAL REPORTING:
   - Your model accuracy: {metrics['mAP50']*100:.2f}%
   - Report as: {metrics['mAP50']*100:.2f}% ± [margin]% (85% CI)
   - Based on n={n_val} independent validation images

5. CURRENT STATUS:
   """
    )

    if val_dir.exists() and len(original_imgs) >= required_n:
        print("   ✅ Your dataset meets statistical requirements!")
        print("   ✅ You can confidently report 85% CI ± 5%")
    else:
        print("   ⚠️  Consider increasing validation set size")
        print(f"   ⚠️  Target: {required_n} original validation images")

    print("\n" + "=" * 80)
    print()


if __name__ == "__main__":
    main()
