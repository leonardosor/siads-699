# Bootstrap Confidence Intervals for Model Comparison

This directory contains scripts for computing bootstrap confidence intervals when comparing object detection model performance.

## Overview

The bootstrap analysis provides statistically rigorous confidence intervals for:
- Individual model mAP50 scores
- The difference in mAP50 between models
- Statistical significance of improvements

## Files

- `bootstrap_confidence_intervals.py` - Main Python script for bootstrap analysis
- `batch_bootstrap_ci.sh` - SLURM batch job for Great Lakes
- `../notebooks/bootstrap_confidence_intervals.ipynb` - Interactive notebook version

## Quick Start

### On Great Lakes

1. **Submit a job** with your fine-tuned model:

```bash
cd /home/$USER/699/siads-699

# Submit with default settings (10,000 iterations)
FINETUNED_MODEL=models/experiments/final/yolo-final-20251123/weights/best.pt \
sbatch src/training/batch_bootstrap_ci.sh
```

2. **Monitor the job**:

```bash
squeue -u $USER
tail -f ~/bootstrap-ci-<job-id>.log
```

3. **Retrieve results**:

```bash
# Download results JSON
scp $USER@login.greatlakes.arc-ts.umich.edu:/home/$USER/699/siads-699/data/output/bootstrap/bootstrap_confidence_intervals.json ./

# Or download the entire results package
scp $USER@login.greatlakes.arc-ts.umich.edu:/home/$USER/699/siads-699/models/artifacts/bootstrap_ci_*.tar.gz ./
```

### Locally

```bash
python src/training/bootstrap_confidence_intervals.py \
    --baseline-model models/pretrained/yolov8n.pt \
    --finetuned-model models/experiments/final/best.pt \
    --data-config src/training/finance-image-parser.yaml \
    --n-bootstrap 10000 \
    --output-dir data/output/bootstrap
```

## Configuration Options

### Environment Variables (Great Lakes)

```bash
# Required
FINETUNED_MODEL=path/to/model.pt

# Optional
PROJECT_ROOT=/home/$USER/699/siads-699  # Default
BASELINE_MODEL=models/pretrained/yolov8n.pt  # Default
N_BOOTSTRAP=10000  # Default (1000-10000 recommended)
CONFIDENCE_LEVEL=0.95  # Default
RANDOM_SEED=42  # Default
OUTPUT_DIR=data/output/bootstrap  # Default
```

### Command Line Arguments (Python script)

```bash
python src/training/bootstrap_confidence_intervals.py --help

Arguments:
  --baseline-model PATH      Path to baseline model weights (required)
  --finetuned-model PATH     Path to fine-tuned model weights (required)
  --data-config PATH         Path to YOLO data YAML (required)
  --output-dir PATH          Output directory (default: data/output)
  --n-bootstrap INT          Number of iterations (default: 10000)
  --confidence-level FLOAT   Confidence level (default: 0.95)
  --random-seed INT          Random seed (default: 42)
  --device {cuda,cpu}        Device (default: auto-detect)
```

## Usage Examples

### Standard Analysis (10,000 iterations)

```bash
FINETUNED_MODEL=models/experiments/final/best.pt \
sbatch src/training/batch_bootstrap_ci.sh
```

### Quick Test (1,000 iterations)

```bash
FINETUNED_MODEL=models/experiments/final/best.pt \
N_BOOTSTRAP=1000 \
sbatch src/training/batch_bootstrap_ci.sh
```

### High-Precision Analysis (50,000 iterations)

```bash
FINETUNED_MODEL=models/experiments/final/best.pt \
N_BOOTSTRAP=50000 \
sbatch src/training/batch_bootstrap_ci.sh
```

### Custom Output Location

```bash
FINETUNED_MODEL=models/experiments/final/best.pt \
OUTPUT_DIR=data/output/bootstrap_final \
sbatch src/training/batch_bootstrap_ci.sh
```

### Compare Multiple Models

```bash
# Model 1
FINETUNED_MODEL=models/experiments/experiment1/best.pt \
OUTPUT_DIR=data/output/bootstrap_exp1 \
sbatch src/training/batch_bootstrap_ci.sh

# Model 2
FINETUNED_MODEL=models/experiments/experiment2/best.pt \
OUTPUT_DIR=data/output/bootstrap_exp2 \
sbatch src/training/batch_bootstrap_ci.sh
```

## Output Files

The script generates:

1. **bootstrap_confidence_intervals.json** - Main results file:
   ```json
   {
     "baseline": {
       "point_estimate": 0.0014,
       "bootstrap_mean": 0.0013,
       "ci_lower": 0.0000,
       "ci_upper": 0.0042,
       "std_error": 0.0011
     },
     "finetuned": {
       "point_estimate": 0.8699,
       "bootstrap_mean": 0.8705,
       "ci_lower": 0.8555,
       "ci_upper": 0.8850,
       "std_error": 0.0075
     },
     "improvement": {
       "point_estimate": 0.8685,
       "bootstrap_mean": 0.8692,
       "ci_lower": 0.8540,
       "ci_upper": 0.8840,
       "std_error": 0.0076,
       "p_value": 0.0000
     },
     "config": {
       "n_bootstrap": 10000,
       "n_images": 481,
       "confidence_level": 0.95,
       "device": "cuda"
     }
   }
   ```

2. **bootstrap_distributions.npz** - Full bootstrap distributions for plotting:
   - `baseline`: Array of baseline mAP50 values from each iteration
   - `finetuned`: Array of fine-tuned mAP50 values from each iteration
   - `improvement`: Array of improvement values from each iteration

3. **bootstrap_ci_TIMESTAMP.tar.gz** - Compressed archive of all results

## Interpreting Results

### Confidence Intervals

The 95% confidence interval tells you: "If we repeated this experiment many times with different test sets, 95% of the time the true mAP50 would fall within this range."

**Example**: Fine-tuned mAP50 = 0.8699 [95% CI: 0.8555, 0.8850]
- Point estimate: 86.99%
- We're 95% confident the true performance is between 85.55% and 88.50%

### Statistical Significance

**P-value interpretation**:
- `p < 0.05`: Strong evidence of improvement (statistically significant)
- `p < 0.01`: Very strong evidence
- `p < 0.001`: Extremely strong evidence
- `p >= 0.05`: Insufficient evidence of improvement

**Confidence interval for improvement**:
- If CI **excludes zero**: Improvement is statistically significant
- If CI **includes zero**: Improvement may not be significant
- If CI is **entirely positive**: Strong evidence of improvement

### Example Interpretations

1. **Clear Improvement**:
   ```
   Improvement: 0.8685 [95% CI: 0.8540, 0.8840]
   P-value: 0.0000
   ```
   ✓ CI excludes zero → statistically significant
   ✓ P-value < 0.001 → extremely strong evidence
   **Conclusion**: Fine-tuning significantly improved performance

2. **Marginal Improvement**:
   ```
   Improvement: 0.0250 [95% CI: -0.0050, 0.0550]
   P-value: 0.0850
   ```
   ⚠ CI includes zero → not statistically significant
   ⚠ P-value > 0.05 → insufficient evidence
   **Conclusion**: Observed improvement may be due to chance

3. **Strong but Narrow Improvement**:
   ```
   Improvement: 0.0500 [95% CI: 0.0420, 0.0580]
   P-value: 0.0001
   ```
   ✓ CI excludes zero, narrow range → precise, significant improvement
   ✓ Low p-value → strong evidence
   **Conclusion**: Fine-tuning consistently improved performance by ~5%

## Performance Notes

### Computation Time

On Great Lakes with GPU (GTX/RTX series):
- 1,000 iterations: ~2-5 minutes
- 10,000 iterations: ~15-30 minutes
- 50,000 iterations: ~1-2 hours

### GPU vs CPU

- **GPU (CUDA)**: 5-10x faster, recommended
- **CPU**: Slower but works without GPU

The script automatically uses GPU if available.

### Resource Requirements

SLURM allocations (already configured in batch script):
- GPU: 1 GPU (any available)
- CPUs: 8 cores
- Memory: 16 GB
- Time: 4 hours (sufficient for up to 50,000 iterations)

## Methodology

The bootstrap procedure:

1. **Paired Resampling**: Both models evaluated on same resampled test images
2. **With Replacement**: Images can appear multiple times in each sample
3. **mAP50 Recalculation**: Full metric computed for each iteration
4. **Percentile CI**: Uses 2.5th and 97.5th percentiles for 95% CI

This approach:
- ✓ Handles complex statistics (mAP50 is non-linear)
- ✓ Preserves correlation between models
- ✓ No distributional assumptions needed
- ✓ Directly estimates uncertainty in improvement

## Troubleshooting

### Job fails immediately

Check that model paths exist:
```bash
ls -lh models/pretrained/yolov8n.pt
ls -lh models/experiments/final/best.pt
```

### Out of memory

Reduce batch operations or request more memory:
```bash
#SBATCH --mem=32G  # Edit in batch_bootstrap_ci.sh
```

### No GPU available

Script automatically falls back to CPU, but will be slower.

### Results don't match notebook

Small differences are normal due to:
- Different random seeds
- Different image loading order
- Floating point precision

The confidence intervals should be very similar.

## For Your Thesis

### Recommended Reporting

> "Model performance was evaluated using mAP50 on a test set of 481 images. To quantify uncertainty and assess statistical significance, we computed 95% confidence intervals via paired bootstrap resampling with 10,000 iterations. For each bootstrap sample, we resampled images with replacement (maintaining pairing across models) and recomputed mAP50 using GPU-accelerated calculations.
>
> **Results:**
> - Baseline model (pretrained YOLOv8n): mAP50 = 0.14% [95% CI: 0.00%, 0.42%]
> - Fine-tuned model: mAP50 = 86.99% [95% CI: 85.55%, 88.50%]
> - Improvement: ΔmAP50 = 86.85% [95% CI: 85.40%, 88.40%], p < 0.001
>
> The 95% confidence interval for improvement excludes zero, providing strong statistical evidence (p < 0.001) that fine-tuning significantly improved detection performance. The narrow confidence interval (width ~3%) indicates high precision in our estimate despite the finite test set."

### Key Points to Include

1. **Sample size**: Number of test images
2. **Method**: Paired bootstrap resampling
3. **Iterations**: Number of bootstrap samples
4. **Results**: Point estimates with 95% CIs for both models and improvement
5. **Significance**: P-value and interpretation
6. **Conclusion**: Whether improvement is statistically significant

## References

- Efron, B., & Tibshirani, R. J. (1994). *An Introduction to the Bootstrap*. Chapman and Hall/CRC.
- Padilla, R., et al. (2020). "A Survey on Performance Metrics for Object-Detection Algorithms." *IWSSIP 2020*.
- Statistical best practices for object detection: [COCO Evaluation](https://cocodataset.org/#detection-eval)

## Questions?

For issues or questions:
1. Check the job log: `~/bootstrap-ci-<job-id>.log`
2. Verify model and data paths
3. Test locally with small N_BOOTSTRAP first
4. Review the Jupyter notebook for interactive exploration
