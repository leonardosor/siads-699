#!/bin/bash
#SBATCH --job-name=prep_augmented
#SBATCH --account=siads699f25_class
#SBATCH --partition=standard
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00  
#SBATCH --mail-user=lcedeno@umich.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/home/%u/%x-%j.log

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/home/${USER}/699/siads-699}
AUGMENTATIONS=${AUGMENTATIONS:-50}  # Number of augmentations per image

echo "Preparing augmented dataset with ${AUGMENTATIONS} augmentations per image"

# Load mamba and activate environment
set +u
module load mamba/py3.12
source /sw/pkgs/arc/mamba/py3.12/etc/profile.d/conda.sh
eval "$(conda shell.bash hook)"
conda activate capstone
set -u

cd "${PROJECT_ROOT}"

echo "Python executable: $(which python)"

# Run augmentation script
python src/utils/dataset/prepare_dataset.py augmented \
  --augmentations-per-image "${AUGMENTATIONS}" \
  --backup-existing

echo "Augmented dataset preparation complete!"
echo "Dataset statistics:"
echo "  Training images: $(find data/input/training/images -name '*.jpg' 2>/dev/null | wc -l)"
echo "  Validation images: $(find data/input/validation/images -name '*.jpg' 2>/dev/null | wc -l)"
echo "  Testing images: $(find data/input/testing/images -name '*.jpg' 2>/dev/null | wc -l)"
