# Importing packages
import supervision as sv

# Load your COCO dataset annotations
dataset = sv.DetectionDataset.from_coco(
    images_directory_path="/workspace/data/input/main/images/",
    annotations_path="/workspace/data/input/main/result.json",
)

# Split the dataset into train and test sets
train_dataset, test_dataset = dataset.split(
    split_ratio=0.7, random_state=42, shuffle=True
)

# Split the dataset into train and validation sets
train_dataset, val_dataset = test_dataset.split(
    split_ratio=0.5, random_state=42, shuffle=False
)

# Output training dataset as coco
train_dataset.as_coco(
    images_directory_path="/workspace/data/input/training/images/",
    annotations_path="/workspace/data/input/training/annotations.json",
)

# Output testing dataset as coco
test_dataset.as_coco(
    images_directory_path="/workspace/data/input/testing/images/",
    annotations_path="/workspace/data/input/testing/annotations.json",
)

# Output validation dataset as coco
val_dataset.as_coco(
    images_directory_path="/workspace/data/input/validation/images/",
    annotations_path="/workspace/data/input/validation/annotations.json",
)
