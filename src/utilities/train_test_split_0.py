import supervision as sv

# Load your COCO dataset annotations
dataset = sv.DetectionDataset.from_coco(
    images_directory_path="/data/input/main-annotated/images",
    annotations_path="./data/input/main-annotated/result.json"
)

# Split the dataset into train, validation, and test sets
train_dataset, test_dataset = dataset.split(split_ratio=0.7)

train_dataset.as_coco(
    images_directory_path="./data/input/training/images/",
    annotations_path="./data/input/training/annotations.json"
)

test_dataset.as_coco(
    images_directory_path="./data/input/testing/images/",
    annotations_path="./data/input/testing/annotations.json"
)