#!/usr/bin/env python3

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

UM_BLUE = "#00274C"
UM_MAIZE = "#FFCB05"


def find_repo_root() -> Path:
    current_path = Path(__file__).resolve()

    if "DOCKER_ENV" in os.environ or str(current_path).startswith("/app"):
        for parent in current_path.parents:
            if parent.name == "app":
                return parent

    for parent in current_path.parents:
        if (
            (parent / "src").is_dir()
            and (parent / "data").is_dir()
            and (parent / "models").is_dir()
        ):
            return parent

    raise RuntimeError(
        "Could not find repository root. Expected to find 'src/', 'data/', and 'models/' directories."
    )


class BoundingBox:
    def __init__(
        self,
        class_id: int,
        x_center: float,
        y_center: float,
        width: float,
        height: float,
    ):
        self.class_id = class_id
        self.x_center = x_center
        self.y_center = y_center
        self.width = width
        self.height = height

    def to_corners(
        self, img_width: int, img_height: int
    ) -> Tuple[float, float, float, float]:
        x_center_px = self.x_center * img_width
        y_center_px = self.y_center * img_height
        width_px = self.width * img_width
        height_px = self.height * img_height

        x_min = x_center_px - width_px / 2
        y_min = y_center_px - height_px / 2
        x_max = x_center_px + width_px / 2
        y_max = y_center_px + height_px / 2

        return x_min, y_min, x_max, y_max

    @staticmethod
    def from_corners(
        class_id: int,
        x_min: float,
        y_min: float,
        x_max: float,
        y_max: float,
        img_width: int,
        img_height: int,
    ) -> "BoundingBox":
        x_min = max(0, min(x_min, img_width))
        y_min = max(0, min(y_min, img_height))
        x_max = max(0, min(x_max, img_width))
        y_max = max(0, min(y_max, img_height))

        width_px = x_max - x_min
        height_px = y_max - y_min
        x_center_px = x_min + width_px / 2
        y_center_px = y_min + height_px / 2

        x_center = x_center_px / img_width if img_width > 0 else 0.5
        y_center = y_center_px / img_height if img_height > 0 else 0.5
        width = width_px / img_width if img_width > 0 else 0
        height = height_px / img_height if img_height > 0 else 0

        return BoundingBox(class_id, x_center, y_center, width, height)

    def to_yolo_string(self) -> str:
        return f"{self.class_id} {self.x_center:.6f} {self.y_center:.6f} {self.width:.6f} {self.height:.6f}"

    def is_valid(self) -> bool:
        return self.width > 0.001 and self.height > 0.001


def load_yolo_labels(label_path: Path) -> List[BoundingBox]:
    boxes = []
    if not label_path.exists():
        return boxes

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                boxes.append(BoundingBox(class_id, x_center, y_center, width, height))
    return boxes


def save_yolo_labels(boxes: List[BoundingBox], output_path: Path) -> None:
    with open(output_path, "w") as f:
        for box in boxes:
            if box.is_valid():
                f.write(box.to_yolo_string() + "\n")


def horizontal_flip(
    img: np.ndarray, boxes: List[BoundingBox]
) -> Tuple[np.ndarray, List[BoundingBox]]:
    flipped_img = cv2.flip(img, 1)
    flipped_boxes = []

    for box in boxes:
        new_box = BoundingBox(
            box.class_id, 1.0 - box.x_center, box.y_center, box.width, box.height
        )
        flipped_boxes.append(new_box)

    return flipped_img, flipped_boxes


def adjust_brightness_contrast(
    img: np.ndarray, alpha: float = 1.0, beta: int = 0
) -> np.ndarray:
    adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return adjusted


def add_gaussian_noise(img: np.ndarray, mean: float = 0, std: float = 10) -> np.ndarray:
    noise = np.random.normal(mean, std, img.shape).astype(np.uint8)
    noisy_img = cv2.add(img, noise)
    return noisy_img


def adjust_hsv(
    img: np.ndarray, h_shift: int = 0, s_scale: float = 1.0, v_scale: float = 1.0
) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)

    hsv[:, :, 0] = (hsv[:, :, 0] + h_shift) % 180
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * s_scale, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * v_scale, 0, 255)

    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def adjust_background_color(img: np.ndarray, target_color: str = "white") -> np.ndarray:
    color_shifts = {
        "white": (1.25, 0.3, 40),
        "cream": (1.15, 0.6, 20),
        "yellow": (1.0, 1.0, 0),
        "gray": (0.90, 0.5, 0),
        "normalize": (1.10, 0.6, 15),
    }

    if target_color not in color_shifts:
        target_color = "white"

    brightness_factor, saturation_factor, brightness_add = color_shifts[target_color]

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)

    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_factor, 0, 255)

    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * brightness_factor + brightness_add, 0, 255)

    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def rotate_image_90(
    img: np.ndarray, boxes: List[BoundingBox], k: int = 1
) -> Tuple[np.ndarray, List[BoundingBox]]:
    """
    Rotate image by 90 degrees k times clockwise
    k=1: 90° CW, k=2: 180°, k=3: 270° CW (90° CCW)
    """
    rotated_img = np.rot90(img, k=-k)  # OpenCV uses different convention
    rotated_boxes = []

    h, w = img.shape[:2]

    for box in boxes:
        # Convert to pixel coordinates
        x_min, y_min, x_max, y_max = box.to_corners(w, h)

        # Apply rotation transformation
        if k == 1:  # 90° CW
            new_x_min = y_min
            new_y_min = w - x_max
            new_x_max = y_max
            new_y_max = w - x_min
            new_w, new_h = h, w
        elif k == 2:  # 180°
            new_x_min = w - x_max
            new_y_min = h - y_max
            new_x_max = w - x_min
            new_y_max = h - y_min
            new_w, new_h = w, h
        elif k == 3:  # 270° CW (90° CCW)
            new_x_min = h - y_max
            new_y_min = x_min
            new_x_max = h - y_min
            new_y_max = x_max
            new_w, new_h = h, w
        else:
            continue

        new_box = BoundingBox.from_corners(
            box.class_id, new_x_min, new_y_min, new_x_max, new_y_max, new_w, new_h
        )
        rotated_boxes.append(new_box)

    return rotated_img, rotated_boxes


def random_crop_with_boxes(
    img: np.ndarray, boxes: List[BoundingBox], crop_ratio: float = 0.9
) -> Tuple[np.ndarray, List[BoundingBox]]:
    """Randomly crop image while keeping all bounding boxes"""
    h, w = img.shape[:2]
    new_w = int(w * crop_ratio)
    new_h = int(h * crop_ratio)

    # Calculate crop coordinates to keep all boxes
    max_x_start = 0
    max_y_start = 0
    min_x_end = w
    min_y_end = h

    for box in boxes:
        x_min, y_min, x_max, y_max = box.to_corners(w, h)
        max_x_start = max(max_x_start, x_max - new_w)
        max_y_start = max(max_y_start, y_max - new_h)
        min_x_end = min(min_x_end, x_min + new_w)
        min_y_end = min(min_y_end, y_min + new_h)

    # Ensure valid crop region
    if max_x_start >= min_x_end - new_w or max_y_start >= min_y_end - new_h:
        return img, boxes  # Can't crop without losing boxes

    x_start = random.randint(int(max_x_start), int(min_x_end - new_w))
    y_start = random.randint(int(max_y_start), int(min_y_end - new_h))

    cropped_img = img[y_start : y_start + new_h, x_start : x_start + new_w]

    # Adjust bounding boxes
    cropped_boxes = []
    for box in boxes:
        x_min, y_min, x_max, y_max = box.to_corners(w, h)
        new_x_min = x_min - x_start
        new_y_min = y_min - y_start
        new_x_max = x_max - x_start
        new_y_max = y_max - y_start

        new_box = BoundingBox.from_corners(
            box.class_id, new_x_min, new_y_min, new_x_max, new_y_max, new_w, new_h
        )
        if new_box.is_valid():
            cropped_boxes.append(new_box)

    return cropped_img, cropped_boxes
