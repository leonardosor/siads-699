"""Merge and renumber a malformed COCO-style annotation file.

The input file (`results_c.txt`) currently contains TWO full COCO JSON objects
concatenated back to back (a copy/paste from different Label Studio exports).
Both objects define their own `images`, `categories`, and `annotations` blocks
with image ids and annotation ids restarting at 0. This breaks downstream
consumers (YOLO / COCO tooling) that expect a single JSON structure with
globally unique, sequential `image_id` and `id` fields.

This script merges those two objects into a single valid COCO JSON:

1. Parse the concatenated file into two JSON objects.
2. Concatenate the image lists and assign NEW sequential image ids 0..N-1.
3. Remap all annotation `image_id` values to the new image ids.
4. Reassign annotation ids sequentially 0..M-1.
5. Keep the `categories` list from the first object (it should match the second).
6. Preserve the first object's `info` block, appending a note about the merge.
7. Write a new JSON file (`results_c_fixed.json`).

Usage (from repository root):
  python src/utils/dataset/merge_and_renumber_coco.py \
         --input data/input/ground-truth/results_c.txt \
         --output data/input/ground-truth/results_c_fixed.json

Validation performed:
  - Ensures exactly two top-level JSON objects are found.
  - Asserts categories are identical (by id+name) across both objects.
  - Ensures resulting image ids are contiguous and start at 0.
  - Ensures annotation image_id references are valid after remap.

Edge cases handled:
  - Missing expected delimiter between JSON objects.
  - Duplicate image file names across the two objects (warning only).
  - Missing images referenced by annotations (raises error).

If you later obtain the missing img_100.jpg you can append it by inserting an
image entry and adjusting downstream splits, but this script intentionally keeps
the original file_name values unchanged.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass
class CocoObject:
    images: List[dict]
    annotations: List[dict]
    categories: List[dict]
    info: dict | None


def load_concatenated_coco(path: Path) -> List[CocoObject]:
    """Parse two concatenated top-level JSON objects by brace counting.

    This is more robust than regex splitting and tolerates arbitrary internal
    whitespace and newlines. Assumes exactly two objects concatenated.
    """
    text = path.read_text(encoding="utf-8")
    start = text.find('{')
    if start == -1:
        raise ValueError("No opening brace found in file.")
    depth = 0
    end_index = None
    for i, ch in enumerate(text[start:], start=start):
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                end_index = i + 1
                break
    if end_index is None:
        raise ValueError("Failed to locate end of first JSON object (brace mismatch).")

    first_chunk = text[start:end_index].strip()
    second_chunk = text[end_index:].strip()
    if not second_chunk.startswith('{'):
        # Trim any leading separators/newlines until next '{'
        next_obj_start = second_chunk.find('{')
        if next_obj_start == -1:
            raise ValueError("Second JSON object not found after first.")
        second_chunk = second_chunk[next_obj_start:]

    objects_raw = [first_chunk, second_chunk]
    coco_objects: List[CocoObject] = []
    for i, raw in enumerate(objects_raw):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            snippet = raw[max(0, e.pos - 50): e.pos + 50]
            raise ValueError(f"Failed to parse chunk {i}: {e}\nContext: {snippet}") from e
        coco_objects.append(
            CocoObject(
                images=data.get("images", []),
                annotations=data.get("annotations", []),
                categories=data.get("categories", []),
                info=data.get("info"),
            )
        )

    if len(coco_objects) != 2:
        raise ValueError(f"Expected exactly two JSON objects, found {len(coco_objects)}.")
    return coco_objects


def assert_categories_match(objs: List[CocoObject]) -> None:
    first = objs[0].categories
    second = objs[1].categories
    if len(first) != len(second):
        raise ValueError("Category count mismatch between objects.")
    f_map = {(c["id"], c["name"]): c for c in first}
    for c in second:
        key = (c.get("id"), c.get("name"))
        if key not in f_map:
            raise ValueError(f"Category mismatch: {key} not in first set.")


def merge_and_renumber(objs: List[CocoObject]) -> dict:
    assert_categories_match(objs)

    images_all: List[dict] = []
    image_id_remap: List[Dict[int, int]] = []  # per-object old->new

    # Combine images with new sequential ids
    next_image_id = 0
    for obj_idx, obj in enumerate(objs):
        remap: Dict[int, int] = {}
        for img in obj.images:
            new_img = img.copy()
            old_id = new_img["id"]
            new_img["id"] = next_image_id
            images_all.append(new_img)
            remap[old_id] = next_image_id
            next_image_id += 1
        image_id_remap.append(remap)

    # Warn if duplicate file names
    file_name_counts: Dict[str, int] = {}
    for img in images_all:
        fn = img.get("file_name")
        file_name_counts[fn] = file_name_counts.get(fn, 0) + 1
    duplicates = [fn for fn, cnt in file_name_counts.items() if cnt > 1]
    if duplicates:
        print(f"[WARNING] Duplicate image file names detected: {duplicates[:10]}{'...' if len(duplicates) > 10 else ''}")

    # Merge and renumber annotations
    annotations_all: List[dict] = []
    next_ann_id = 0
    for obj_idx, obj in enumerate(objs):
        remap = image_id_remap[obj_idx]
        for ann in obj.annotations:
            new_ann = ann.copy()
            old_image_id = new_ann["image_id"]
            if old_image_id not in remap:
                raise ValueError(
                    f"Annotation references missing image_id {old_image_id} in object {obj_idx}"
                )
            new_ann["image_id"] = remap[old_image_id]
            new_ann["id"] = next_ann_id
            annotations_all.append(new_ann)
            next_ann_id += 1

    merged_info = objs[0].info.copy() if objs[0].info else {}
    note = "Merged two concatenated COCO exports; ids renumbered sequentially."
    if merged_info.get("description"):
        merged_info["description"] += f" | {note}"
    else:
        merged_info["description"] = note

    result = {
        "info": merged_info,
        "images": images_all,
        "categories": objs[0].categories,  # validated identical
        "annotations": annotations_all,
    }

    # Final validation
    image_ids = [img["id"] for img in images_all]
    assert image_ids == list(range(len(image_ids))), "Image ids are not contiguous starting at 0."
    ann_ids = [ann["id"] for ann in annotations_all]
    assert ann_ids == list(range(len(ann_ids))), "Annotation ids are not contiguous starting at 0."
    return result


def main():
    parser = argparse.ArgumentParser(description="Merge & renumber concatenated COCO JSON file.")
    parser.add_argument("--input", required=True, type=Path, help="Path to concatenated COCO file (results_c.txt)")
    parser.add_argument(
        "--output", required=False, type=Path, default=Path("data/input/ground-truth/results_c_fixed.json"),
        help="Destination path for merged JSON"
    )
    parser.add_argument("--indent", type=int, default=2, help="Indent level for JSON output")
    args = parser.parse_args()

    objs = load_concatenated_coco(args.input)
    merged = merge_and_renumber(objs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(merged, indent=args.indent), encoding="utf-8")
    print(f"Wrote merged file: {args.output}")
    print(f"Total images: {len(merged['images'])}; annotations: {len(merged['annotations'])}")


if __name__ == "__main__":
    main()
