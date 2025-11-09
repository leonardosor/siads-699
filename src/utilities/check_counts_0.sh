#!/bin/bash

folders=("/workspace/data/input/training/images/" "/workspace/data/input/testing/images/" "/workspace/data/input/validation/images/")
for folder in "${folders[@]}"; do
    count=$(cd $folder && ls -ld * | uniq | wc -l)
    echo "$folder = $count"
done
