#!/bin/bash

folders=("/workspace/data/input/training/images/" "/workspace/data/input/testing/images/" "/workspace/data/input/validation/images/")
for folder in "${folders[@]}"; do
    echo "Processing folder from array: $folder"
    cd $folder && ls -l | uniq | wc -l

done
