#!bin/bash

echo "Cleaning up files!"

cd /workspace/data/input/training/images && rm -rf * && \
cd /workspace/data/input/testing/images && rm -rf * && \
cd /workspace/data/input/validation/images && rm -rf *

echo "Cleaned up files!"
