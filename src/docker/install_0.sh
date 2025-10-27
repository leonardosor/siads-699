#!/bin/bash
set -e

yes | apt install vim && \
python3 -m venv .env && \
bash .env/bin/activate && \
python3 -m pip install -r /workspace/requirements.txt --break-system-packages