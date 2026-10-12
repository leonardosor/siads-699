# SIADS 699 Repository Summary

## Overview
This repository is the University of Michigan MADS Capstone project focused on building a financial form text extraction system. The project aims to detect and extract meaningful text from scanned financial documents by combining computer vision, OCR, and a lightweight web application interface.

The core idea is to process document images, identify relevant regions with a YOLOv8 object detection model, extract text from those regions using Tesseract OCR, and present the results in a Streamlit application. The system is designed for a practical document-processing workflow and is built with Docker-based deployment in mind.

## Project Goal
The repo description frames it as a PDF/text extraction project for financial forms that uses:
- CNN / YOLOv8-based detection
- Tesseract OCR
- Docker
- Python
- Streamlit hosting

In short, the project is meant to take scanned forms and produce structured, usable output from textual content embedded in those images.

## Major Components

### 1. Model Training and Inference
The repository includes a training pipeline for YOLOv8 models, with configuration and scripts for local training and HPC/SLURM-based training workflows. Model artifacts are stored in the `models/` directory and include trained weights, experiment runs, and deployment-ready checkpoints.

Key files and directories:
- `src/training/`
- `models/`
- `scripts/`

### 2. OCR Processing
The repo includes an OCR pipeline that combines object detection and text extraction. After regions of interest are detected, OCR is run to convert those visual areas into text data.

Key areas:
- `src/processing/`
- `data/`

### 3. Streamlit Web App
The repository contains a Streamlit web interface for users to upload document images, adjust detection parameters, view predictions, and download annotated results.

Key area:
- `src/web/`

### 4. Database and Data Storage
The project uses PostgreSQL to persist OCR results, document metadata, and relevant pipeline outputs.

Key area:
- `src/database/`

### 5. Utilities and Operations
A collection of support scripts is included for dataset validation, model management, deployment, and monitoring.

Key areas:
- `src/utils/`
- `docs/`
- `config/`

## Tech Stack
- Python
- Jupyter Notebook
- YOLOv8 (Ultralytics)
- Tesseract OCR
- PostgreSQL
- Streamlit
- Docker / Docker Compose
- Shell scripting

## Repository Structure
At a high level, the repo is organized like this:

- `README.md` – project overview and main documentation
- `docs/` – quickstart and component documentation
- `src/` – application source code
  - `training/` – model training logic
  - `processing/` – OCR and post-processing pipeline
  - `web/` – Streamlit app
  - `database/` – database schema and related code
  - `utils/` – operations and dataset/model utilities
  - `environments/` – Docker and Conda environment config
- `models/` – training checkpoints, deployed model weights, and run artifacts
- `data/` – input, output, and raw datasets
- `notebooks/` – exploratory and analysis notebooks
- `scripts/` – project automation scripts
- `requirements.txt` / `packages.txt` – environment dependencies

## Typical Workflow
The repo appears to support the following end-to-end workflow:

1. Prepare or collect scanned financial form images.
2. Train or deploy a YOLOv8 model for document region detection.
3. Run inference on document images to detect relevant sections.
4. Use OCR (Tesseract) on detected areas to extract text.
5. Store results in PostgreSQL.
6. Visualize output and allow user interaction through the Streamlit app.

## Documentation Quality
The repository includes detailed documentation in:
- `README.md`
- `docs/README.md`
- `src/web/README.md`
- `src/training/README.md`
- `models/README.md`
- `tests/README.md`

This indicates the project is designed to be understandable and operational for both development and deployment.

## Overall Assessment
This is a substantial applied ML/data engineering project with a strong emphasis on:
- model training
- document understanding
- OCR integration
- deployment infrastructure
- practical user-facing workflow

It is best understood as a capstone-style end-to-end system for document extraction and visual recognition rather than a simple notebook experiment.

## Summary in One Line
SIADS 699 is a capstone project that builds a Dockerized, Python-based financial document extraction system using YOLOv8 object detection, Tesseract OCR, PostgreSQL, and a Streamlit interface.
