# Product Overview

## Financial Form Text Extractor

A full-stack text extraction application for scanned financial forms (invoices, receipts, documents). The system uses computer vision to identify document regions (header, body, footer) and extracts text specifically from the body section to avoid irrelevant content.

## Core Workflow

1. **Region Detection**: Fine-tuned YOLOv8 model identifies header/body/footer regions in document images
2. **Text Extraction**: Tesseract OCR extracts text only from body regions
3. **Data Storage**: Extracted data stored in PostgreSQL with full metadata tracking
4. **Web Interface**: Streamlit application for interactive document processing and visualization

## Key Features

- Manual ground-truth annotations (100 images) with proper bounding boxes
- Data augmentation pipeline to generate training datasets (1k-20k samples)
- Hyperparameter optimization with Optuna
- Docker-based microservices architecture
- Great Lakes HPC integration for GPU training
- University of Michigan branding (maize/blue color scheme)

## Target Use Case

Processing scanned financial forms where only the body content is relevant, filtering out headers (logos, titles) and footers (page numbers, disclaimers).
