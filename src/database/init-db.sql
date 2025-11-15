-- SIADS 699 Capstone - PDF OCR & CNN Training Database - Initialization script for PostgreSQL database

-- Enable UUID extension for unique identifiers
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- PDF metadata
CREATE TABLE IF NOT EXISTS documents_metadata (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    filename VARCHAR(255) NOT NULL,
    file_path TEXT NOT NULL,
    file_size_bytes BIGINT,
    page_count INTEGER,
    upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    document_type VARCHAR(50), -- e.g., 'invoice', 'form', 'receipt', etc.
    status VARCHAR(20) DEFAULT 'uploaded', -- 'uploaded', 'processing', 'completed', 'failed'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Document Pages: Individual pages extracted from PDFs
CREATE TABLE IF NOT EXISTS document_pages (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID NOT NULL REFERENCES documents_metadata(id) ON DELETE CASCADE,
    page_number INTEGER NOT NULL,
    image_path TEXT, -- Path to extracted page image
    width INTEGER,
    height INTEGER,
    dpi INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(document_id, page_number)
);

-- OCR Results: Store OCR text extraction results
CREATE TABLE IF NOT EXISTS ocr_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    page_id UUID NOT NULL REFERENCES document_pages(id) ON DELETE CASCADE,
    ocr_engine VARCHAR(50) DEFAULT 'tesseract', -- 'tesseract', 'easyocr', etc.
    full_text TEXT,
    confidence_score FLOAT, -- Overall confidence (0-100)
    language VARCHAR(10) DEFAULT 'eng',
    processing_time_seconds FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- OCR Bounding Boxes: Word-level or line-level OCR results with coordinates
CREATE TABLE IF NOT EXISTS ocr_bounding_boxes (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    ocr_result_id UUID NOT NULL REFERENCES ocr_results(id) ON DELETE CASCADE,
    text VARCHAR(500),
    confidence FLOAT,
    x_min INTEGER, -- Bounding box coordinates
    y_min INTEGER,
    x_max INTEGER,
    y_max INTEGER,
    box_type VARCHAR(20) DEFAULT 'word', -- 'word', 'line', 'paragraph'
    sequence_number INTEGER -- Order in document
);

-- Annotations: Ground truth annotations for training data
CREATE TABLE IF NOT EXISTS annotations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    page_id UUID NOT NULL REFERENCES document_pages(id) ON DELETE CASCADE,
    annotation_type VARCHAR(50), -- 'bbox', 'classification', 'segmentation'
    label VARCHAR(100), -- Class label or entity type
    x_min INTEGER,
    y_min INTEGER,
    x_max INTEGER,
    y_max INTEGER,
    polygon_points JSONB, -- For complex shapes
    confidence FLOAT,
    annotator VARCHAR(100), -- Who created this annotation
    annotation_source VARCHAR(50) DEFAULT 'manual', -- 'manual', 'model', 'label-studio'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Models: Track trained models
CREATE TABLE IF NOT EXISTS models (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_name VARCHAR(100) NOT NULL,
    model_type VARCHAR(50), -- 'yolo', 'cnn', 'transformer', etc.
    architecture VARCHAR(100), -- 'yolov8', 'resnet50', etc.
    model_path TEXT, -- Path to saved model file
    hyperparameters JSONB, -- Store model config as JSON
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Training Runs: Track Optuna trials
CREATE TABLE IF NOT EXISTS training_runs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_id UUID REFERENCES models(id) ON DELETE SET NULL,
    run_name VARCHAR(100),
    dataset_split JSONB, -- {'train': 0.7, 'val': 0.2, 'test': 0.1}
    num_epochs INTEGER,
    batch_size INTEGER,
    learning_rate FLOAT,
    training_metrics JSONB, -- Store metrics as JSON
    validation_metrics JSONB,
    best_accuracy FLOAT,
    best_loss FLOAT,
    status VARCHAR(20) DEFAULT 'running', -- 'running', 'completed', 'failed'
    start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    end_time TIMESTAMP,
    duration_seconds INTEGER,
    tensorboard_log_dir TEXT
);

-- Training Epochs: Detailed metrics per epoch
CREATE TABLE IF NOT EXISTS training_epochs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    training_run_id UUID NOT NULL REFERENCES training_runs(id) ON DELETE CASCADE,
    epoch_number INTEGER NOT NULL,
    train_loss FLOAT,
    train_accuracy FLOAT,
    val_loss FLOAT,
    val_accuracy FLOAT,
    learning_rate FLOAT,
    epoch_time_seconds FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Predictions: Store model inference results
CREATE TABLE IF NOT EXISTS predictions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_id UUID REFERENCES models(id) ON DELETE SET NULL,
    page_id UUID REFERENCES document_pages(id) ON DELETE CASCADE,
    prediction_class VARCHAR(100),
    confidence_score FLOAT,
    bounding_box JSONB, -- {x_min, y_min, x_max, y_max}
    inference_time_ms FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

<<<<<<<< HEAD:src/database/init-db.sql
-- ============================================================================
-- Parquet-based OCR Processing Tables
-- ============================================================================
-- These tables support OCR processing from parquet files (ocr_processor.py)

-- Main OCR results from parquet files
CREATE TABLE IF NOT EXISTS parquet_ocr_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    parquet_file VARCHAR(255) NOT NULL,
    row_index INTEGER NOT NULL,
    label VARCHAR(100),
    image_size_width INTEGER,
    image_size_height INTEGER,
    image_mode VARCHAR(20), -- 'RGB', 'L', 'RGBA', etc.
    ocr_engine VARCHAR(50) NOT NULL, -- 'tesseract', 'yolo', 'combined'
    tesseract_full_text TEXT,
    tesseract_confidence FLOAT,
    tesseract_word_count INTEGER DEFAULT 0,
    yolo_region_count INTEGER DEFAULT 0,
    processing_status VARCHAR(20) DEFAULT 'success', -- 'success', 'error'
    processing_error TEXT,
    processing_time_seconds FLOAT,
    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(parquet_file, row_index, ocr_engine)
);

-- Word-level Tesseract OCR details
CREATE TABLE IF NOT EXISTS parquet_ocr_words (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    ocr_result_id UUID NOT NULL REFERENCES parquet_ocr_results(id) ON DELETE CASCADE,
    word_text VARCHAR(500),
    confidence INTEGER, -- 0-100
    x_min INTEGER,
    y_min INTEGER,
    x_max INTEGER,
    y_max INTEGER,
    sequence_number INTEGER -- Order in document
);

-- YOLO detected text regions
CREATE TABLE IF NOT EXISTS parquet_yolo_regions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    ocr_result_id UUID NOT NULL REFERENCES parquet_ocr_results(id) ON DELETE CASCADE,
    class_id INTEGER,
    confidence FLOAT, -- 0-1
    x_min INTEGER,
    y_min INTEGER,
    x_max INTEGER,
    y_max INTEGER
);

-- ============================================================================
========
>>>>>>>> 7e81b0977500d76cdcdca4eb42dadd0eb5b3110d:scripts/database/init-db.sql
-- Indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_documents_metadata_status ON documents_metadata(status);
CREATE INDEX IF NOT EXISTS idx_documents_metadata_type ON documents_metadata(document_type);
CREATE INDEX IF NOT EXISTS idx_document_pages_doc_id ON document_pages(document_id);
CREATE INDEX IF NOT EXISTS idx_ocr_results_page_id ON ocr_results(page_id);
CREATE INDEX IF NOT EXISTS idx_annotations_page_id ON annotations(page_id);
CREATE INDEX IF NOT EXISTS idx_annotations_label ON annotations(label);
CREATE INDEX IF NOT EXISTS idx_training_runs_model_id ON training_runs(model_id);
CREATE INDEX IF NOT EXISTS idx_training_epochs_run_id ON training_epochs(training_run_id);
CREATE INDEX IF NOT EXISTS idx_predictions_model_id ON predictions(model_id);
CREATE INDEX IF NOT EXISTS idx_predictions_page_id ON predictions(page_id);

<<<<<<<< HEAD:src/database/init-db.sql
-- Parquet tables indexes
CREATE INDEX IF NOT EXISTS idx_parquet_ocr_results_parquet_file ON parquet_ocr_results(parquet_file);
CREATE INDEX IF NOT EXISTS idx_parquet_ocr_results_status ON parquet_ocr_results(processing_status);
CREATE INDEX IF NOT EXISTS idx_parquet_ocr_results_engine ON parquet_ocr_results(ocr_engine);
CREATE INDEX IF NOT EXISTS idx_parquet_ocr_words_result_id ON parquet_ocr_words(ocr_result_id);
CREATE INDEX IF NOT EXISTS idx_parquet_yolo_regions_result_id ON parquet_yolo_regions(ocr_result_id);

-- ============================================================================
========
>>>>>>>> 7e81b0977500d76cdcdca4eb42dadd0eb5b3110d:scripts/database/init-db.sql
-- Update trigger for updated_at columns
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_documents_metadata_updated_at BEFORE UPDATE ON documents_metadata
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_annotations_updated_at BEFORE UPDATE ON annotations
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

<<<<<<<< HEAD:src/database/init-db.sql
-- ============================================================================
-- Sample data (optional - for testing)
-- ============================================================================
-- Insert a sample document
INSERT INTO documents (filename, file_path, file_size_bytes, page_count, document_type, status)
VALUES
    ('sample_invoice.pdf', '/workspace/datasets/sample_invoice.pdf', 524288, 1, 'invoice', 'uploaded'),
    ('sample_form.pdf', '/workspace/datasets/sample_form.pdf', 1048576, 2, 'form', 'uploaded')
ON CONFLICT DO NOTHING;

-- Create a sample model record
INSERT INTO models (model_name, model_type, architecture, description)
VALUES
    ('yolov8-invoice-detector', 'yolo', 'yolov8n', 'YOLOv8 nano model for detecting invoice fields'),
    ('document-classifier', 'cnn', 'resnet50', 'ResNet50 for document type classification')
ON CONFLICT DO NOTHING;

-- ============================================================================
========
>>>>>>>> 7e81b0977500d76cdcdca4eb42dadd0eb5b3110d:scripts/database/init-db.sql
-- Useful Views
-- View to get document processing status with page counts
CREATE OR REPLACE VIEW document_processing_status AS
SELECT
    d.id,
    d.filename,
    d.document_type,
    d.status,
    d.page_count,
    COUNT(DISTINCT dp.id) as pages_extracted,
    COUNT(DISTINCT ocr.id) as pages_with_ocr,
    d.created_at
FROM documents_metadata d
LEFT JOIN document_pages dp ON d.id = dp.document_id
LEFT JOIN ocr_results ocr ON dp.id = ocr.page_id
GROUP BY d.id, d.filename, d.document_type, d.status, d.page_count, d.created_at;

-- View for training run summary
CREATE OR REPLACE VIEW training_run_summary AS
SELECT
    tr.id,
    tr.run_name,
    m.model_name,
    m.architecture,
    tr.num_epochs,
    tr.best_accuracy,
    tr.best_loss,
    tr.status,
    tr.duration_seconds,
    COUNT(te.id) as epochs_completed,
    tr.start_time,
    tr.end_time
FROM training_runs tr
LEFT JOIN models m ON tr.model_id = m.id
LEFT JOIN training_epochs te ON tr.id = te.training_run_id
GROUP BY tr.id, tr.run_name, m.model_name, m.architecture,
         tr.num_epochs, tr.best_accuracy, tr.best_loss,
         tr.status, tr.duration_seconds, tr.start_time, tr.end_time;

-- PARQUET FILE METADATA & OCR RESULTS

-- Parquet file metadata table (lmcheck)
CREATE TABLE IF NOT EXISTS lmcheck (
    id SERIAL PRIMARY KEY,
    parquet_file VARCHAR(255) NOT NULL,
    row_index INTEGER NOT NULL,
    label INTEGER NOT NULL,
    image_size_bytes INTEGER,
    image_path TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(parquet_file, row_index)
);

-- Parquet OCR results table
CREATE TABLE IF NOT EXISTS parquet_ocr_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    parquet_file VARCHAR(255) NOT NULL,
    row_index INTEGER NOT NULL,
    label INTEGER,
    image_size_width INTEGER,
    image_size_height INTEGER,
    image_mode VARCHAR(10),
    ocr_engine VARCHAR(50) NOT NULL,
    tesseract_full_text TEXT,
    tesseract_confidence FLOAT,
    tesseract_word_count INTEGER,
    yolo_region_count INTEGER DEFAULT 0,
    processing_status VARCHAR(20) DEFAULT 'success',
    processing_error TEXT,
    processing_time_seconds FLOAT,
    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(parquet_file, row_index, ocr_engine)
);

-- Parquet OCR word-level details
CREATE TABLE IF NOT EXISTS parquet_ocr_words (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    ocr_result_id UUID NOT NULL REFERENCES parquet_ocr_results(id) ON DELETE CASCADE,
    word_text VARCHAR(500),
    confidence FLOAT,
    x_min INTEGER,
    y_min INTEGER,
    x_max INTEGER,
    y_max INTEGER,
    sequence_number INTEGER
);

-- Parquet YOLO detected regions
CREATE TABLE IF NOT EXISTS parquet_yolo_regions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    ocr_result_id UUID NOT NULL REFERENCES parquet_ocr_results(id) ON DELETE CASCADE,
    class_id INTEGER,
    confidence FLOAT,
    x_min INTEGER,
    y_min INTEGER,
    x_max INTEGER,
    y_max INTEGER
);

-- =============================================================================
-- INDEXES FOR PARQUET TABLES
-- =============================================================================

CREATE INDEX IF NOT EXISTS idx_lmcheck_label ON lmcheck(label);
CREATE INDEX IF NOT EXISTS idx_lmcheck_parquet_file ON lmcheck(parquet_file);
CREATE INDEX IF NOT EXISTS idx_lmcheck_size ON lmcheck(image_size_bytes);
CREATE INDEX IF NOT EXISTS idx_parquet_ocr_parquet_file ON parquet_ocr_results(parquet_file);
CREATE INDEX IF NOT EXISTS idx_parquet_ocr_label ON parquet_ocr_results(label);
CREATE INDEX IF NOT EXISTS idx_parquet_ocr_engine ON parquet_ocr_results(ocr_engine);
CREATE INDEX IF NOT EXISTS idx_parquet_ocr_status ON parquet_ocr_results(processing_status);
CREATE INDEX IF NOT EXISTS idx_parquet_ocr_words_result_id ON parquet_ocr_words(ocr_result_id);
CREATE INDEX IF NOT EXISTS idx_parquet_yolo_regions_result_id ON parquet_yolo_regions(ocr_result_id);

-- =============================================================================
-- VIEWS FOR PARQUET DATA
-- =============================================================================

-- Combined view of OCR results with parquet metadata
CREATE OR REPLACE VIEW parquet_ocr_summary AS
SELECT
    ocr.id,
    ocr.parquet_file,
    ocr.row_index,
    ocr.label,
    ocr.ocr_engine,
    ocr.tesseract_full_text,
    ocr.tesseract_confidence,
    ocr.tesseract_word_count,
    ocr.yolo_region_count,
    ocr.processing_status,
    ocr.processed_at,
    lm.image_size_bytes
FROM parquet_ocr_results ocr
LEFT JOIN lmcheck lm ON ocr.parquet_file = lm.parquet_file AND ocr.row_index = lm.row_index
ORDER BY ocr.processed_at DESC;

-- =============================================================================
-- SUCCESS MESSAGE
-- =============================================================================

DO $$
BEGIN
    RAISE NOTICE '✓ Database schema initialized successfully';
<<<<<<<< HEAD:src/database/init-db.sql
    RAISE NOTICE '✓ PDF workflow tables: documents, document_pages, ocr_results, ocr_bounding_boxes, annotations';
    RAISE NOTICE '✓ Training tables: models, training_runs, training_epochs, predictions';
    RAISE NOTICE '✓ Parquet workflow tables: parquet_ocr_results, parquet_ocr_words, parquet_yolo_regions';
    RAISE NOTICE '✓ Sample data inserted';
========
    RAISE NOTICE '✓ PDF Processing: documents_metadata, document_pages, ocr_results, ocr_bounding_boxes, annotations';
    RAISE NOTICE '✓ ML Training: models, training_runs, training_epochs, predictions';
    RAISE NOTICE '✓ Parquet Data: lmcheck, parquet_ocr_results, parquet_ocr_words, parquet_yolo_regions';
    RAISE NOTICE '✓ Views: document_processing_status, training_run_summary, parquet_ocr_summary';
>>>>>>>> 7e81b0977500d76cdcdca4eb42dadd0eb5b3110d:scripts/database/init-db.sql
END $$;
