-- SIADS 699 Capstone - PDF OCR & CNN Training Database
-- Initialization script for PostgreSQL database

-- Enable UUID extension for unique identifiers
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================================================
-- Raw Training Images: Store training dataset images
-- ============================================================================
CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    image BYTEA NOT NULL,  -- Store image bytes directly
    label INTEGER NOT NULL,  -- Image classification label
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- Documents Table: PDF metadata
-- ============================================================================
CREATE TABLE IF NOT EXISTS raw_documents (
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

-- ============================================================================
-- Document Pages: Individual pages extracted from PDFs
-- ============================================================================
CREATE TABLE IF NOT EXISTS document_pages (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID NOT NULL REFERENCES raw_documents(id) ON DELETE CASCADE,
    page_number INTEGER NOT NULL,
    image_path TEXT, -- Path to extracted page image
    width INTEGER,
    height INTEGER,
    dpi INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(document_id, page_number)
);

-- ============================================================================
-- OCR Results: Store OCR text extraction results
-- ============================================================================
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

-- ============================================================================
-- OCR Bounding Boxes: Word-level or line-level OCR results with coordinates
-- ============================================================================
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

-- ============================================================================
-- Annotations: Ground truth annotations for training data
-- ============================================================================
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

-- ============================================================================
-- Models: Track trained models
-- ============================================================================
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

-- ============================================================================
-- Training Runs: Track model training sessions
-- ============================================================================
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

-- ============================================================================
-- Training Epochs: Detailed metrics per epoch
-- ============================================================================
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

-- ============================================================================
-- Predictions: Store model inference results
-- ============================================================================
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

-- ============================================================================
-- Indexes for better query performance
-- ============================================================================
CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status);
CREATE INDEX IF NOT EXISTS idx_documents_type ON documents(document_type);
CREATE INDEX IF NOT EXISTS idx_document_pages_doc_id ON document_pages(document_id);
CREATE INDEX IF NOT EXISTS idx_ocr_results_page_id ON ocr_results(page_id);
CREATE INDEX IF NOT EXISTS idx_annotations_page_id ON annotations(page_id);
CREATE INDEX IF NOT EXISTS idx_annotations_label ON annotations(label);
CREATE INDEX IF NOT EXISTS idx_training_runs_model_id ON training_runs(model_id);
CREATE INDEX IF NOT EXISTS idx_training_epochs_run_id ON training_epochs(training_run_id);
CREATE INDEX IF NOT EXISTS idx_predictions_model_id ON predictions(model_id);
CREATE INDEX IF NOT EXISTS idx_predictions_page_id ON predictions(page_id);

-- ============================================================================
-- Update trigger for updated_at columns
-- ============================================================================
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_documents_updated_at BEFORE UPDATE ON documents
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_annotations_updated_at BEFORE UPDATE ON annotations
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- Sample data (optional - for testing)
-- ============================================================================
-- Insert a sample document
INSERT INTO documents (filename, file_path, file_size_bytes, page_count, document_type, status)
VALUES
    ('sample_invoice.pdf', '/workspace/data/input/sample_invoice.pdf', 524288, 1, 'invoice', 'uploaded'),
    ('sample_form.pdf', '/workspace/data/input/sample_form.pdf', 1048576, 2, 'form', 'uploaded')
ON CONFLICT DO NOTHING;

-- Create a sample model record
INSERT INTO models (model_name, model_type, architecture, description)
VALUES
    ('yolov8-invoice-detector', 'yolo', 'yolov8n', 'YOLOv8 nano model for detecting invoice fields'),
    ('document-classifier', 'cnn', 'resnet50', 'ResNet50 for document type classification')
ON CONFLICT DO NOTHING;

-- ============================================================================
-- Useful Views
-- ============================================================================
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
FROM documents d
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

-- ============================================================================
-- Grant permissions (if needed for specific users)
-- ============================================================================
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO postgres;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO postgres;

-- Success message
DO $$
BEGIN
    RAISE NOTICE '✓ Database schema initialized successfully';
    RAISE NOTICE '✓ Tables created: documents, document_pages, ocr_results, annotations, models, training_runs, predictions';
    RAISE NOTICE '✓ Sample data inserted';
END $$;
