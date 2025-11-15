#!/usr/bin/env pwsh
# PowerShell wrapper to run OCR visualization in Docker container
# Usage: .\visualize.ps1 [image_index] [parquet_file]
# Example: .\visualize.ps1 5 train-00001-of-00005.parquet

param(
    [int]$ImageIndex = 0,
    [string]$ParquetFile = "train-00000-of-00005.parquet"
)

Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 69) -ForegroundColor Cyan
Write-Host "  OCR Visualization (Running in Docker)" -ForegroundColor Yellow
Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 69) -ForegroundColor Cyan

# Check if Docker is running
try {
    docker ps | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "`n✗ Docker is not running. Please start Docker Desktop.`n" -ForegroundColor Red
        exit 1
    }
} catch {
    Write-Host "`n✗ Docker is not available. Please install Docker Desktop.`n" -ForegroundColor Red
    exit 1
}

# Check if container is running
$containerStatus = docker ps --filter "name=699-devcontainer" --format "{{.Status}}"
if (-not $containerStatus) {
    Write-Host "`n✗ Container '699-devcontainer' is not running." -ForegroundColor Red
    Write-Host "  Starting containers with docker-compose...`n" -ForegroundColor Yellow
    docker-compose up -d
    Start-Sleep -Seconds 5
}

# Build the parquet path
$parquetPath = "/workspace/data/raw/$ParquetFile"

Write-Host "`nProcessing:" -ForegroundColor Cyan
Write-Host "  • Parquet file: $ParquetFile" -ForegroundColor White
Write-Host "  • Image index: $ImageIndex" -ForegroundColor White
Write-Host ""

# Run the visualization
docker exec 699-devcontainer bash -c "cd /workspace && python scripts/visualize_ocr_boxes.py $parquetPath $ImageIndex"

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n✓ Success! Visualization saved to:" -ForegroundColor Green
    Write-Host "  data\output\visualizations\ocr_visualization_$($ParquetFile.Replace('.parquet',''))_img$ImageIndex.png`n" -ForegroundColor White

    # Optionally open the image
    $imagePath = "data\output\visualizations\ocr_visualization_$($ParquetFile.Replace('.parquet',''))_img$ImageIndex.png"
    if (Test-Path $imagePath) {
        $response = Read-Host "Open the image now? (y/n)"
        if ($response -eq 'y' -or $response -eq 'Y') {
            Start-Process $imagePath
        }
    }
} else {
    Write-Host "`n✗ Visualization failed. See error above.`n" -ForegroundColor Red
}
