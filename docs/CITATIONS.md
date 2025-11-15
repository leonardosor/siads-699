# Citations and References

## Technical References

- [YOLOv8 Usage Examples](https://docs.ultralytics.com/models/yolov8/#yolov8-usage-examples)
- [YOLOv8 Research Paper](https://arxiv.org/html/2408.15857)
- [Train YOLOv8 on Custom Dataset](https://learnopencv.com/train-yolov8-on-custom-dataset/)

## AI Assistance Acknowledgment

This deployment stack was prepared in collaboration with AI assistants (OpenAI GPT-5 Thinking and Anthropic Claude).

If referenced academically, a generic software-collaboration citation format is appropriate:

> OpenAI. "GPT-5 Thinking (Assistant)," collaborative systems engineering guidance for containerized ML pipelines (YOLOv8/Tesseract/PostgreSQL/Streamlit), 2025. Contribution: environment orchestration, reproducible builds, and runbook generation. URL: https://openai.com/

> Anthropic. "Claude Code," AI-assisted code refactoring and repository organization, 2025. Contribution: codebase restructuring, best practices implementation, and documentation. URL: https://claude.ai/

## Reproducibility

For reproducibility statements, include:
- Dockerfile/Compose digests
- Package versions in [src/environments/requirements-prod.txt](../src/environments/requirements-prod.txt) and [src/environments/requirements-dev.txt](../src/environments/requirements-dev.txt)
- Docker Compose configuration in [src/environments/docker/compose.yml](../src/environments/docker/compose.yml)
