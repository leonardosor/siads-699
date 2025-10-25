# Dev Container Changes

## Summary

The `.devcontainer` now uses the root `../Dockerfile` instead of a separate Dockerfile, providing consistency between dev and standalone setups.

## Changes

- **Removed**: `.devcontainer/Dockerfile` (now uses `../Dockerfile`)
- **Updated**: `docker-compose.yml`, `devcontainer.json`, and `README.md`

## Benefits

- Single shared Docker image
- Consistent dev/prod environments
- Faster builds and ~50% disk space savings
- Easier maintenance

## Usage

Build the image:
```bash
docker-compose build ocr-cnn
```

Open in VS Code:
```bash
code .
# Ctrl+Shift+P → "Dev Containers: Reopen in Container"
```

## Key Differences

- **Dev Container**: Shared Dockerfile + PostgreSQL + VS Code
- **Standalone**: Shared Dockerfile only

## Notes

- GPU support enabled by default (comment out `deploy` section in `docker-compose.yml` if no GPU)
- Database: `psql -h localhost -U postgres` (password: postgres)
- For CPU version: Edit `docker-compose.yml` to use `Dockerfile.cpu`
