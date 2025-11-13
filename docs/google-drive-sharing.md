# Google Drive Distribution Checklist

Use this checklist whenever you need to hand off the project bundle so teammates can run Streamlit locally without cloning GitHub.

## 1. Source Snapshot
- From repo root: `zip -r siads-699-src.zip . -x "*.git*" "models/runs/*/weights/*.pt" "models/best.pt"`
- Upload `siads-699-src.zip` to Drive (e.g., `SIADS-699/shared`).

## 2. Model Weights
- Include the active checkpoint (`models/best.pt`).
- Include any archived weights you want others to test:
  - `models/runs/<run-name>/weights/best.pt`
  - `models/runs/<run-name>/weights/last.pt` (optional)
- Compress large weights if desired (e.g., `tar -czf models-weights.tar.gz models/best.pt models/runs/<run-name>/weights/best.pt`).

## 3. Run Metadata
- Already tracked in Git, but mirror them on Drive for non-git users:
  - `models/runs/<run-name>/results.csv`
  - `models/runs/<run-name>/args.yaml`
  - `models/runs/<run-name>/train_batch*.jpg`
- Package via: `zip -r run-metadata-<run-name>.zip models/runs/<run-name> -x "*/weights/*.pt"`

## 4. Optional Dataset Links
- Add a short README section pointing to the canonical dataset location or shareable Label Studio export so others can retrain.

## 5. Setup Instructions (text snippet to paste in Drive description)
```
1. Download `siads-699-src.zip` and extract.
2. Place `models/best.pt` inside the extracted `models/` folder (overwrite if prompted).
3. (Optional) Copy additional `models/runs/<run-name>` folders if you want historical metrics.
4. From repo root run `docker compose -f src/docker/compose.yml up --build --remove-orphans`.
5. Visit http://localhost:8501 to use the Streamlit interface.
```

Keep this file local-only; it's ignored by Git.
