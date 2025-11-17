# Claude Code Usage Guide

This guide explains how to effectively use Claude Code (AI-assisted development) when working on this YOLOv8 financial document parser project.

## What is Claude Code?

Claude Code is an AI-powered CLI tool that helps with software engineering tasks including:
- Writing and refactoring code
- Debugging and troubleshooting issues
- Navigating and understanding codebases
- Running commands and managing workflows
- Creating documentation
- Git operations and pull requests

## Getting Started

### Installation

If you haven't already installed Claude Code, follow the official setup guide at https://docs.claude.com/docs/claude-code

### Project Permissions

This project has pre-configured permissions in [.claude/settings.local.json](../.claude/settings.local.json) that allow Claude Code to:
- Run Docker commands (docker-compose, docker exec)
- Execute Python scripts and conda commands
- Perform git operations (add, commit, status)
- Use common shell commands (mkdir, mv, tree, etc.)

These permissions mean Claude Code won't ask for approval before running these common operations, making your workflow faster.

## Common Workflows

### 1. Dataset Preparation and Augmentation

**Ask Claude Code to help with:**

```
"Help me prepare the dataset for training. Check if the images and labels are properly formatted."
```

```
"Run the dataset augmentation script and let me know if there are any errors."
```

```
"I need to check for corrupt images in the training dataset. Can you scan and remove them?"
```

Claude Code can:
- Run `src/utils/dataset/prepare_dataset.py`
- Execute data augmentation scripts
- Verify image/label pairs
- Check dataset statistics

### 2. Model Training

**Local training:**

```
"Start a training run with 50 epochs on my local GPU using the default configuration."
```

```
"I want to run hyperparameter optimization with Optuna. Use 20 trials and cache the dataset."
```

```
"Resume the last training run that was interrupted."
```

**Great Lakes HPC:**

```
"Submit a SLURM batch job for training on Great Lakes with 200 epochs and batch size 8."
```

```
"Check the status of my running SLURM jobs and show me the latest training log."
```

Claude Code can:
- Run training scripts with appropriate arguments
- Monitor training progress
- Submit and monitor SLURM batch jobs
- Parse training logs and results

### 3. Model Evaluation and Analysis

**Ask Claude Code to:**

```
"Analyze the results.csv from the latest training run and tell me if the model is overfitting."
```

```
"Compare the mAP scores between my last three training runs."
```

```
"Load the Jupyter notebook and explain what experiments were run."
```

Claude Code can:
- Parse training metrics from CSV files
- Read and analyze Jupyter notebooks
- Compare model performance across runs
- Identify training issues (overfitting, underfitting, etc.)

### 4. Deployment and Web Interface

**Ask Claude Code to:**

```
"Deploy the best model from my latest training run to the Streamlit app."
```

```
"Start the Streamlit application using Docker Compose."
```

```
"Check if the web app is running and show me the container logs."
```

Claude Code can:
- Copy trained models to production directories
- Start/stop Docker containers
- Update configuration files
- Monitor application logs

### 5. Code Refactoring and Features

**Ask Claude Code to:**

```
"Add a new preprocessing function to normalize invoice images before detection."
```

```
"Refactor the training script to support mixed precision training."
```

```
"Add error handling to the data loading pipeline."
```

Claude Code can:
- Write new functions and classes
- Refactor existing code
- Add error handling and logging
- Implement new features based on requirements

### 6. Git Operations

**Ask Claude Code to:**

```
"Create a commit with all my training script changes."
```

```
"Show me what files have changed since the last commit."
```

```
"Create a pull request for my new data augmentation feature."
```

Claude Code can:
- Stage and commit changes with descriptive messages
- Create branches and pull requests
- Review diffs and file changes
- Follow git best practices

### 7. Documentation

**Ask Claude Code to:**

```
"Add docstrings to all functions in src/training/train.py."
```

```
"Update the README with instructions for the new inference API."
```

```
"Explain what the batch_job.sh script does."
```

Claude Code can:
- Write and update documentation
- Add code comments and docstrings
- Explain complex code sections
- Create markdown documentation

## Best Practices

### 1. Be Specific with Context

**Good:**
```
"I'm getting a CUDA out of memory error when training with batch size 16 and image size 1024.
Can you reduce the batch size in the training command?"
```

**Less helpful:**
```
"Fix the training error."
```

### 2. Ask for Explanations

Claude Code can help you understand the codebase:

```
"Explain how the dataset augmentation pipeline works in src/utils/dataset/."
```

```
"What's the difference between train.py and the batch_job.sh approach?"
```

### 3. Verify Before Committing

Always review changes Claude Code makes before committing:

```
"Show me a diff of the changes you made to train.py."
```

Then if satisfied:
```
"Looks good! Create a commit with these changes."
```

### 4. Use Claude Code for Exploration

When working with unfamiliar parts of the codebase:

```
"Search the codebase for where we handle bounding box predictions."
```

```
"Find all places where we use the YOLOv8 model configuration."
```

### 5. Iterative Development

Break complex tasks into steps:

```
"First, help me understand the current inference pipeline in streamlit_application.py."
```

Then:
```
"Now add a confidence threshold slider to the Streamlit UI."
```

### 6. Leverage Multi-Step Workflows

Claude Code can handle complex multi-step operations:

```
"I want to run a complete experiment: prepare the dataset, run a training with 100 epochs,
analyze the results, and if mAP is above 0.7, deploy the model to the Streamlit app."
```

## Project-Specific Tips

### Training on Great Lakes HPC

Claude Code understands the SLURM environment:

```
"Submit a batch job with EPOCHS=200 BATCH=4 IMGSZ=1024, then monitor the logs."
```

### Docker Workflows

Since this project uses Docker extensively:

```
"Rebuild the Docker image and restart the Streamlit container."
```

```
"Execute a Python script inside the pdf-ocr-devcontainer."
```

### Optuna Optimization

Claude Code can help with hyperparameter tuning:

```
"Start an Optuna study with 30 trials. After it completes, show me the best hyperparameters."
```

```
"Visualize the Optuna optimization history from the SQLite database."
```

### Pre-commit Hooks

This project has pre-commit hooks configured. Claude Code knows to run them:

```
"Format all Python files with black and isort, then commit."
```

## Understanding Claude Code's Capabilities

### What Claude Code CAN Do:

- Read and write files anywhere in the project
- Execute shell commands (with configured permissions)
- Search code using patterns and keywords
- Understand context from multiple files
- Make multi-file edits
- Run Python scripts and Docker commands
- Create commits and pull requests
- Navigate Jupyter notebooks
- Fetch web documentation

### What Claude Code CANNOT Do:

- Access external servers without credentials
- Install system packages (without appropriate permissions)
- Modify files outside the project directory
- Access private APIs without tokens
- Execute destructive commands without confirmation (unless pre-approved)

## Troubleshooting

### Claude Code seems stuck

Ask for status:
```
"What are you working on right now?"
```

### Too many file changes

Ask Claude Code to be more targeted:
```
"Only modify the training script, don't change other files."
```

### Wrong file modified

Be explicit about file paths:
```
"Edit src/training/train.py, not the one in the notebooks directory."
```

### Permission denied errors

Check [.claude/settings.local.json](../.claude/settings.local.json) and add the command pattern if needed.

### Claude Code made unwanted changes

Use git to revert:
```
"Revert the changes to streamlit_application.py."
```

Or manually:
```bash
git checkout -- src/web/streamlit_application.py
```

## Advanced Usage

### Custom Slash Commands

You can create custom slash commands in `.claude/commands/` for common tasks:

**Example: `.claude/commands/train.md`**
```markdown
Start a training run with default settings:
- 100 epochs
- batch size 16
- image size 640
- Cache enabled
Run the training and monitor progress.
```

Then use: `/train`

### Hooks

Configure hooks in `.claude/settings.local.json` to run commands automatically:
- Before tool execution
- After commits
- On file changes

See Claude Code documentation for hook examples.

### MCP Servers

Claude Code supports Model Context Protocol (MCP) servers for extended capabilities like:
- Database access
- API integrations
- Custom tools

Consult the Claude Code docs for MCP setup.

## Getting Help

- **Claude Code docs**: https://docs.claude.com/docs/claude-code
- **Report issues**: https://github.com/anthropics/claude-code/issues
- **Within Claude Code**: Type `/help` for quick reference

## Example Session

Here's an example of a complete training workflow with Claude Code:

```
You: "I want to run a training experiment with higher resolution images."

Claude: I'll help you run a high-resolution training experiment. Let me create a plan...
[Creates todo list, configures training parameters]

You: "Use 1280 image size and 150 epochs. Name it 'high-res-v1'."

Claude: [Executes training command with specified parameters]

You: "Monitor the training and let me know if there are any issues."

Claude: [Monitors logs, reports progress]

You: "Training completed! Analyze the results and compare to the baseline model."

Claude: [Parses results.csv, compares metrics, provides analysis]

You: "Great! Deploy this model to production."

Claude: [Copies best.pt to production directory, updates config, restarts container]

You: "Create a commit and PR for the training configuration changes."

Claude: [Creates commit, pushes branch, opens PR with summary]
```

---

**Happy coding with Claude Code!** Remember: Claude Code is a powerful assistant, but always review changes before committing, especially for critical paths like model training and production deployments.
