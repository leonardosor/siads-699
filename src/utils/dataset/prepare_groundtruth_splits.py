#!/usr/bin/env python3
"""
BACKWARD COMPATIBILITY WRAPPER

This script is now a wrapper around prepare_dataset.py
Use the new unified script instead:
  python src/utils/dataset/prepare_dataset.py groundtruth [options]

This wrapper maintains backward compatibility for existing workflows.
"""

import subprocess
import sys
from pathlib import Path


def main() -> None:
    """Wrapper that calls the unified prepare_dataset.py script"""
    print("=" * 70)
    print("NOTE: This script is now a wrapper for backward compatibility")
    print("Consider using: python src/utils/dataset/prepare_dataset.py groundtruth")
    print("=" * 70)
    print()

    # Build command to call the new unified script
    script_dir = Path(__file__).resolve().parent
    unified_script = script_dir / "prepare_dataset.py"

    # Convert arguments to pass to unified script
    args = ["groundtruth"] + sys.argv[1:]

    # Execute the unified script
    result = subprocess.run(
        [sys.executable, str(unified_script)] + args,
        cwd=script_dir.parent.parent.parent
    )
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()