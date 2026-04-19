"""Start the Unified AI Gateway on port 4000."""
import subprocess, sys
from pathlib import Path

subprocess.run([
    sys.executable, "-m", "uvicorn",
    "middleware.app:app",
    "--host", "0.0.0.0",
    "--port", "4000",
    "--reload",
], cwd=Path(__file__).parent)
