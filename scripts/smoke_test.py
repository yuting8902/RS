#!/usr/bin/env python3
"""No H100, no real weights: run deterministic algorithm tests with pytest."""
import sys
import subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[1]
raise SystemExit(subprocess.call([sys.executable,'-m','pytest','-q',str(root/'tests')],cwd=root))
