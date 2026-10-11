#!/usr/bin/env python3
"""Convenience entry point for the bundled Ada megakernel launcher."""
from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).resolve().parent / "megakernel" / "serve_ada.py"),
               run_name="__main__")
