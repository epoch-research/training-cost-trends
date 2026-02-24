#!/usr/bin/env python3
"""
Backward-compatible entry point for the training cost analysis.

The actual implementation has been moved to the training_cost_trends/ package.
See training_cost_trends/ for the modular source code.
"""

from training_cost_trends.__main__ import cli

if __name__ == "__main__":
    cli()
