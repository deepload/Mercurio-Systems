#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to run database migrations for the Mercurio Edge subscription system.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))


def run_migrations(action='upgrade', revision='head'):
    """
    Run alembic migrations.
    
    Args:
        action: Action to perform ('upgrade', 'downgrade', 'revision', 'show')
        revision: Revision identifier (e.g., 'head', 'base', specific revision ID)
    """
    # Set environment variables
    os.environ['PYTHONPATH'] = str(project_root)
    
    # Construct alembic command
    if action == 'show':
        cmd = ['alembic', 'current']
    elif action == 'history':
        cmd = ['alembic', 'history']
    elif action == 'revision':
        cmd = ['alembic', 'revision', '--autogenerate', '-m', revision]
    else:
        cmd = ['alembic', action, revision]
    
    # Run the command
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(project_root))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run database migrations for Mercurio Edge")
    parser.add_argument(
        "action", 
        choices=['upgrade', 'downgrade', 'revision', 'show', 'history'],
        help="Migration action to perform"
    )
    parser.add_argument(
        "--revision", 
        default="head",
        help="Revision identifier (default: 'head')"
    )
    
    args = parser.parse_args()
    run_migrations(args.action, args.revision)
