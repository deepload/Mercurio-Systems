#!/usr/bin/env python
"""
Simple wrapper script to run the day trader with correct Python path
"""
import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Now import and run the main function from run_daytrader
from scripts.run_daytrader import main
import asyncio

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
