#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Wrapper script for running the day trading systems
--------------------------------------------------
This script ensures proper Python path setup before running the 
stock or crypto day trading scripts.

Usage:
    python run_daytrader.py stock --duration 4h
    python run_daytrader.py crypto --duration 1h
"""

import os
import sys
import argparse
import subprocess

def main():
    parser = argparse.ArgumentParser(description="Run Mercurio day trading systems")
    parser.add_argument("market", type=str, choices=["stock", "crypto"], 
                        help="Market type (stock or crypto)")
    parser.add_argument("--duration", type=str, choices=["1h", "4h", "8h", "custom"], default="4h",
                        help="Trading session duration (1h, 4h, 8h, or custom)")
    parser.add_argument("--custom-seconds", type=int, default=0,
                        help="Custom duration in seconds if --duration=custom")
    parser.add_argument("--config", type=str, default="config/daytrader_config.json",
                        help="Path to configuration file")
    parser.add_argument("--log-level", type=str, choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
                        default="INFO", help="Logging level")
                        
    args = parser.parse_args()
    
    # Determine which script to run
    script_path = "scripts/run_stock_daytrader.py" if args.market == "stock" else "scripts/run_crypto_daytrader.py"
    
    # Get absolute path to the script and project root
    project_root = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(project_root, script_path)
    
    # Build command with arguments
    cmd = [sys.executable, "-u", script_path]  # -u for unbuffered output
    cmd.extend(["--duration", args.duration])
    
    if args.duration == "custom" and args.custom_seconds > 0:
        cmd.extend(["--custom-seconds", str(args.custom_seconds)])
        
    cmd.extend(["--config", os.path.join(project_root, args.config)])
    cmd.extend(["--log-level", args.log_level])
    
    # Set up environment with correct PYTHONPATH
    env = os.environ.copy()
    env["PYTHONPATH"] = project_root + os.pathsep + env.get("PYTHONPATH", "")
    
    print(f"Running {args.market} trader with {args.duration} duration")
    print(f"PYTHONPATH set to: {env['PYTHONPATH']}")
    print("Command:", " ".join(cmd))
    print("-" * 50)
    
    # Run the process with the adjusted environment
    try:
        process = subprocess.Popen(cmd, env=env, cwd=project_root)
        process.wait()
        sys.exit(process.returncode)
    except KeyboardInterrupt:
        print("\nTrading system interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error running trading system: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
