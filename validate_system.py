"""
Shield-Ryzen V2 — System Validation Manager (TASK 12.1)
=======================================================
Orchestrates the full test suite and generates a compliance report.
Ensures all subsystems (Engine, HUD, Security, AMD NPU) are green.

Usage: python validate_system.py

Developer: Inayat Hussain | AMD Slingshot 2026
Part 12 of 14 — Comprehensive Validation
"""

import subprocess
import sys
import os
import re
from datetime import datetime

def run_validation():
    print("Starting Shield-Ryzen V2 System Validation...")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("-" * 60)
    
    # Run Pytest
    cmd = [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"]
    
    start_time = datetime.now()
    result = subprocess.run(cmd, capture_output=True, text=True)
    duration = datetime.now() - start_time
    
    print(result.stdout)
    if result.stderr:
        print("ERRORS:")
        print(result.stderr)
        
    print("-" * 60)
    
    # Parse Output
    passed = len(re.findall(r"PASSED", result.stdout))
    failed = len(re.findall(r"FAILED", result.stdout))
    errors = len(re.findall(r"ERROR", result.stdout))
    warnings = len(re.findall(r"WARNING", result.stdout))
    
    status = "SUCCESS" if (failed == 0 and errors == 0) else "FAILURE"
    
    # Generate Report
    report = f"""# Shield-Ryzen V2 Validation Report

**Date:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Status:** {status}
**Duration:** {duration.total_seconds():.2f}s

## Summary
- **Total Tests:** {passed + failed + errors}
- **Passed:** {passed}
- **Failed:** {failed}
- **Errors:** {errors}

## Details
### Execution Log
```
{result.stdout[-2000:] if result.stdout else "No output"}
```

## Failures (if any)
```
{result.stderr if failed > 0 else "None"}
```
"""
    
    with open("docs/VALIDATION_REPORT.md", "w") as f:
        f.write(report)
        
    print(f"Validation Complete. Status: {status}")
    print(f"Report saved to docs/VALIDATION_REPORT.md")
    
    return result.returncode

if __name__ == "__main__":
    sys.exit(run_validation())
