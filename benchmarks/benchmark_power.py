
import sys
import os
import time
import json
import psutil
import platform

def measure_power(duration_seconds: int = 15) -> dict:
    """
    Measures system power draw during inference vs idle.
    Uses psutil battery status if available.
    """
    print(f"Running Power Benchmark ({duration_seconds}s)...")
    
    # 1. Idle Measurement (Stub)
    # We can't really isolate idle vs inference without hardware meter
    # But we can check battery discharge rate if on laptop
    
    battery = psutil.sensors_battery()
    
    if not battery:
        print("⚠️ No battery detected. Cannot measure power via discharge rate.")
        report = {
            "measured": False,
            "reason": "Power measurement not available on AC power without external meter.",
            "note": "The claim of 'extreme battery efficiency' is specific to Ryzen AI laptops on battery."
        }
    else:
        # Initial check
        p_plugged = battery.power_plugged
        if p_plugged:
            print("⚠️ System plugged in. Discharge rate irrelevant.")
            report = {
                "measured": False,
                "reason": "System is plugged in.",
                "note": "Unplug to measure battery drain."
            }
        else:
            # Measure drain? This takes hours to be accurate.
            # Short test is unreliable.
            # But let's try reading discharge rate if OS exposes it?
            # Windows usually doesn't expose strict wattage via psutil easily.
            # We'll stick to the honest fallback for this script.
            pass
            
            report = {
               "idle_watts": "Unknown",
               "inference_watts": "Unknown",
               "delta_watts": "Unknown",
               "measurement_method": "Battery Discharge Rate (Not reliable in short test)",
               "measured": False 
            }

    out_path = "benchmarks/power_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
        
    print(f"✅ Power Benchmark Report saved to {out_path}")
    return report

if __name__ == "__main__":
    measure_power()
