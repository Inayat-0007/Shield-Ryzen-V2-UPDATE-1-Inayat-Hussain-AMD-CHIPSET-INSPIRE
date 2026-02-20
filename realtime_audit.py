"""
Shield-Ryzen V2 — Real-Time Audit Monitor
==========================================
Tails the audit log and prints live detection results.
Press Ctrl+C to stop.
"""
import json
import time
import os
import sys

LOG_PATH = "logs/shield_audit.jsonl"

def colorize(state):
    """ANSI color codes for terminal output."""
    colors = {
        "REAL":       "\033[92m",  # Green
        "VERIFIED":   "\033[96m",  # Cyan
        "WAIT_BLINK": "\033[93m",  # Yellow
        "SUSPICIOUS": "\033[93m",  # Yellow
        "HIGH_RISK":  "\033[91m",  # Red
        "FAKE":       "\033[91m",  # Red
        "CRITICAL":   "\033[95m",  # Magenta
        "UNKNOWN":    "\033[90m",  # Gray
    }
    reset = "\033[0m"
    c = colors.get(state, "\033[37m")
    return f"{c}{state:12s}{reset}"

def format_tier(tier):
    if tier in ("REAL", "PASS"):
        return f"\033[92m{tier}\033[0m"
    elif tier == "FAIL" or tier == "FAKE":
        return f"\033[91m{tier}\033[0m"
    return tier

print("=" * 70)
print("  SHIELD-RYZEN V2 — REAL-TIME AUDIT MONITOR")
print("  Watching:", LOG_PATH)
print("  Press Ctrl+C to stop")
print("=" * 70)
print()

# Seek to end of file
if not os.path.exists(LOG_PATH):
    print(f"[ERROR] Log file not found: {LOG_PATH}")
    sys.exit(1)

with open(LOG_PATH, "r") as f:
    # Go to end
    f.seek(0, 2)
    frame_count = 0
    
    try:
        while True:
            line = f.readline()
            if not line:
                time.sleep(0.05)
                continue
            
            line = line.strip()
            if not line:
                continue
                
            try:
                d = json.loads(line)
                data = d.get("data", {})
                
                if "face_results" not in data:
                    continue
                
                frame_count += 1
                faces = data.get("face_results", [])
                fps = data.get("fps", 0)
                timing = data.get("timing_breakdown", data.get("timing", {}))
                
                # Print every 3rd frame to avoid flooding
                if frame_count % 3 != 0:
                    continue
                
                n_faces = data.get("faces_detected", len(faces))
                total_ms = timing.get("total_ms", 0)
                infer_ms = timing.get("infer_ms", 0)
                
                print(f"\033[90m─── Frame #{frame_count:04d} │ {n_faces} face(s) │ {fps:.1f} FPS │ {total_ms:.0f}ms total │ {infer_ms:.0f}ms infer ───\033[0m")
                
                for fr in faces:
                    state = fr.get("state", "?")
                    fid = fr.get("face_id", "?")
                    neural = fr.get("neural_confidence", 0)
                    ear = fr.get("ear_value", 0)
                    ear_rel = fr.get("ear_reliability", "?")
                    tex = fr.get("texture_score", 0)
                    tiers = fr.get("tier_results", ["?","?","?"])
                    plugins = fr.get("plugin_votes", [])
                    adv = fr.get("advanced_info", {})
                    
                    # Main verdict line
                    t1_str = format_tier(str(tiers[0])) if len(tiers) > 0 else "?"
                    t2_str = format_tier(str(tiers[1])) if len(tiers) > 1 else "?"
                    t3_str = format_tier(str(tiers[2])) if len(tiers) > 2 else "?"
                    
                    blinks = adv.get("blinks", 0)
                    dist = adv.get("distance_cm", 0)
                    face_age = adv.get("face_age_s", 0)
                    head_pose = adv.get("head_pose", {})
                    yaw = head_pose.get("yaw", 0)
                    pitch = head_pose.get("pitch", 0)
                    pattern = adv.get("blink_pattern_score", 0)
                    
                    print(f"  Face #{fid} → {colorize(state)} │ Neural: {neural:.3f} │ EAR: {ear:.4f} ({ear_rel}) │ Tex: {tex:.1f}")
                    print(f"           Tiers: [{t1_str}, {t2_str}, {t3_str}] │ Blinks: {blinks} (pat:{pattern:.0%}) │ Dist: {dist:.0f}cm │ Age: {face_age:.0f}s │ Pose: Y{yaw:+.0f} P{pitch:+.0f}")
                    
                    # Plugin summary (compact)
                    plugin_parts = []
                    for p in plugins:
                        name = p.get("name", "?")
                        verdict = p.get("verdict", "?")
                        # Short name
                        short = name.replace("_", "")[:8]
                        if verdict == "REAL":
                            plugin_parts.append(f"\033[92m{short}✓\033[0m")
                        elif verdict == "FAKE":
                            plugin_parts.append(f"\033[91m{short}✗\033[0m")
                        else:
                            plugin_parts.append(f"\033[90m{short}?\033[0m")
                    
                    if plugin_parts:
                        print(f"           Plugins: {' │ '.join(plugin_parts)}")
                    print()
                    
            except json.JSONDecodeError:
                pass
            except Exception as e:
                print(f"[Parse Error] {e}")
                
    except KeyboardInterrupt:
        print("\n[MONITOR] Stopped by user.")
