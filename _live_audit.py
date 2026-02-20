"""
Shield-Ryzen V2 — REAL-TIME AUDIT MONITOR
Run alongside prototype: python _live_audit.py
"""
import json, time, os, sys

AUDIT = os.path.join(os.path.dirname(__file__), "logs", "shield_audit.jsonl")

def c(t, code): return f"\033[{code}m{t}\033[0m"

def get_val(face, *keys):
    """Search for a key in face top-level and advanced_info."""
    for k in keys:
        if k in face and face[k] is not None:
            return face[k]
        adv = face.get("advanced_info", {}) or {}
        if k in adv and adv[k] is not None:
            return adv[k]
    return None

def main():
    print(c("=" * 100, "1;36"))
    print(c("  SHIELD-RYZEN V2 — LIVE AUDIT MONITOR (Ctrl+C to stop)", "1;33"))
    print(c("=" * 100, "1;36"))
    
    while not os.path.exists(AUDIT):
        print(c("Waiting for audit log...", "33"), end="\r")
        time.sleep(0.5)
    
    with open(AUDIT, 'r') as f:
        f.seek(0, 2)
        pos = f.tell()
    
    n = 0
    while True:
        try:
            with open(AUDIT, 'r') as f:
                f.seek(pos)
                lines = f.readlines()
                pos = f.tell()
            
            for line in lines:
                if not line.strip(): continue
                try:
                    d = json.loads(line.strip())
                    data = d.get("data", {}) or {}
                    faces = data.get("face_results", []) or []
                    
                    for face in faces:
                        n += 1
                        adv = face.get("advanced_info", {}) or {}
                        
                        fid = get_val(face, "face_id", "tracker_id") or "?"
                        state = get_val(face, "state") or "?"
                        conf = get_val(face, "confidence") or 0
                        tex = get_val(face, "texture_score") or 0
                        sr = get_val(face, "screen_replay") or False
                        ear = get_val(face, "ear") or 0
                        blinks = get_val(face, "blinks", "blink_count") or 0
                        dist = get_val(face, "distance_cm") or 0
                        bsrc = get_val(face, "blink_source") or ""
                        bpat = get_val(face, "blink_pattern_score") or ""
                        alert = get_val(face, "face_alert") or ""
                        age = get_val(face, "face_age_s") or 0
                        tex_exp = get_val(face, "texture_explain") or ""
                        hp = get_val(face, "head_pose") or {}
                        yaw = hp.get("yaw", 0) if isinstance(hp, dict) else 0
                        pitch = hp.get("pitch", 0) if isinstance(hp, dict) else 0
                        
                        # Tiers
                        tiers = get_val(face, "tiers") or {}
                        t1 = tiers.get("neural", "?") if isinstance(tiers, dict) else "?"
                        t2 = tiers.get("liveness", "?") if isinstance(tiers, dict) else "?"
                        t3 = tiers.get("forensic", "?") if isinstance(tiers, dict) else "?"
                        
                        # State coloring
                        sc = {"VERIFIED": "1;42;30", "REAL": "1;32", "SUSPICIOUS": "1;43;30",
                              "FAKE": "1;41;37", "CRITICAL": "1;41;5;37"}.get(state, "37")
                        state_s = c(f" {state:^12s} ", sc)
                        
                        # Confidence color
                        cc = "32" if conf >= 0.8 else ("33" if conf >= 0.5 else "31")
                        
                        # Tier colors
                        def tc(v):
                            if v in ("REAL","PASS"): return c(v,"32")
                            if v in ("FAKE","FAIL"): return c(v,"31")
                            return c(v,"33")
                        
                        sr_s = c("!!SCREEN!!", "1;31") if sr else c("CLEAN", "32")
                        
                        # LINE 1: Main metrics
                        print(
                            f"{c(f'#{n:5d}','36')} "
                            f"F{fid:<2} "
                            f"{state_s} "
                            f"Conf:{c(f'{conf:.1%}',cc)} "
                            f"EAR:{ear:.3f} "
                            f"Bk:{blinks:<3} "
                            f"D:{dist:>3.0f}cm "
                            f"Tex:{tex:>6.0f} "
                            f"[{tc(t1)} {tc(t2)} {tc(t3)}] "
                            f"{sr_s}"
                        )
                        
                        # LINE 2: Detail (dimmed)
                        det = f"       Y:{yaw:+.0f} P:{pitch:+.0f} Age:{age:.0f}s Src:{bsrc}"
                        if "moire=" in tex_exp:
                            try: det += f" M:{tex_exp.split('moire=')[1].split('|')[0].strip()}"
                            except: pass
                        if "light=" in tex_exp:
                            try: det += f" L:{tex_exp.split('light=')[1].split('|')[0].strip()}"
                            except: pass
                        if "SCREEN_REPLAY" in tex_exp:
                            det += c(" !! SCREEN_REPLAY !!", "1;31")  
                        if alert:
                            det += c(f" !! {alert} !!", "1;31")
                        print(c(det, "90"))
                        
                except Exception:
                    pass
            
            time.sleep(0.1)
        except KeyboardInterrupt:
            print(c("\nMonitor stopped.", "33"))
            break
        except:
            time.sleep(0.5)

if __name__ == "__main__":
    main()
