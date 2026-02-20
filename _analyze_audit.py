"""Quick audit analyzer — reads last N frame entries from the log."""
import json, sys, os

LOG_PATH = 'logs/shield_audit.jsonl'
if not os.path.exists(LOG_PATH):
    print(f"[ERROR] Log file not found: {LOG_PATH}")
    sys.exit(1)

lines = open(LOG_PATH, encoding='utf-8', errors='replace').readlines()
frame_entries = []
for l in lines:
    l = l.strip()
    if not l:
        continue
    try:
        e = json.loads(l)
        if e.get('data', {}).get('face_results'):
            frame_entries.append(e)
    except Exception:
        pass

print(f"Total frame entries: {len(frame_entries)}")
print("=" * 80)

# Last 5 frame entries
for entry in frame_entries[-5:]:
    d = entry['data']
    fps = d.get('fps', 0)
    timing = d.get('timing_breakdown', d.get('timing', {}))
    total_ms = timing.get('total_ms', 0)
    for fr in d.get('face_results', []):
        state = fr.get('state', '?')
        nc = fr.get('neural_confidence', 0)
        ear = fr.get('ear_value', 0)
        ear_rel = fr.get('ear_reliability', '?')
        tex = fr.get('texture_score', 0)
        tex_explain = fr.get('texture_explanation', '')
        tiers = fr.get('tier_results', ['?', '?', '?'])
        adv = fr.get('advanced_info', {})
        blinks = adv.get('blinks', 0)
        dist = adv.get('distance_cm', 0)
        age = adv.get('face_age_s', 0)
        blink_src = adv.get('blink_source', '?')
        head_pose = adv.get('head_pose', {})
        yaw = head_pose.get('yaw', 0)
        pitch = head_pose.get('pitch', 0)
        roll = head_pose.get('roll', 0)
        pattern = adv.get('blink_pattern_score', 0)
        screen = adv.get('screen_replay', False)
        ear_anom = adv.get('ear_anomaly', False)
        pvotes = [(p.get('name', '?'), p.get('verdict', '?'), p.get('confidence', 0))
                  for p in fr.get('plugin_votes', [])]
        
        screen_label = " ** SCREEN REPLAY **" if screen else ""
        ear_anom_label = " [EAR ANOMALY]" if ear_anom else ""
        
        print(f"STATE={state:12s} | Neural={nc:.3f} | EAR={ear:.4f} ({ear_rel:6s}) | Tex={tex:.1f}{screen_label}{ear_anom_label}")
        print(f"  Blinks={blinks} (src={blink_src}, pat={pattern:.0%}) | Dist={dist:.0f}cm | Age={age:.1f}s")
        print(f"  Head: Yaw={yaw:+.1f} Pitch={pitch:+.1f} Roll={roll:+.1f}")
        print(f"  Tiers: [{tiers[0]}, {tiers[1]}, {tiers[2]}]")
        # Print texture explanation - truncate if long
        tex_short = tex_explain[:100] if len(tex_explain) > 100 else tex_explain
        print(f"  Texture: {tex_short}")
        for pname, pverdict, pconf in pvotes:
            print(f"    Plugin: {pname:20s} -> {pverdict:10s} (conf={pconf:.2f})")
        print(f"  FPS={fps:.1f} | Total={total_ms:.0f}ms")
        print("-" * 80)
