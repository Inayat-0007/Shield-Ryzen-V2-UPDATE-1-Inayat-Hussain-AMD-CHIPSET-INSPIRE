"""Real-time audit snapshot — shows the LATEST 12 frames with full detail."""
import json, os

LOG_PATH = 'logs/shield_audit.jsonl'
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
    except:
        pass

with open('_audit_result.txt', 'w', encoding='utf-8') as out:
    out.write(f"REAL-TIME SNAPSHOT — {len(frame_entries)} total frames\n")
    out.write(f"Showing LAST 12 frame entries:\n")
    out.write("=" * 100 + "\n\n")
    
    for entry in frame_entries[-12:]:
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
            face_alert = adv.get('face_alert', '')
            
            # Screen replay physics breakdown
            if dist > 40:
                ref_tex = 130.0
                ref_dist = 50.0
                max_expected = ref_tex * (ref_dist / dist) ** 2
                max_allowed = max_expected * 2.5
                physics_violation = tex / max_allowed if max_allowed > 0 else 0
            else:
                max_expected = 0
                max_allowed = 0
                physics_violation = 0
            
            flags = []
            if screen:
                flags.append("SCREEN_REPLAY")
            if ear_anom:
                flags.append("EAR_ANOMALY")
            if face_alert:
                flags.append(f"ALERT:{face_alert}")
            flag_str = " | ".join(flags) if flags else "CLEAN"
            
            out.write(f"STATE: {state:12s} | FLAGS: {flag_str}\n")
            out.write(f"  Neural Conf:   {nc:.3f} (raw trust score)\n")
            out.write(f"  EAR:           {ear:.4f} ({ear_rel})\n")
            out.write(f"  Blinks:        {blinks} (src={blink_src}, pattern={pattern:.0%})\n")
            out.write(f"  Distance:      {dist:.0f} cm\n")
            out.write(f"  Head Pose:     Yaw={yaw:+.1f}  Pitch={pitch:+.1f}  Roll={roll:+.1f}\n")
            out.write(f"  Face Age:      {age:.1f}s\n")
            out.write(f"  Tiers:         [{tiers[0]:5s}, {tiers[1]:5s}, {tiers[2]:5s}]\n")
            out.write(f"  Texture:       {tex:.1f}\n")
            if dist > 40:
                out.write(f"  Physics Check: max_expected={max_expected:.1f} | max_allowed={max_allowed:.1f} | actual={tex:.1f} | ratio={physics_violation:.2f}x\n")
                if physics_violation > 1.0:
                    out.write(f"                 >>> VIOLATION: {physics_violation:.1f}x over physics limit <<<\n")
            out.write(f"  Tex Detail:    {tex_explain[:140]}\n")
            for p in fr.get('plugin_votes', []):
                pn = p.get('name', '?')
                pv = p.get('verdict', '?')
                pc = p.get('confidence', 0)
                out.write(f"    {pn:24s} -> {pv:10s} ({pc:.2f})\n")
            out.write(f"  FPS: {fps:.1f} | Latency: {total_ms:.0f}ms\n")
            out.write("-" * 100 + "\n")

print("Done")
