"""Quick debug script to read last N audit log entries cleanly."""
import json
import sys

N = int(sys.argv[1]) if len(sys.argv) > 1 else 10

with open('logs/shield_audit.jsonl') as f:
    lines = f.readlines()

for line in lines[-N:]:
    d = json.loads(line)
    if d.get('face_results'):
        for fr in d['face_results']:
            pvotes = ' | '.join([
                f"{v['name'][:8]}={v['verdict']}" 
                for v in fr.get('plugin_votes', [])
            ])
            t = fr.get('tier_results', ['?','?','?'])
            adv = fr.get('advanced_info', {})
            nc = fr['neural_confidence']
            ear = fr['ear_value']
            ear_r = fr['ear_reliability']
            tex = fr['texture_score']
            blinks = adv.get('blinks', 0)
            dist = adv.get('distance_cm', 0)
            state = fr['state']
            fps = d['fps']
            
            print(f"FPS={fps:5.1f} | {state:12s} | Neur={nc:.3f} | "
                  f"EAR={ear:.3f}({ear_r:3s}) | Tex={tex:6.1f} | "
                  f"Blinks={blinks} | Dist={dist:4.0f}cm | "
                  f"T=[{t[0]},{t[1]},{t[2]}] | {pvotes}")
    else:
        print(f"FPS={d.get('fps',0):.1f} | NO FACES")
