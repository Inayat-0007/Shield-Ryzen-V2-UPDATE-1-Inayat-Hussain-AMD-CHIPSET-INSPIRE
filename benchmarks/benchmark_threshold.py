
import sys
import os
import json
import numpy as np
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

def find_optimal_thresholds(y_true, y_scores):
    """
    Find thresholds for optimal F1, FAR=0.1%, FAR=1.0%.
    """
    # Sort by score descending
    desc_score_indices = np.argsort(y_scores)[::-1]
    y_scores_sorted = np.array(y_scores)[desc_score_indices]
    y_true_sorted = np.array(y_true)[desc_score_indices]
    
    tps = np.cumsum(y_true_sorted)
    fps = np.cumsum(1 - y_true_sorted)
    
    precision = tps / (tps + fps)
    recall = tps / tps[-1]
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    # Optimal F1
    best_f1_idx = np.argmax(f1)
    best_f1_thresh = y_scores_sorted[best_f1_idx]
    
    # FAR Targets (FAR = FPR)
    fpr = fps / fps[-1]
    
    # FAR = 0.001 (0.1%)
    idx_01 = np.where(fpr <= 0.001)[0]
    thresh_far_01 = y_scores_sorted[idx_01[-1]] if len(idx_01) > 0 else 1.0

    # FAR = 0.01 (1.0%)
    idx_10 = np.where(fpr <= 0.01)[0]
    thresh_far_10 = y_scores_sorted[idx_10[-1]] if len(idx_10) > 0 else 1.0

    return {
        "optimal_f1": float(best_f1_thresh),
        "f1_score": float(f1[best_f1_idx]),
        "threshold_far_0.1%": float(thresh_far_01),
        "threshold_far_1.0%": float(thresh_far_10)
    }

def run_threshold_analysis(mock=True):
    # Simulated data (Real: 0, Fake: 1)
    # y_scores = predict_proba(fake)
    if mock:
        print("Running Threshold Analysis (MOCK Mode)...")
        y_true = np.concatenate([np.zeros(1000), np.ones(1000)])
        # Real -> Low Fake prob (0.0-0.3)
        # Fake -> High Fake prob (0.7-1.0)
        # Some overlap (0.4-0.6)
        scores_real = np.random.uniform(0.0, 0.4, 1000)
        scores_fake = np.random.uniform(0.6, 1.0, 1000)
        # Add noise/hard samples
        scores_real[:50] = np.random.uniform(0.4, 0.8, 50) # Hard Reals
        scores_fake[:50] = np.random.uniform(0.2, 0.6, 50) # Hard Fakes
        
        y_scores = np.concatenate([scores_real, scores_fake])
    else:
        # Load from dataset
        pass
        
    metrics = find_optimal_thresholds(y_true, y_scores)
    
    report = {
        "optimal_thresholds": metrics,
        "current_arbitrary_threshold": 0.89, # From v2 config
        "recommendation": (
            f"Switch to {metrics['optimal_f1']:.2f} for balanced F1, "
            f"or {metrics['threshold_far_0.1%']:.2f} for high security."
        )
    }
    
    out_path = "benchmarks/threshold_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
        
    print(f"✅ Threshold Analysis Complete. Saved to {out_path}")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    run_threshold_analysis(mock=True)
