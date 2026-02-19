"""
Shield-Ryzen V2 — Adversarial Robustness Test Suite (TASK 4.3)
================================================================
Tests model robustness against gradient-based adversarial attacks.
Measures the confidence drop under increasing perturbation budgets.

Attacks Implemented:
  - FGSM (Fast Gradient Sign Method)   — single-step attack
  - PGD  (Projected Gradient Descent)  — multi-step iterative attack

Targets:
  - Clean accuracy baseline
  - FGSM ε=4/255 accuracy drop < 5%  (aspirational for non-AT models)
  - PGD  ε=4/255 accuracy drop (20 steps)

OUTPUT:
  - security/adversarial_robustness.json   — structured report
  - data/adversarial_test_set/             — saved adversarial images

Developer: Inayat Hussain | AMD Slingshot 2026
Part 4 of 14 — Neural Model Verification & Integrity
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

# Add project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from shield_xception import ShieldXception, load_model_with_verification

_log = logging.getLogger("AdversarialSuite")

# ═══════════════════════════════════════════════════════════════
#  ATTACK IMPLEMENTATIONS
# ═══════════════════════════════════════════════════════════════

class AdversarialAttacker:
    """Gradient-based adversarial attack engine.

    Works with ShieldXception which outputs softmax probabilities.
    Since we can't extract raw logits from the softmax output,
    we use -log(prob[true_class]) as our loss function, which is
    equivalent to NLL loss and provides valid gradients.
    """

    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device

    def fgsm_attack(
        self,
        image: torch.Tensor,
        true_label: int,
        epsilon: float,
    ) -> torch.Tensor:
        """Fast Gradient Sign Method (Goodfellow et al., 2014).

        Single-step attack: x_adv = x + ε * sign(∇_x L(x, y))

        Args:
            image:      Input tensor (1, 3, 299, 299) in [-1, 1]
            true_label: Ground-truth class index (0=Fake, 1=Real)
            epsilon:    Perturbation budget (in pixel space normalized to [-1,1])

        Returns:
            Adversarial image tensor clamped to [-1, 1]
        """
        image = image.clone().detach().requires_grad_(True)
        outputs = self.model(image.to(self.device))

        # NLL loss via softmax output
        loss = -torch.log(outputs[0, true_label] + 1e-10)

        self.model.zero_grad()
        loss.backward()

        data_grad = image.grad.data
        perturbed = image + epsilon * data_grad.sign()
        perturbed = torch.clamp(perturbed, -1.0, 1.0)
        return perturbed.detach()

    def pgd_attack(
        self,
        image: torch.Tensor,
        true_label: int,
        epsilon: float,
        alpha: float = 2 / 255,
        num_steps: int = 20,
    ) -> torch.Tensor:
        """Projected Gradient Descent (Madry et al., 2018).

        Multi-step attack with projection back to ε-ball:
          x_{t+1} = Π_{ε}(x_t + α * sign(∇_x L(x_t, y)))

        Args:
            image:      Input tensor (1, 3, 299, 299) in [-1, 1]
            true_label: Ground-truth class index (0=Fake, 1=Real)
            epsilon:    L∞ perturbation budget
            alpha:      Step size per iteration
            num_steps:  Number of PGD iterations

        Returns:
            Adversarial image tensor clamped to [-1, 1]
        """
        original = image.clone().detach()
        adv = image.clone().detach()

        for step in range(num_steps):
            adv.requires_grad_(True)
            outputs = self.model(adv.to(self.device))
            loss = -torch.log(outputs[0, true_label] + 1e-10)

            self.model.zero_grad()
            loss.backward()

            grad = adv.grad.data
            adv = adv.detach() + alpha * grad.sign()

            # Project perturbation back to ε-ball around original
            delta = torch.clamp(adv - original, -epsilon, epsilon)
            adv = torch.clamp(original + delta, -1.0, 1.0).detach()

        return adv


# ═══════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def _denormalize_and_save(tensor: torch.Tensor, path: str) -> None:
    """Convert [-1,1] normalized tensor back to BGR image and save."""
    img = tensor.cpu().squeeze(0).permute(1, 2, 0).numpy()
    img = (img * 0.5 + 0.5) * 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def _generate_synthetic_test_image() -> torch.Tensor:
    """Generate a synthetic test image when no real test face is available.

    Creates a gradient pattern that provides meaningful gradients
    for adversarial testing without requiring actual face images.
    """
    # Create synthetic face-like pattern (smooth gradients + noise)
    np.random.seed(42)
    img = np.random.randn(1, 3, 299, 299).astype(np.float32) * 0.1
    # Add smooth structure
    for c in range(3):
        y_grad = np.linspace(-0.5, 0.5, 299).reshape(1, 1, 299, 1)
        x_grad = np.linspace(-0.5, 0.5, 299).reshape(1, 1, 1, 299)
        img[:, c:c+1, :, :] += y_grad * 0.3 + x_grad * 0.3
    img = np.clip(img, -1.0, 1.0)
    return torch.from_numpy(img)


# ═══════════════════════════════════════════════════════════════
#  MAIN TEST FUNCTION
# ═══════════════════════════════════════════════════════════════

def test_adversarial_robustness(
    model_path: str = "ffpp_c23.pth",
    test_image_path: Optional[str] = None,
    output_dir: str = "data/adversarial_test_set/",
    report_path: str = "security/adversarial_robustness.json",
) -> dict:
    """Full adversarial robustness evaluation.

    Tests:
      1. Clean accuracy (baseline confidence)
      2. FGSM attacks at ε = {1/255, 2/255, 4/255, 8/255}
      3. PGD attack at ε = 4/255 with 20 steps

    Args:
        model_path:      Path to .pth checkpoint
        test_image_path: Path to a known REAL face image (optional)
        output_dir:      Directory to save adversarial images
        report_path:     Path for JSON report

    Returns:
        Report dict with all results
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{'=' * 60}")
    print(f"  ADVERSARIAL ROBUSTNESS TEST SUITE")
    print(f"  Device: {device}")
    print(f"{'=' * 60}")

    # ── Load Model ────────────────────────────────────────────
    try:
        state_dict = load_model_with_verification(model_path, device)
        model = ShieldXception().to(device)
        model.model.load_state_dict(state_dict, strict=False)
        model.eval()
        print("✅ Model loaded and verified")
    except Exception as e:
        error_msg = f"Failed to load model: {e}"
        print(f"❌ {error_msg}")
        return {"error": error_msg}

    attacker = AdversarialAttacker(model, device)

    # ── Prepare test image ────────────────────────────────────
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    if test_image_path and os.path.exists(test_image_path):
        print(f"Using test image: {test_image_path}")
        img_pil = Image.open(test_image_path).convert("RGB")
        input_tensor = transform(img_pil).unsqueeze(0).to(device)
        image_source = test_image_path
    else:
        print("No test image found — using synthetic gradient pattern")
        input_tensor = _generate_synthetic_test_image().to(device)
        image_source = "synthetic_gradient_pattern"

    # Ground truth: assume REAL (index 1)
    true_label = 1

    # ── Baseline ──────────────────────────────────────────────
    with torch.no_grad():
        base_probs = model(input_tensor)
        base_real = float(base_probs[0, 1])
        base_fake = float(base_probs[0, 0])

    base_correct = base_real > 0.5  # Is classified as REAL?
    print(f"\nBaseline: Real={base_real:.4f}  Fake={base_fake:.4f}  "
          f"Pred={'REAL' if base_correct else 'FAKE'}")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    # ── FGSM attacks ──────────────────────────────────────────
    fgsm_epsilons = [1/255, 2/255, 4/255, 8/255]
    fgsm_results = {}

    print(f"\n{'─' * 40}")
    print("FGSM Attack Results:")
    print(f"{'─' * 40}")

    for eps in fgsm_epsilons:
        eps_label = f"{int(eps * 255)}_255"
        adv = attacker.fgsm_attack(input_tensor.clone(), true_label, eps)

        with torch.no_grad():
            adv_probs = model(adv)
            adv_real = float(adv_probs[0, 1])

        drop = base_real - adv_real
        adv_correct = adv_real > 0.5

        fgsm_results[f"fgsm_eps_{eps_label}"] = {
            "epsilon": eps,
            "epsilon_label": f"{int(eps*255)}/255",
            "clean_confidence": round(base_real, 6),
            "adversarial_confidence": round(adv_real, 6),
            "confidence_drop": round(drop, 6),
            "still_correct": adv_correct,
        }

        status = "✅" if adv_correct else "❌"
        print(f"  ε={int(eps*255):2d}/255: {base_real:.4f} → {adv_real:.4f}  "
              f"(drop={drop:+.4f}) {status}")

        # Save adversarial image
        save_path = os.path.join(output_dir, f"fgsm_eps_{eps_label}.png")
        _denormalize_and_save(adv, save_path)

    # ── PGD attack ────────────────────────────────────────────
    pgd_eps = 4 / 255
    pgd_steps = 20

    print(f"\n{'─' * 40}")
    print(f"PGD Attack (ε=4/255, {pgd_steps} steps):")
    print(f"{'─' * 40}")

    adv_pgd = attacker.pgd_attack(
        input_tensor.clone(), true_label, pgd_eps, num_steps=pgd_steps
    )

    with torch.no_grad():
        pgd_probs = model(adv_pgd)
        pgd_real = float(pgd_probs[0, 1])

    pgd_drop = base_real - pgd_real
    pgd_correct = pgd_real > 0.5

    pgd_result = {
        "epsilon": pgd_eps,
        "epsilon_label": "4/255",
        "num_steps": pgd_steps,
        "alpha": 2 / 255,
        "clean_confidence": round(base_real, 6),
        "adversarial_confidence": round(pgd_real, 6),
        "confidence_drop": round(pgd_drop, 6),
        "still_correct": pgd_correct,
    }

    status = "✅" if pgd_correct else "❌"
    print(f"  PGD: {base_real:.4f} → {pgd_real:.4f}  "
          f"(drop={pgd_drop:+.4f}) {status}")

    save_path = os.path.join(output_dir, "pgd_eps_4_255.png")
    _denormalize_and_save(adv_pgd, save_path)

    # ── Build report ──────────────────────────────────────────
    # Assess overall robustness
    fgsm_4_drop = abs(fgsm_results.get("fgsm_eps_4_255", {}).get(
        "confidence_drop", 1.0
    ))
    target_met = fgsm_4_drop < 0.05

    report = {
        "test_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_path": model_path,
        "image_source": image_source,
        "device": str(device),
        "baseline": {
            "real_confidence": round(base_real, 6),
            "fake_confidence": round(base_fake, 6),
            "prediction": "REAL" if base_correct else "FAKE",
        },
        "fgsm_attacks": fgsm_results,
        "pgd_attack": pgd_result,
        "assessment": {
            "target": "FGSM ε=4/255 accuracy drop < 5%",
            "actual_drop_pct": round(fgsm_4_drop * 100, 2),
            "target_met": target_met,
            "note": (
                "Non-adversarially trained models are expected to be vulnerable "
                "to gradient attacks. System-level defenses (liveness, texture, "
                "temporal consistency) provide layered security."
            ),
        },
        "adversarial_images_dir": output_dir,
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"  FGSM ε=4/255 drop: {fgsm_4_drop*100:.1f}%  "
          f"(target: <5%)  {'PASS' if target_met else 'EXPECTED FAIL'}")
    print(f"  Report saved: {report_path}")
    print(f"  Images saved: {output_dir}")
    print(f"{'=' * 60}")

    return report


if __name__ == "__main__":
    test_adversarial_robustness(
        model_path="ffpp_c23.pth",
        test_image_path="tests/fixtures/frontal_face.jpg",
    )
