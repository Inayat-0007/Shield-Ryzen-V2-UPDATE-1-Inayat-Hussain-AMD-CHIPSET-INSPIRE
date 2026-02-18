"""
Shield-Ryzen V2 -- Adversarial Robustness Test Suite
=====================================================
Tests model robustness against FGSM and PGD attacks.
Ensures the model is not brittle to small perturbations.

Attacks:
  - FGSM (Fast Gradient Sign Method)
  - PGD (Projected Gradient Descent)

Developer: Inayat Hussain | AMD Slingshot 2026
Part 4 of 12 -- Neural Model Verification
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shield_xception import ShieldXception, load_model_with_verification

class AdversarialAttacker:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.loss_fn = nn.CrossEntropyLoss()

    def fgsm_attack(self, image, label, epsilon):
        """Fast Gradient Sign Method"""
        image.requires_grad = True
        outputs = self.model(image.to(self.device)) # Output is Softmax
        # Convert softmax to log_softmax for NLLLoss or use CrossEntropyWithLogits
        # ShieldXception outputs softmax. So we need to take log.
        # But CrossEntropyLoss expects logits (unnormalized).
        # We need to access the logits from the model?
        # ShieldXception forward returns: torch.softmax(logits, dim=1)
        # We can't easily invert softmax numerically to get exact logits for gradients.
        
        # FIX: We need valid gradients. Minimizing prob of true class is equivalent
        # to maximizing loss.
        # Loss = -log(prob[true_class])
        
        loss = -torch.log(outputs[0, label] + 1e-10)
        
        self.model.zero_grad()
        loss.backward()
        
        data_grad = image.grad.data
        sign_data_grad = data_grad.sign()
        perturbed_image = image + epsilon * sign_data_grad
        perturbed_image = torch.clamp(perturbed_image, -1, 1) # Valid range [-1, 1]
        return perturbed_image

    def pgd_attack(self, image, label, epsilon, alpha=2/255, iters=10):
        """Projected Gradient Descent"""
        original_image = image.clone().detach()
        adv_image = image.clone().detach()
        
        for i in range(iters):
            adv_image.requires_grad = True
            outputs = self.model(adv_image.to(self.device))
            loss = -torch.log(outputs[0, label] + 1e-10)
            
            self.model.zero_grad()
            loss.backward()
            
            adv_image_grad = adv_image.grad.data
            adv_image = adv_image + alpha * adv_image_grad.sign()
            
            # Projection
            eta = torch.clamp(adv_image - original_image, -epsilon, epsilon)
            adv_image = torch.clamp(original_image + eta, -1, 1)
            adv_image = adv_image.detach()
            
        return adv_image

def run_adversarial_robustness(
    model_path: str,
    test_image_path: str, # Use a known 'Real' face
    output_dir: str = "data/adversarial_test_set/"
) -> dict:
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running adversarial tests on {device}...")
    
    # Load Model (Securely)
    try:
        state_dict = load_model_with_verification(model_path, device)
        model = ShieldXception().to(device)
        model.model.load_state_dict(state_dict, strict=False)
        model.eval()
    except Exception as e:
        print(f"Failed to load model: {e}")
        return {"error": str(e)}
        
    attacker = AdversarialAttacker(model, device)
    
    # Prepare Image
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    if not os.path.exists(test_image_path):
        print(f"Test image not found: {test_image_path}")
        return {"error": "Test image missing"}
        
    img_pil = Image.open(test_image_path).convert('RGB')
    params = img_pil.size
    input_tensor = transform(img_pil).unsqueeze(0).to(device) # (1, 3, 299, 299)
    
    # Assume the test image is REAL (Label 1)
    # ffpp_c23: 0=Fake, 1=Real
    true_label = 1
    
    # Baseline Prediction
    with torch.no_grad():
        base_probs = model(input_tensor)
        base_real_conf = base_probs[0, 1].item()
    
    print(f"Baseline Real Confidence: {base_real_conf:.4f}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    results = {
        "baseline_confidence": base_real_conf,
        "attacks": {}
    }
    
    # --- FGSM Attacks ---
    epsilons = [0.01, 0.03, 0.07] # 0.01 is subtle, 0.07 is visible noise
    
    for eps in epsilons:
        adv_tensor = attacker.fgsm_attack(input_tensor.clone(), true_label, eps)
        with torch.no_grad():
            adv_probs = model(adv_tensor)
            adv_real_conf = adv_probs[0, 1].item()
            
        print(f"FGSM eps={eps}: Conf {base_real_conf:.4f} -> {adv_real_conf:.4f}")
        
        results["attacks"][f"fgsm_eps_{eps}"] = {
            "confidence": adv_real_conf,
            "drop": base_real_conf - adv_real_conf
        }
        
        # Save Adversarial Image
        # Denormalize: x * 0.5 + 0.5 -> [0, 1] -> [0, 255]
        adv_img = adv_tensor.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
        adv_img = (adv_img * 0.5 + 0.5) * 255
        adv_img = np.clip(adv_img, 0, 255).astype(np.uint8)
        # RGB to BGR for OpenCV save
        cv2.imwrite(f"{output_dir}/fgsm_eps_{eps}.jpg", cv2.cvtColor(adv_img, cv2.COLOR_RGB2BGR))

    # --- PGD Attack ---
    pgd_eps = 0.03
    adv_tensor_pgd = attacker.pgd_attack(input_tensor.clone(), true_label, pgd_eps)
    with torch.no_grad():
        pgd_probs = model(adv_tensor_pgd)
        pgd_real_conf = pgd_probs[0, 1].item()
        
    print(f"PGD eps={pgd_eps}: Conf {base_real_conf:.4f} -> {pgd_real_conf:.4f}")
    results["attacks"][f"pgd_eps_{pgd_eps}"] = {
        "confidence": pgd_real_conf,
        "drop": base_real_conf - pgd_real_conf
    }
    
    # Save PGD Image
    adv_img = adv_tensor_pgd.cpu().squeeze(0).permute(1, 2, 0).numpy()
    adv_img = (adv_img * 0.5 + 0.5) * 255
    adv_img = np.clip(adv_img, 0, 255).astype(np.uint8)
    cv2.imwrite(f"{output_dir}/pgd_eps_{pgd_eps}.jpg", cv2.cvtColor(adv_img, cv2.COLOR_RGB2BGR))

    # Save JSON Report
    import json
    report_path = "security/adversarial_robustness_report.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"✅ Adversarial report saved to {report_path}")

    return results

if __name__ == "__main__":
    run_adversarial_robustness(
        "ffpp_c23.pth",
        "tests/fixtures/frontal_face.jpg"
    )
