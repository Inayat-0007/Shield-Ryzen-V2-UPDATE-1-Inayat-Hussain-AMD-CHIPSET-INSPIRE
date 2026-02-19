"""
Shield-Ryzen V2 — Knowledge Distillation (TASK 11.3)
===================================================
Implementation of Teacher-Student training loop.
Distills the large XceptionNet (Teacher) into a compact MobileNetV3 (Student)
for ultra-low latency on older CPUs.

Teacher: XceptionNet (88MB FP32) -> Output: Softmax Probabilities
Student: MobileNetV3-Small (1.5MB FP32) -> Trained to match Teacher Outputs (KL Div)

Developer: Inayat Hussain | AMD Slingshot 2026
Part 11 of 14 — Enterprise Optimization
"""

import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import timm

# Adjust sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shield_xception import ShieldXception

def distill_model(teacher_weights_path: str, save_path: str, data_loader: DataLoader, epochs=10):
    """
    Main distillation loop.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting Distillation on {device}...")

    # 1. Load Teacher (Frozen)
    print("Loading Teacher (Xception)...")
    teacher = ShieldXception().to(device)
    try:
        sd = torch.load(teacher_weights_path, map_location=device)
        teacher.model.load_state_dict(sd, strict=False)
        print("Teacher weights loaded.")
    except Exception as e:
        print(f"Warning: Could not load teacher weights: {e}")
    
    teacher.eval() # Freeze
    for param in teacher.parameters():
        param.requires_grad = False

    # 2. Initialize Student (MobileNetV3 Small)
    print("Initializing Student (MobileNetV3-Small)...")
    # Using timm for consistent efficient architecture
    student = timm.create_model('mobilenetv3_small_100', pretrained=True, num_classes=2).to(device)
    student.train()

    # 3. Optimization Setup
    optimizer = optim.Adam(student.parameters(), lr=0.001)
    # Distillation Loss: KL Divergence (T=Temperature) + CrossEntropy (Hard Labels)
    temperature = 4.0
    alpha = 0.5 # Balance between hard and soft loss

    # 4. Training Loop
    print(f"Training for {epochs} epochs...")
    for epoch in range(epochs):
        total_loss = 0.0
        batches = 0
        
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            with torch.no_grad():
                teacher_logits = teacher.model(inputs) # Raw logits
            
            student_logits = student(inputs)
            
            # Soft Loss (KLDiv)
            soft_loss = nn.KLDivLoss(reduction="batchmean")(
                F.log_softmax(student_logits / temperature, dim=1),
                F.softmax(teacher_logits / temperature, dim=1)
            ) * (temperature * temperature)
            
            # Hard Loss (CE)
            hard_loss = F.cross_entropy(student_logits, labels)
            
            loss = alpha * soft_loss + (1 - alpha) * hard_loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batches += 1
            
            if batches % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Batch {batches} | Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / max(1, batches)
        print(f"Epoch {epoch+1} Complete. Avg Loss: {avg_loss:.4f}")

    # 5. Save Student
    torch.save(student.state_dict(), save_path)
    print(f"Student model saved to {save_path}")
    
    return student

if __name__ == "__main__":
    # Example usage (Dummy Data)
    print("Test Run (Dummy Data)...")
    
    # Dummy Dataset
    dummy_data = []
    for _ in range(10):
        img = torch.randn(1, 3, 299, 299) # Xception size
        label = torch.randint(0, 2, (1,)).item()
        dummy_data.append((img, label))
        
    class SimpleDataset(torch.utils.data.Dataset):
        def __len__(self): return len(dummy_data)
        def __getitem__(self, idx): return dummy_data[idx][0].squeeze(0), dummy_data[idx][1]
        
    loader = DataLoader(SimpleDataset(), batch_size=2)
    
    distill_model("ffpp_c23.pth", "models/shield_student_mobilenet.pth", loader, epochs=1)
