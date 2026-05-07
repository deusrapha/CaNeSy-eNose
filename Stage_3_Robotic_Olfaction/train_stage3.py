import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from velocity_model_training import VelocitySimDataset, MTL_TemporalTransformer_Velocity
from tqdm import tqdm
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_stage3():
    print("Initializing Stage 3: Velocity-Aware Training...")
    data_path = r"../Dataset/dynamic_mixtures/ethylene_CO.txt"
    
    # Load the massive dataset with simulated aerodynamics
    dataset = VelocitySimDataset(data_path, window_size=50, sampling_rate=50)
    
    # Temporal Split
    train_size = int(0.8 * len(dataset))
    indices = list(range(len(dataset)))
    train_dataset = torch.utils.data.Subset(dataset, indices[:train_size])
    val_dataset = torch.utils.data.Subset(dataset, indices[train_size:])
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    model = MTL_TemporalTransformer_Velocity(d_model=16, n_heads=2, num_layers=1, window_size=50).to(device)
    
    # Strong regularization to combat aerodynamic simulation noise
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    criterion_cls = nn.CrossEntropyLoss() # Removed label_smoothing for compatibility
    criterion_reg = nn.MSELoss()
    
    epochs = 10
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        try:
            for x, y_reg, y_cls, v in pbar:
                x, y_reg, y_cls, v = x.to(device), y_reg.to(device), y_cls.to(device), v.to(device)
                optimizer.zero_grad()
                
                out_cls, out_reg = model(x, v)
                loss_cls = criterion_cls(out_cls, y_cls)
                loss_reg = criterion_reg(out_reg, y_reg)
                loss = loss_cls + loss_reg
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                running_loss += loss.item()
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        except Exception as e:
            import traceback
            print(f"\nCRITICAL ERROR DURING TRAINING:\n")
            traceback.print_exc()
            return
            
        avg_loss = running_loss / max(1, len(train_loader))
        
        # Validation
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for x, y_reg, y_cls, v in val_loader:
                x, y_reg, y_cls, v = x.to(device), y_reg.to(device), y_cls.to(device), v.to(device)
                out_cls, out_reg = model(x, v)
                
                loss = criterion_cls(out_cls, y_cls) + criterion_reg(out_reg, y_reg)
                val_loss += loss.item()
                
                preds = torch.argmax(out_cls, dim=1)
                correct += (preds == y_cls).sum().item()
                total += y_cls.size(0)
                
        val_loss /= len(val_loader)
        val_acc = correct / total
        print(f"Epoch {epoch+1} - Train Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print("--> Saved new best velocity model!")
            torch.save(model.state_dict(), "best_velocity_model_stage3B_50.pth")

if __name__ == "__main__":
    train_stage3()
