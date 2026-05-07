import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class VelocitySimDataset(Dataset):
    def __init__(self, txt_path, window_size=50, sampling_rate=100):
        print(f"Loading data for Velocity Sim from {txt_path}...")
        df = pd.read_csv(txt_path, sep=r'\s+', skiprows=1, header=None)
        if sampling_rate > 1:
            df = df.iloc[::sampling_rate]
            
        self.targets_reg_raw = df.iloc[:, 1:3].values.astype(np.float32)
        self.sensors = df.iloc[:, 3:19].values.astype(np.float32)
        
        y_cls_list = []
        for co_ppm, eth_ppm in self.targets_reg_raw:
            if co_ppm < 0.1 and eth_ppm < 0.1: y_cls_list.append(0)
            elif co_ppm >= 0.1 and eth_ppm < 0.1: y_cls_list.append(1)
            elif co_ppm < 0.1 and eth_ppm >= 0.1: y_cls_list.append(2)
            else: y_cls_list.append(3)
        self.targets_cls = np.array(y_cls_list, dtype=np.int64)

        self.target_scaler = StandardScaler()
        self.targets_reg = self.target_scaler.fit_transform(self.targets_reg_raw).astype(np.float32)
        
        self.sensor_scaler = StandardScaler()
        self.sensors = self.sensor_scaler.fit_transform(self.sensors).astype(np.float32)
        
        self.window_size = window_size
        self.length = len(df) - (window_size * 2) # Buffer for time warping
        
    def __len__(self): return self.length

    def __getitem__(self, idx):
        # 1. Simulate velocity between 0.5 (slow) and 5.0 (fast) m/s
        velocity = np.random.uniform(0.5, 5.0)
        
        # 2. Physics-Informed Aerodynamic Simulation (Time Warping)
        warp_factor = np.clip(velocity / 2.0, 0.5, 2.0)
        raw_window_size = int(self.window_size * warp_factor)
        
        raw_x = self.sensors[idx : idx + raw_window_size]
        raw_x_tensor = torch.tensor(raw_x, dtype=torch.float32).T.unsqueeze(0)
        
        # Interpolate to fixed window size
        warped_x = F.interpolate(raw_x_tensor, size=self.window_size, mode='linear', align_corners=False)
        warped_x = warped_x.squeeze(0).T
        
        # Amplitude Scaling (Wind dilution)
        amplitude_scale = 1.0 / (1.0 + 0.2 * (velocity - 1.0))
        warped_x = warped_x * amplitude_scale
        
        target_idx = idx + raw_window_size - 1
        return warped_x, torch.tensor(self.targets_reg[target_idx], dtype=torch.float32), torch.tensor(self.targets_cls[target_idx], dtype=torch.long), torch.tensor([velocity], dtype=torch.float32)

class MTL_TemporalTransformer_Velocity(nn.Module):
    def __init__(self, num_sensors=16, d_model=16, n_heads=2, num_layers=1, window_size=50):
        super().__init__()
        self.input_proj = nn.Linear(num_sensors, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, window_size, d_model))
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # NEW: Velocity projector to inject speed into the Transformer sequence
        self.velocity_projector = nn.Linear(1, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4, dropout=0.3, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.cls_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 4))
        self.reg_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 2))

    def forward(self, x, velocity):
        B, W, S = x.shape
        x = self.input_proj(x) + self.pos_encoder
        
        vel_token = self.velocity_projector(velocity).unsqueeze(1) # (B, 1, d_model)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        
        # Sequence: [CLS, VELOCITY, Time_0...Time_W]
        x = torch.cat((cls_tokens, vel_token, x), dim=1) 
        x = self.transformer(x)
        
        return self.cls_head(x[:, 0, :]), self.reg_head(x[:, 0, :])

if __name__ == "__main__":
    print("Testing Dataset and Architecture...")
    data_path = r"../Dataset/dynamic_mixtures/ethylene_CO.txt"
    dataset = VelocitySimDataset(data_path, window_size=50, sampling_rate=100)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    x, y_reg, y_cls, v = next(iter(loader))
    print(f"Data Batch Shapes -> x: {x.shape}, v: {v.shape}")
    
    model = MTL_TemporalTransformer_Velocity().to(device)
    out_cls, out_reg = model(x.to(device), v.to(device))
    print(f"Model Outputs -> cls: {out_cls.shape}, reg: {out_reg.shape}")
    print("Velocity-Aware Architecture is Ready!")
