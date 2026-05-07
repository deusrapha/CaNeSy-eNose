import traceback
import sys
import torch
from torch.utils.data import DataLoader
from velocity_model_training import VelocitySimDataset, MTL_TemporalTransformer_Velocity

def run_test():
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        data_path = r"../Dataset/dynamic_mixtures/ethylene_CO.txt"
        dataset = VelocitySimDataset(data_path, window_size=100, max_rows=1000)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        model = MTL_TemporalTransformer_Velocity().to(device)
        model.train()
        
        criterion_cls = torch.nn.CrossEntropyLoss()
        criterion_reg = torch.nn.MSELoss()
        
        for x, y_reg, y_cls, v in loader:
            x, y_reg, y_cls, v = x.to(device), y_reg.to(device), y_cls.to(device), v.to(device)
            out_cls, out_reg = model(x, v)
            loss = criterion_cls(out_cls, y_cls) + criterion_reg(out_reg, y_reg)
            loss.backward()
            print("Batch successful!")
            break
        print("Success! No error.")
    except Exception as e:
        traceback.print_exc(file=sys.stdout)

if __name__ == "__main__":
    run_test()
