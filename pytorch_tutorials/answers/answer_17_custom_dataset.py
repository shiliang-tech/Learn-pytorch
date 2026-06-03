# 第 17 课答案：自定义 Dataset
# 先独立完成 lesson 文件，再打开这个答案对照。

import math
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


class SineDataset(Dataset):
    def __init__(self, n=300):
        self.x = torch.linspace(-math.pi, math.pi, n).unsqueeze(1)
        self.y = torch.sin(self.x)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def main():
    torch.manual_seed(0)
    loader = DataLoader(SineDataset(), batch_size=32, shuffle=True)
    model = nn.Sequential(nn.Linear(1, 32), nn.Tanh(), nn.Linear(32, 1))
    opt = torch.optim.Adam(model.parameters(), lr=0.02)
    loss_fn = nn.MSELoss()
    for _ in range(80):
        for xb, yb in loader:
            loss = loss_fn(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
    print("final batch loss:", loss.item())


if __name__ == "__main__":
    main()
