# 第 28 课答案：AutoEncoder：压缩与重建
# 先独立完成 lesson 文件，再打开这个答案对照。

import math
import torch
from torch import nn


class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 1))
        self.decoder = nn.Sequential(nn.Linear(1, 8), nn.Tanh(), nn.Linear(8, 2))

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


def main():
    torch.manual_seed(0)
    t = torch.linspace(0, 2 * math.pi, 300)
    x = torch.stack([torch.cos(t), torch.sin(t)], dim=1)
    model = AutoEncoder()
    opt = torch.optim.Adam(model.parameters(), lr=0.02)
    loss_fn = nn.MSELoss()
    for _ in range(250):
        recon = model(x)
        loss = loss_fn(recon, x)
        opt.zero_grad(); loss.backward(); opt.step()
    print("reconstruction_mse:", loss.item())


if __name__ == "__main__":
    main()
