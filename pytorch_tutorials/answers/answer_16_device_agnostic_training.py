# 第 16 课答案：设备无关训练：CPU/GPU 通用代码
# 先独立完成 lesson 文件，再打开这个答案对照。

import torch
from torch import nn


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    x = torch.randn(256, 6).to(device)
    y = (x[:, 0] - x[:, 1] > 0).long().to(device)
    model = nn.Sequential(nn.Linear(6, 16), nn.ReLU(), nn.Linear(16, 2)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.02)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(60):
        loss = loss_fn(model(x), y)
        opt.zero_grad(); loss.backward(); opt.step()
    print("device:", device)
    print("accuracy:", (model(x).argmax(1) == y).float().mean().item())


if __name__ == "__main__":
    main()
