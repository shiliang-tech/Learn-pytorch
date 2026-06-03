# 第 09 课答案：TensorDataset 与 DataLoader
# 先独立完成 lesson 文件，再打开这个答案对照。

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def main():
    torch.manual_seed(0)
    x = torch.linspace(-3, 3, 200).unsqueeze(1)
    y = 4 * x - 2 + 0.1 * torch.randn_like(x)
    loader = DataLoader(TensorDataset(x, y), batch_size=16, shuffle=True)
    model = nn.Linear(1, 1)
    opt = torch.optim.SGD(model.parameters(), lr=0.05)
    loss_fn = nn.MSELoss()
    for _ in range(20):
        for xb, yb in loader:
            loss = loss_fn(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
    print("weight:", model.weight.item(), "bias:", model.bias.item())


if __name__ == "__main__":
    main()
