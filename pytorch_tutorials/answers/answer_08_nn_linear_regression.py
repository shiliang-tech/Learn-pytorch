# 第 08 课答案：nn.Module 版线性回归
# 先独立完成 lesson 文件，再打开这个答案对照。

import torch
from torch import nn


def main():
    torch.manual_seed(0)
    x = torch.linspace(-2, 2, 120).unsqueeze(1)
    y = -3 * x + 0.5
    model = nn.Linear(1, 1)
    loss_fn = nn.MSELoss()
    opt = torch.optim.SGD(model.parameters(), lr=0.05)
    for _ in range(180):
        pred = model(x)
        loss = loss_fn(pred, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    print("weight:", model.weight.item(), "bias:", model.bias.item(), "loss:", loss.item())


if __name__ == "__main__":
    main()
