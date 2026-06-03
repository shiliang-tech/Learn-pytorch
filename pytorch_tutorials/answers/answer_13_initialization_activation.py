# 第 13 课答案：初始化与激活函数
# 先独立完成 lesson 文件，再打开这个答案对照。

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 2))
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.net(x)


def main():
    torch.manual_seed(0)
    x = torch.randn(400, 2)
    y = ((x[:, 0] ** 2 + x[:, 1] ** 2) > 1.2).long()
    model = MLP()
    opt = torch.optim.Adam(model.parameters(), lr=0.02)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(120):
        loss = loss_fn(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    print("accuracy:", (model(x).argmax(1) == y).float().mean().item())


if __name__ == "__main__":
    main()
