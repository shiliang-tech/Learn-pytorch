# 第 14 课答案：正则化：weight decay 与 Dropout
# 先独立完成 lesson 文件，再打开这个答案对照。

import torch
from torch import nn


def main():
    torch.manual_seed(0)
    x = torch.randn(300, 10)
    y = (x[:, :3].sum(dim=1) > 0).long()
    model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Dropout(0.4), nn.Linear(32, 2))
    opt = torch.optim.Adam(model.parameters(), lr=0.02, weight_decay=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(80):
        model.train()
        loss = loss_fn(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    sample = x[:1]
    model.train(); a = model(sample)
    model.train(); b = model(sample)
    model.eval(); c = model(sample); d = model(sample)
    print("accuracy:", (model(x).argmax(1) == y).float().mean().item())
    print("train outputs equal:", torch.allclose(a, b))
    print("eval outputs equal:", torch.allclose(c, d))


if __name__ == "__main__":
    main()
