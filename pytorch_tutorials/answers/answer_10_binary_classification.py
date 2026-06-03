# 第 10 课答案：二分类：logits、Sigmoid 与 BCEWithLogitsLoss
# 先独立完成 lesson 文件，再打开这个答案对照。

import torch
from torch import nn


def main():
    torch.manual_seed(0)
    x = torch.randn(300, 2)
    y = (x[:, 0] + x[:, 1] > 0).float().unsqueeze(1)
    model = nn.Linear(2, 1)
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    loss_fn = nn.BCEWithLogitsLoss()
    for _ in range(120):
        logits = model(x)
        loss = loss_fn(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    probs = torch.sigmoid(model(x))
    acc = ((probs > 0.5) == y.bool()).float().mean()
    print("loss:", loss.item(), "accuracy:", acc.item())


if __name__ == "__main__":
    main()
