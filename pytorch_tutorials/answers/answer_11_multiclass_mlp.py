# 第 11 课答案：多分类 MLP 与 CrossEntropyLoss
# 先独立完成 lesson 文件，再打开这个答案对照。

import torch
from torch import nn


def main():
    torch.manual_seed(1)
    centers = torch.tensor([[-2.0, 0.0], [2.0, 0.0], [0.0, 2.5]])
    labels = torch.arange(3).repeat_interleave(120)
    x = centers[labels] + 0.6 * torch.randn(360, 2)
    model = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 3))
    opt = torch.optim.Adam(model.parameters(), lr=0.03)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(150):
        logits = model(x)
        loss = loss_fn(logits, labels)
        opt.zero_grad()
        loss.backward()
        opt.step()
    acc = (model(x).argmax(dim=1) == labels).float().mean()
    print("loss:", loss.item(), "accuracy:", acc.item())


if __name__ == "__main__":
    main()
