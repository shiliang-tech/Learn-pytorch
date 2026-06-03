# 第 12 课答案：训练集、验证集与 eval 模式
# 先独立完成 lesson 文件，再打开这个答案对照。

import torch
from torch import nn


def accuracy(model, x, y):
    model.eval()
    with torch.no_grad():
        return (model(x).argmax(1) == y).float().mean().item()


def main():
    torch.manual_seed(2)
    x = torch.randn(500, 4)
    y = ((x[:, 0] * x[:, 1] + x[:, 2]) > 0).long()
    train_x, val_x = x[:400], x[400:]
    train_y, val_y = y[:400], y[400:]
    model = nn.Sequential(nn.Linear(4, 24), nn.ReLU(), nn.Linear(24, 2))
    opt = torch.optim.Adam(model.parameters(), lr=0.02)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(60):
        model.train()
        loss = loss_fn(model(train_x), train_y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    print("train_acc:", accuracy(model, train_x, train_y))
    print("val_acc:", accuracy(model, val_x, val_y))


if __name__ == "__main__":
    main()
