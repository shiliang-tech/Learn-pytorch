# 第 21 课答案：学习率调度与早停思路
# 先独立完成 lesson 文件，再打开这个答案对照。

import torch
from torch import nn


def main():
    torch.manual_seed(0)
    x = torch.randn(500, 5)
    y = (x[:, 0] + 0.5 * x[:, 1] > 0).long()
    train_x, val_x = x[:400], x[400:]
    train_y, val_y = y[:400], y[400:]
    model = nn.Sequential(nn.Linear(5, 16), nn.ReLU(), nn.Linear(16, 2))
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=20, gamma=0.5)
    loss_fn = nn.CrossEntropyLoss()
    best = float("inf")
    patience, bad_epochs = 8, 0
    for epoch in range(100):
        model.train()
        loss = loss_fn(model(train_x), train_y)
        opt.zero_grad(); loss.backward(); opt.step(); scheduler.step()
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(val_x), val_y).item()
        if val_loss < best - 1e-4:
            best, bad_epochs = val_loss, 0
        else:
            bad_epochs += 1
        if bad_epochs >= patience:
            break
    print("stopped_epoch:", epoch + 1, "best_val_loss:", best, "lr:", scheduler.get_last_lr()[0])


if __name__ == "__main__":
    main()
