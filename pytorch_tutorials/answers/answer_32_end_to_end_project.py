# 第 32 课答案：端到端小项目：从数据到保存模型
# 先独立完成 lesson 文件，再打开这个答案对照。

from pathlib import Path
import torch
from torch import nn


def train_one_epoch(model, x, y, optimizer, loss_fn):
    model.train()
    loss = loss_fn(model(x), y)
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    return loss.item()


def evaluate(model, x, y, loss_fn):
    model.eval()
    with torch.no_grad():
        logits = model(x)
        loss = loss_fn(logits, y).item()
        acc = (logits.argmax(1) == y).float().mean().item()
    return loss, acc


def main():
    torch.manual_seed(0)
    x = torch.randn(800, 6)
    y = ((x[:, 0] * x[:, 1] + x[:, 2] - x[:, 3].abs()) > 0).long()
    train_x, val_x = x[:650], x[650:]
    train_y, val_y = y[:650], y[650:]
    model = nn.Sequential(nn.Linear(6, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 2))
    opt = torch.optim.Adam(model.parameters(), lr=0.02)
    loss_fn = nn.CrossEntropyLoss()
    best_acc = 0.0
    path = Path(__file__).resolve().parents[1] / "best_project_model.pt"
    for epoch in range(120):
        train_loss = train_one_epoch(model, train_x, train_y, opt, loss_fn)
        val_loss, val_acc = evaluate(model, val_x, val_y, loss_fn)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), path)
    print("train_loss:", train_loss, "val_loss:", val_loss, "best_val_acc:", best_acc)
    print("saved:", path)


if __name__ == "__main__":
    main()
