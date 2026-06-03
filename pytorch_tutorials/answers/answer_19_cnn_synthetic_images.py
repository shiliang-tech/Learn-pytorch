# 第 19 课答案：CNN 小项目：识别条纹方向
# 先独立完成 lesson 文件，再打开这个答案对照。

import torch
from torch import nn


def make_data(n=300):
    imgs = torch.zeros(n, 1, 16, 16)
    y = torch.arange(n) % 2
    for i in range(n):
        if y[i] == 0:
            imgs[i, 0, :, ::4] = 1.0
        else:
            imgs[i, 0, ::4, :] = 1.0
    imgs += 0.1 * torch.randn_like(imgs)
    return imgs, y.long()


def main():
    torch.manual_seed(0)
    x, y = make_data()
    model = nn.Sequential(
        nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(), nn.Linear(16 * 4 * 4, 2)
    )
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(40):
        loss = loss_fn(model(x), y)
        opt.zero_grad(); loss.backward(); opt.step()
    print("accuracy:", (model(x).argmax(1) == y).float().mean().item())


if __name__ == "__main__":
    main()
