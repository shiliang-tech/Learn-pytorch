# 第 07 课答案：手写梯度下降：拟合 y=2x+1
# 先独立完成 lesson 文件，再打开这个答案对照。

import torch


def main():
    torch.manual_seed(0)
    x = torch.linspace(-1, 1, 100).unsqueeze(1)
    y = 2 * x + 1
    w = torch.randn(1, 1, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    lr = 0.1
    for epoch in range(120):
        pred = x @ w + b
        loss = ((pred - y) ** 2).mean()
        loss.backward()
        with torch.no_grad():
            w -= lr * w.grad
            b -= lr * b.grad
            w.grad.zero_()
            b.grad.zero_()
    print("w:", w.item(), "b:", b.item(), "loss:", loss.item())


if __name__ == "__main__":
    main()
