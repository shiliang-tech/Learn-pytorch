# 第 06 课答案：自动求导 autograd
# 先独立完成 lesson 文件，再打开这个答案对照。

import torch


def main():
    w = torch.tensor(1.0, requires_grad=True)
    loss = (w - 3) ** 2
    loss.backward()
    print("loss:", loss.item())
    print("grad:", w.grad.item())
    print("expected grad:", 2 * (w.item() - 3))


if __name__ == "__main__":
    main()
