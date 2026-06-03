# 第 03 课答案：形状变换与广播机制
# 先独立完成 lesson 文件，再打开这个答案对照。

import torch


def main():
    x = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    bias = torch.tensor([10, 20, 30, 40], dtype=torch.float32)
    y = x + bias
    z = y.unsqueeze(0)
    print("x shape:", x.shape)
    print("y=", y)
    print("z shape:", z.shape)


if __name__ == "__main__":
    main()
