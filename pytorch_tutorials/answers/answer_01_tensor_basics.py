# 第 01 课答案：Tensor 入门：创建、运算、矩阵乘法
# 先独立完成 lesson 文件，再打开这个答案对照。

import torch


def main():
    x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
    w = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=torch.float32)
    y = x @ w
    print("y=", y)
    print("shape=", tuple(y.shape))
    print("mean=", y.mean().item())
    print("max=", y.max().item())


if __name__ == "__main__":
    main()
