# 第 18 课答案：图像张量与 Conv2d 形状
# 先独立完成 lesson 文件，再打开这个答案对照。

import torch
from torch import nn


def main():
    x = torch.randn(8, 1, 28, 28)
    conv = nn.Conv2d(1, 4, kernel_size=3, padding=1)
    pool = nn.MaxPool2d(2)
    y = conv(x)
    z = pool(y)
    print("input:", x.shape)
    print("after conv:", y.shape)
    print("after pool:", z.shape)


if __name__ == "__main__":
    main()
