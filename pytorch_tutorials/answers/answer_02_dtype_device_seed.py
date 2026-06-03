# 第 02 课答案：dtype、device 与随机种子
# 先独立完成 lesson 文件，再打开这个答案对照。

import torch


def main():
    torch.manual_seed(42)
    x = torch.randn(2, 3)
    x64 = x.to(torch.float64)
    x32 = x64.float()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_device = x32.to(device)
    print("dtype before/after:", x.dtype, x64.dtype, x32.dtype)
    print("device:", x_device.device)
    print(x_device)


if __name__ == "__main__":
    main()
