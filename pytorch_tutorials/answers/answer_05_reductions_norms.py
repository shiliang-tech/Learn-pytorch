# 第 05 课答案：聚合统计、范数与标准化
# 先独立完成 lesson 文件，再打开这个答案对照。

import torch


def main():
    torch.manual_seed(0)
    x = torch.randn(5, 3) * 2 + 5
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True)
    z = (x - mean) / (std + 1e-8)
    print("column mean before:", mean.squeeze(0))
    print("column std before:", std.squeeze(0))
    print("column mean after:", z.mean(dim=0))
    print("row norms:", torch.linalg.vector_norm(z, dim=1))


if __name__ == "__main__":
    main()
