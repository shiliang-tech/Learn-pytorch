# 第 04 课答案：索引、切片与布尔 mask
# 先独立完成 lesson 文件，再打开这个答案对照。

import torch


def main():
    x = torch.arange(20, dtype=torch.float32).reshape(4, 5)
    row2 = x[1]
    first_three_cols = x[:, :3]
    last_col = x[:, -1]
    selected = x[x > 10]
    print("row2:", row2)
    print("first_three_cols:\n", first_three_cols)
    print("last_col:", last_col)
    print("selected mean:", selected.mean().item())


if __name__ == "__main__":
    main()
