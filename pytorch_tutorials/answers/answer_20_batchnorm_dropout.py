# 第 20 课答案：BatchNorm 与 Dropout 的 train/eval 差异
# 先独立完成 lesson 文件，再打开这个答案对照。

import torch
from torch import nn


def main():
    torch.manual_seed(0)
    bn = nn.BatchNorm1d(8)
    model = nn.Sequential(nn.Linear(4, 8), bn, nn.ReLU(), nn.Dropout(0.5), nn.Linear(8, 2))
    x = torch.randn(16, 4)
    model.train()
    train_a = model(x)
    train_b = model(x)
    model.eval()
    eval_a = model(x)
    eval_b = model(x)
    print("train equal:", torch.allclose(train_a, train_b))
    print("eval equal:", torch.allclose(eval_a, eval_b))
    print("running_mean:", bn.running_mean)


if __name__ == "__main__":
    main()
