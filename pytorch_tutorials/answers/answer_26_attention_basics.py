# 第 26 课答案：Attention：Query、Key、Value
# 先独立完成 lesson 文件，再打开这个答案对照。

import math
import torch


def scaled_dot_product_attention(q, k, v):
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
    weights = torch.softmax(scores, dim=-1)
    output = weights @ v
    return output, weights


def main():
    torch.manual_seed(0)
    q = torch.randn(2, 4, 8)
    k = torch.randn(2, 4, 8)
    v = torch.randn(2, 4, 8)
    output, weights = scaled_dot_product_attention(q, k, v)
    print("output shape:", output.shape)
    print("weights shape:", weights.shape)
    print("row sums:", weights.sum(dim=-1))


if __name__ == "__main__":
    main()
