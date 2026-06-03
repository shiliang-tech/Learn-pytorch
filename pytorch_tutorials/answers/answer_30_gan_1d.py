# 第 30 课答案：GAN：一维分布生成
# 先独立完成 lesson 文件，再打开这个答案对照。

import torch
from torch import nn


def main():
    torch.manual_seed(0)
    g = nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 1))
    d = nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 1))
    opt_g = torch.optim.Adam(g.parameters(), lr=0.01)
    opt_d = torch.optim.Adam(d.parameters(), lr=0.01)
    loss_fn = nn.BCEWithLogitsLoss()
    for _ in range(250):
        real = torch.randn(64, 1) * 0.5 + 2.0
        noise = torch.randn(64, 1)
        fake = g(noise).detach()
        d_loss = loss_fn(d(real), torch.ones(64, 1)) + loss_fn(d(fake), torch.zeros(64, 1))
        opt_d.zero_grad(); d_loss.backward(); opt_d.step()

        noise = torch.randn(64, 1)
        fake = g(noise)
        g_loss = loss_fn(d(fake), torch.ones(64, 1))
        opt_g.zero_grad(); g_loss.backward(); opt_g.step()
    samples = g(torch.randn(1000, 1)).detach()
    print("generated mean/std:", samples.mean().item(), samples.std().item())


if __name__ == "__main__":
    main()
