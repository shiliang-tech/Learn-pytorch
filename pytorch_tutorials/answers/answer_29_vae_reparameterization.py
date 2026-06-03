# 第 29 课答案：VAE：均值、方差与重参数化
# 先独立完成 lesson 文件，再打开这个答案对照。

import torch
from torch import nn


def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + std * eps


class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(2, 16), nn.ReLU())
        self.mu = nn.Linear(16, 1)
        self.logvar = nn.Linear(16, 1)
        self.dec = nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 2))

    def forward(self, x):
        h = self.enc(x)
        mu, logvar = self.mu(h), self.logvar(h)
        z = reparameterize(mu, logvar)
        return self.dec(z), mu, logvar


def main():
    torch.manual_seed(0)
    x = torch.randn(300, 2) * torch.tensor([2.0, 0.5])
    model = VAE()
    opt = torch.optim.Adam(model.parameters(), lr=0.02)
    for _ in range(160):
        recon, mu, logvar = model(x)
        recon_loss = ((recon - x) ** 2).mean()
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + 0.05 * kl
        opt.zero_grad(); loss.backward(); opt.step()
    print("loss:", loss.item(), "recon:", recon_loss.item(), "kl:", kl.item())


if __name__ == "__main__":
    main()
