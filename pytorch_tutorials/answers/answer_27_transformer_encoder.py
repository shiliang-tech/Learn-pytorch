# 第 27 课答案：TransformerEncoder：序列建模小例子
# 先独立完成 lesson 文件，再打开这个答案对照。

import torch
from torch import nn


class TinyTransformer(nn.Module):
    def __init__(self, vocab=30, dim=16):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.pos = nn.Parameter(torch.randn(1, 6, dim) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model=dim, nhead=4, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.fc = nn.Linear(dim, 2)

    def forward(self, tokens):
        x = self.embed(tokens) + self.pos
        x = self.encoder(x)
        return self.fc(x.mean(dim=1))


def main():
    torch.manual_seed(0)
    tokens = torch.randint(0, 30, (400, 6))
    y = (tokens[:, 0] > tokens[:, -1]).long()
    model = TinyTransformer()
    opt = torch.optim.Adam(model.parameters(), lr=0.02)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(120):
        loss = loss_fn(model(tokens), y)
        opt.zero_grad(); loss.backward(); opt.step()
    print("accuracy:", (model(tokens).argmax(1) == y).float().mean().item())


if __name__ == "__main__":
    main()
