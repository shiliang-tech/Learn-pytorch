# 第 23 课答案：Embedding：把离散 id 变成向量
# 先独立完成 lesson 文件，再打开这个答案对照。

import torch
from torch import nn


class TextClassifier(nn.Module):
    def __init__(self, vocab_size=20, dim=8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.fc = nn.Linear(dim, 2)

    def forward(self, tokens):
        emb = self.embedding(tokens)
        pooled = emb.mean(dim=1)
        return self.fc(pooled)


def main():
    torch.manual_seed(0)
    tokens = torch.randint(0, 20, (300, 6))
    y = (tokens.eq(7).any(dim=1) | tokens.eq(13).any(dim=1)).long()
    model = TextClassifier()
    opt = torch.optim.Adam(model.parameters(), lr=0.03)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(120):
        loss = loss_fn(model(tokens), y)
        opt.zero_grad(); loss.backward(); opt.step()
    print("accuracy:", (model(tokens).argmax(1) == y).float().mean().item())


if __name__ == "__main__":
    main()
