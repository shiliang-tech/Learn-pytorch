# 第 24 课答案：RNN：序列分类入门
# 先独立完成 lesson 文件，再打开这个答案对照。

import torch
from torch import nn


class RNNClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(input_size=1, hidden_size=12, batch_first=True)
        self.fc = nn.Linear(12, 2)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1])


def main():
    torch.manual_seed(0)
    x = torch.randn(400, 8, 1)
    y = (x[:, 4:].mean(dim=(1, 2)) > x[:, :4].mean(dim=(1, 2))).long()
    model = RNNClassifier()
    opt = torch.optim.Adam(model.parameters(), lr=0.02)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(100):
        loss = loss_fn(model(x), y)
        opt.zero_grad(); loss.backward(); opt.step()
    print("accuracy:", (model(x).argmax(1) == y).float().mean().item())


if __name__ == "__main__":
    main()
