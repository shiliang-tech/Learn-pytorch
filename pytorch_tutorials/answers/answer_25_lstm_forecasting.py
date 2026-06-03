# 第 25 课答案：LSTM：预测正弦序列下一步
# 先独立完成 lesson 文件，再打开这个答案对照。

import math
import torch
from torch import nn


class Forecaster(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=16, batch_first=True)
        self.fc = nn.Linear(16, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1])


def main():
    t = torch.linspace(0, 8 * math.pi, 260)
    series = torch.sin(t)
    xs, ys = [], []
    for i in range(len(series) - 10):
        xs.append(series[i:i + 10])
        ys.append(series[i + 10])
    x = torch.stack(xs).unsqueeze(-1)
    y = torch.stack(ys).unsqueeze(-1)
    model = Forecaster()
    opt = torch.optim.Adam(model.parameters(), lr=0.02)
    loss_fn = nn.MSELoss()
    for _ in range(120):
        loss = loss_fn(model(x), y)
        opt.zero_grad(); loss.backward(); opt.step()
    print("mse:", loss.item())
    print("first prediction/target:", model(x[:1]).item(), y[:1].item())


if __name__ == "__main__":
    main()
