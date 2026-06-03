# 第 15 课答案：保存与加载模型参数
# 先独立完成 lesson 文件，再打开这个答案对照。

from pathlib import Path
import torch
from torch import nn


def main():
    torch.manual_seed(0)
    x = torch.randn(100, 3)
    y = x @ torch.tensor([[2.0], [-1.0], [0.5]]) + 0.3
    model = nn.Linear(3, 1)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    loss_fn = nn.MSELoss()
    for _ in range(80):
        loss = loss_fn(model(x), y)
        opt.zero_grad(); loss.backward(); opt.step()
    path = Path(__file__).resolve().parents[1] / "tmp_linear.pt"
    torch.save(model.state_dict(), path)
    loaded = nn.Linear(3, 1)
    loaded.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval(); loaded.eval()
    print("saved to:", path)
    print("outputs equal:", torch.allclose(model(x[:5]), loaded(x[:5])))


if __name__ == "__main__":
    main()
