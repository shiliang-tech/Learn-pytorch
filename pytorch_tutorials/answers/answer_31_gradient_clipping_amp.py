# 第 31 课答案：梯度裁剪与混合精度入口
# 先独立完成 lesson 文件，再打开这个答案对照。

import torch
from torch import nn


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    x = torch.randn(256, 20, device=device)
    y = (x[:, :5].sum(1) > 0).long().to(device)
    model = nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 2)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.02)
    loss_fn = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    for _ in range(50):
        opt.zero_grad()
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            loss = loss_fn(model(x), y)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(opt)
        scaler.update()
    print("device:", device, "amp:", use_amp, "last_grad_norm:", float(grad_norm))


if __name__ == "__main__":
    main()
