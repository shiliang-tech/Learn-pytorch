from pathlib import Path
import re
import textwrap


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "pytorch_tutorials"
LESSONS_DIR = BASE / "lessons"
ANSWERS_DIR = BASE / "answers"
DOCS_DIR = BASE / "docs"


LESSONS = [
    {
        "slug": "tensor_basics",
        "title": "Tensor 入门：创建、运算、矩阵乘法",
        "principles": [
            "Tensor 是 PyTorch 的核心数据结构，可以把它理解成支持 GPU 和自动求导的多维数组。",
            "标量是 0 维，向量是 1 维，矩阵是 2 维，更高维常用来表达 batch、通道、图像高宽、时间步等。",
            "深度学习里的大多数计算最终都会落到张量运算：加减乘除、矩阵乘法、聚合统计。",
        ],
        "tasks": [
            "创建一个 2x3 的浮点张量 x。",
            "创建一个 3x2 的浮点张量 w，并计算 x @ w。",
            "打印结果的 shape、均值、最大值。",
        ],
        "starter": "def main():\n    # TODO: 创建 x 和 w，然后完成矩阵乘法。\n    pass\n\n\nif __name__ == \"__main__\":\n    main()\n",
        "answer": "import torch\n\n\ndef main():\n    x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)\n    w = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=torch.float32)\n    y = x @ w\n    print(\"y=\", y)\n    print(\"shape=\", tuple(y.shape))\n    print(\"mean=\", y.mean().item())\n    print(\"max=\", y.max().item())\n\n\nif __name__ == \"__main__\":\n    main()\n",
    },
    {
        "slug": "dtype_device_seed",
        "title": "dtype、device 与随机种子",
        "principles": [
            "dtype 决定数值类型，常见训练默认用 torch.float32；分类标签常用 torch.long。",
            "device 决定张量在 CPU 还是 GPU。模型和输入必须在同一个 device 上。",
            "随机种子让实验更容易复现，尤其是初始化、随机数据、DataLoader shuffle。",
        ],
        "tasks": [
            "设置 torch.manual_seed(42)。",
            "创建一个随机张量，把它转换成 float64，再转换回 float32。",
            "选择 cuda 或 cpu 作为 device，并把张量移动过去。",
        ],
        "starter": "def main():\n    # TODO: 设置随机种子，练习 dtype 与 device 转换。\n    pass\n\n\nif __name__ == \"__main__\":\n    main()\n",
        "answer": "import torch\n\n\ndef main():\n    torch.manual_seed(42)\n    x = torch.randn(2, 3)\n    x64 = x.to(torch.float64)\n    x32 = x64.float()\n    device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n    x_device = x32.to(device)\n    print(\"dtype before/after:\", x.dtype, x64.dtype, x32.dtype)\n    print(\"device:\", x_device.device)\n    print(x_device)\n\n\nif __name__ == \"__main__\":\n    main()\n",
    },
    {
        "slug": "reshape_broadcast",
        "title": "形状变换与广播机制",
        "principles": [
            "view/reshape 可以改变张量形状，但元素总数必须一致。",
            "unsqueeze/squeeze 用于增加或删除长度为 1 的维度。",
            "广播让不同形状的张量参与运算，例如 batch 中每一行都加同一个偏置向量。",
        ],
        "tasks": [
            "把 torch.arange(12) 变成 3x4 矩阵。",
            "创建长度为 4 的 bias，并加到矩阵每一行。",
            "用 unsqueeze 把结果变成 1x3x4。",
        ],
        "starter": "def main():\n    # TODO: 练习 reshape、广播和 unsqueeze。\n    pass\n\n\nif __name__ == \"__main__\":\n    main()\n",
        "answer": "import torch\n\n\ndef main():\n    x = torch.arange(12, dtype=torch.float32).reshape(3, 4)\n    bias = torch.tensor([10, 20, 30, 40], dtype=torch.float32)\n    y = x + bias\n    z = y.unsqueeze(0)\n    print(\"x shape:\", x.shape)\n    print(\"y=\", y)\n    print(\"z shape:\", z.shape)\n\n\nif __name__ == \"__main__\":\n    main()\n",
    },
    {
        "slug": "indexing_masking",
        "title": "索引、切片与布尔 mask",
        "principles": [
            "索引用来取局部数据，切片常用于 batch、序列、图像区域。",
            "布尔 mask 可以筛选满足条件的元素，是处理异常值、条件统计的常用手段。",
            "训练中经常会按标签、置信度、padding mask 来选择样本或位置。",
        ],
        "tasks": [
            "创建 4x5 的矩阵。",
            "取出第 2 行、前 3 列、最后一列。",
            "用 mask 找出所有大于 10 的元素，并计算它们的均值。",
        ],
        "starter": "def main():\n    # TODO: 练习索引、切片和布尔 mask。\n    pass\n\n\nif __name__ == \"__main__\":\n    main()\n",
        "answer": "import torch\n\n\ndef main():\n    x = torch.arange(20, dtype=torch.float32).reshape(4, 5)\n    row2 = x[1]\n    first_three_cols = x[:, :3]\n    last_col = x[:, -1]\n    selected = x[x > 10]\n    print(\"row2:\", row2)\n    print(\"first_three_cols:\\n\", first_three_cols)\n    print(\"last_col:\", last_col)\n    print(\"selected mean:\", selected.mean().item())\n\n\nif __name__ == \"__main__\":\n    main()\n",
    },
    {
        "slug": "reductions_norms",
        "title": "聚合统计、范数与标准化",
        "principles": [
            "mean、sum、max、argmax 这类 reduction 会沿某些维度汇总信息。",
            "范数常用来度量向量大小，训练中也会用于正则化、梯度裁剪、相似度计算。",
            "标准化让数据均值接近 0、标准差接近 1，通常能让优化更稳定。",
        ],
        "tasks": [
            "创建一个 5x3 的随机数据矩阵。",
            "计算每一列的均值和标准差。",
            "把每一列标准化，并验证标准化后的均值接近 0。",
        ],
        "starter": "def main():\n    # TODO: 完成按列标准化。\n    pass\n\n\nif __name__ == \"__main__\":\n    main()\n",
        "answer": "import torch\n\n\ndef main():\n    torch.manual_seed(0)\n    x = torch.randn(5, 3) * 2 + 5\n    mean = x.mean(dim=0, keepdim=True)\n    std = x.std(dim=0, keepdim=True)\n    z = (x - mean) / (std + 1e-8)\n    print(\"column mean before:\", mean.squeeze(0))\n    print(\"column std before:\", std.squeeze(0))\n    print(\"column mean after:\", z.mean(dim=0))\n    print(\"row norms:\", torch.linalg.vector_norm(z, dim=1))\n\n\nif __name__ == \"__main__\":\n    main()\n",
    },
    {
        "slug": "autograd_basics",
        "title": "自动求导 autograd",
        "principles": [
            "requires_grad=True 会让 PyTorch 记录张量参与的计算图。",
            "loss.backward() 会从标量 loss 反向传播，计算叶子张量的 grad。",
            "每次反向传播前通常要清空旧梯度，因为 PyTorch 默认会累加梯度。",
        ],
        "tasks": [
            "创建一个可求导的标量 w。",
            "令 loss=(w-3)^2，调用 backward。",
            "打印 w.grad，并解释为什么梯度等于 2*(w-3)。",
        ],
        "starter": "def main():\n    # TODO: 创建 w，计算 loss，然后 backward。\n    pass\n\n\nif __name__ == \"__main__\":\n    main()\n",
        "answer": "import torch\n\n\ndef main():\n    w = torch.tensor(1.0, requires_grad=True)\n    loss = (w - 3) ** 2\n    loss.backward()\n    print(\"loss:\", loss.item())\n    print(\"grad:\", w.grad.item())\n    print(\"expected grad:\", 2 * (w.item() - 3))\n\n\nif __name__ == \"__main__\":\n    main()\n",
    },
    {
        "slug": "manual_gradient_descent",
        "title": "手写梯度下降：拟合 y=2x+1",
        "principles": [
            "训练的基本循环是：前向计算、计算损失、反向传播、更新参数、清空梯度。",
            "梯度下降用参数减去 learning_rate * gradient，让 loss 往下降方向移动。",
            "先手写一次更新过程，可以更清楚 nn.Module 和 optimizer 后面在替你做什么。",
        ],
        "tasks": [
            "生成 x 与 y=2x+1 的训练数据。",
            "用两个 requires_grad 参数 w、b 手动训练。",
            "每轮用 no_grad 更新参数，并把梯度清零。",
        ],
        "starter": "def main():\n    # TODO: 手写一个 100 轮的梯度下降线性回归。\n    pass\n\n\nif __name__ == \"__main__\":\n    main()\n",
        "answer": "import torch\n\n\ndef main():\n    torch.manual_seed(0)\n    x = torch.linspace(-1, 1, 100).unsqueeze(1)\n    y = 2 * x + 1\n    w = torch.randn(1, 1, requires_grad=True)\n    b = torch.zeros(1, requires_grad=True)\n    lr = 0.1\n    for epoch in range(120):\n        pred = x @ w + b\n        loss = ((pred - y) ** 2).mean()\n        loss.backward()\n        with torch.no_grad():\n            w -= lr * w.grad\n            b -= lr * b.grad\n            w.grad.zero_()\n            b.grad.zero_()\n    print(\"w:\", w.item(), \"b:\", b.item(), \"loss:\", loss.item())\n\n\nif __name__ == \"__main__\":\n    main()\n",
    },
    {
        "slug": "nn_linear_regression",
        "title": "nn.Module 版线性回归",
        "principles": [
            "nn.Module 是 PyTorch 组织模型参数和前向计算的标准方式。",
            "nn.Linear(in_features, out_features) 实现 y=xW^T+b。",
            "optimizer 负责根据梯度更新参数，常见有 SGD、Adam。",
        ],
        "tasks": [
            "用 nn.Linear(1, 1) 拟合 y=-3x+0.5。",
            "使用 MSELoss 和 SGD。",
            "训练结束后打印 weight、bias 和 loss。",
        ],
        "starter": "def main():\n    # TODO: 使用 nn.Linear、MSELoss、SGD 完成训练。\n    pass\n\n\nif __name__ == \"__main__\":\n    main()\n",
        "answer": "import torch\n+from torch import nn\n\n\ndef main():\n    torch.manual_seed(0)\n    x = torch.linspace(-2, 2, 120).unsqueeze(1)\n    y = -3 * x + 0.5\n    model = nn.Linear(1, 1)\n    loss_fn = nn.MSELoss()\n    opt = torch.optim.SGD(model.parameters(), lr=0.05)\n    for _ in range(180):\n        pred = model(x)\n        loss = loss_fn(pred, y)\n        opt.zero_grad()\n        loss.backward()\n        opt.step()\n    print(\"weight:\", model.weight.item(), \"bias:\", model.bias.item(), \"loss:\", loss.item())\n\n\nif __name__ == \"__main__\":\n    main()\n",
    },
    {
        "slug": "dataloader_basics",
        "title": "TensorDataset 与 DataLoader",
        "principles": [
            "Dataset 定义如何取一个样本，DataLoader 负责批量、打乱、多进程读取。",
            "batch 训练比一次喂全部数据更常见，也更接近真实项目。",
            "shuffle=True 常用于训练集，验证集和测试集通常不需要 shuffle。",
        ],
        "tasks": [
            "用 TensorDataset 包装 x 和 y。",
            "用 DataLoader 每次取 16 条样本。",
            "写一个 mini-batch 训练循环拟合线性回归。",
        ],
        "starter": "def main():\n    # TODO: 使用 TensorDataset 和 DataLoader 训练模型。\n    pass\n\n\nif __name__ == \"__main__\":\n    main()\n",
        "answer": "import torch\n+from torch import nn\n+from torch.utils.data import DataLoader, TensorDataset\n\n\ndef main():\n    torch.manual_seed(0)\n    x = torch.linspace(-3, 3, 200).unsqueeze(1)\n    y = 4 * x - 2 + 0.1 * torch.randn_like(x)\n    loader = DataLoader(TensorDataset(x, y), batch_size=16, shuffle=True)\n    model = nn.Linear(1, 1)\n    opt = torch.optim.SGD(model.parameters(), lr=0.05)\n    loss_fn = nn.MSELoss()\n    for _ in range(20):\n        for xb, yb in loader:\n            loss = loss_fn(model(xb), yb)\n            opt.zero_grad()\n            loss.backward()\n            opt.step()\n    print(\"weight:\", model.weight.item(), \"bias:\", model.bias.item())\n\n\nif __name__ == \"__main__\":\n    main()\n",
    },
    {
        "slug": "binary_classification",
        "title": "二分类：logits、Sigmoid 与 BCEWithLogitsLoss",
        "principles": [
            "二分类模型常输出一个 logit，logit 经过 sigmoid 后变成属于正类的概率。",
            "BCEWithLogitsLoss 内部包含 sigmoid，比手动 sigmoid 后再 BCE 更数值稳定。",
            "预测时一般用 sigmoid(logit)>0.5 得到类别。",
        ],
        "tasks": [
            "生成二维点，标签为 x1+x2>0。",
            "训练 nn.Linear(2,1) 二分类器。",
            "计算训练准确率。",
        ],
        "starter": "def main():\n    # TODO: 训练一个二分类线性模型，并打印 accuracy。\n    pass\n\n\nif __name__ == \"__main__\":\n    main()\n",
        "answer": "import torch\n+from torch import nn\n\n\ndef main():\n    torch.manual_seed(0)\n    x = torch.randn(300, 2)\n    y = (x[:, 0] + x[:, 1] > 0).float().unsqueeze(1)\n    model = nn.Linear(2, 1)\n    opt = torch.optim.Adam(model.parameters(), lr=0.05)\n    loss_fn = nn.BCEWithLogitsLoss()\n    for _ in range(120):\n        logits = model(x)\n        loss = loss_fn(logits, y)\n        opt.zero_grad()\n        loss.backward()\n        opt.step()\n    probs = torch.sigmoid(model(x))\n    acc = ((probs > 0.5) == y.bool()).float().mean()\n    print(\"loss:\", loss.item(), \"accuracy:\", acc.item())\n\n\nif __name__ == \"__main__\":\n    main()\n",
    },
    {
        "slug": "multiclass_mlp",
        "title": "多分类 MLP 与 CrossEntropyLoss",
        "principles": [
            "多分类模型输出每一类的 logit，shape 通常是 [batch, num_classes]。",
            "CrossEntropyLoss 内部包含 log_softmax，标签必须是类别索引 long 类型。",
            "MLP 通过 Linear + 非线性激活堆叠，能学习比线性模型更复杂的边界。",
        ],
        "tasks": [
            "生成 3 类二维点。",
            "搭建 Linear-ReLU-Linear 的 MLP。",
            "使用 CrossEntropyLoss 训练并打印准确率。",
        ],
        "starter": "def main():\n    # TODO: 完成三分类 MLP。\n    pass\n\n\nif __name__ == \"__main__\":\n    main()\n",
        "answer": "import torch\n+from torch import nn\n\n\ndef main():\n    torch.manual_seed(1)\n    centers = torch.tensor([[-2.0, 0.0], [2.0, 0.0], [0.0, 2.5]])\n    labels = torch.arange(3).repeat_interleave(120)\n    x = centers[labels] + 0.6 * torch.randn(360, 2)\n    model = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 3))\n    opt = torch.optim.Adam(model.parameters(), lr=0.03)\n    loss_fn = nn.CrossEntropyLoss()\n    for _ in range(150):\n        logits = model(x)\n        loss = loss_fn(logits, labels)\n        opt.zero_grad()\n        loss.backward()\n        opt.step()\n    acc = (model(x).argmax(dim=1) == labels).float().mean()\n    print(\"loss:\", loss.item(), \"accuracy:\", acc.item())\n\n\nif __name__ == \"__main__\":\n    main()\n",
    },
    {
        "slug": "train_eval_split",
        "title": "训练集、验证集与 eval 模式",
        "principles": [
            "训练集用来更新参数，验证集用来观察泛化效果。",
            "model.train() 与 model.eval() 会影响 Dropout、BatchNorm 等层的行为。",
            "验证阶段通常配合 torch.no_grad()，避免构建计算图，节省显存和时间。",
        ],
        "tasks": [
            "把合成分类数据切成训练集和验证集。",
            "训练 MLP，并在每个 epoch 后计算验证准确率。",
            "验证时使用 model.eval() 和 torch.no_grad()。",
        ],
        "starter": "def main():\n    # TODO: 写一个带验证集评估的训练循环。\n    pass\n\n\nif __name__ == \"__main__\":\n    main()\n",
        "answer": "import torch\n+from torch import nn\n\n\ndef accuracy(model, x, y):\n    model.eval()\n    with torch.no_grad():\n        return (model(x).argmax(1) == y).float().mean().item()\n\n\ndef main():\n    torch.manual_seed(2)\n    x = torch.randn(500, 4)\n    y = ((x[:, 0] * x[:, 1] + x[:, 2]) > 0).long()\n    train_x, val_x = x[:400], x[400:]\n    train_y, val_y = y[:400], y[400:]\n    model = nn.Sequential(nn.Linear(4, 24), nn.ReLU(), nn.Linear(24, 2))\n    opt = torch.optim.Adam(model.parameters(), lr=0.02)\n    loss_fn = nn.CrossEntropyLoss()\n    for epoch in range(60):\n        model.train()\n        loss = loss_fn(model(train_x), train_y)\n        opt.zero_grad()\n        loss.backward()\n        opt.step()\n    print(\"train_acc:\", accuracy(model, train_x, train_y))\n    print(\"val_acc:\", accuracy(model, val_x, val_y))\n\n\nif __name__ == \"__main__\":\n    main()\n",
    },
    {
        "slug": "initialization_activation",
        "title": "初始化与激活函数",
        "principles": [
            "激活函数引入非线性，否则多层 Linear 仍等价于一个 Linear。",
            "ReLU 常配合 Kaiming 初始化，Tanh/Sigmoid 更常配合 Xavier 初始化。",
            "初始化会影响早期梯度大小，过大或过小都可能让训练不稳定。",
        ],
        "tasks": [
            "定义一个两层 MLP 类。",
            "对 Linear 层使用 kaiming_normal_ 初始化。",
            "训练一个非线性二分类任务。",
        ],
        "starter": "class MLP:\n    # TODO: 继承 nn.Module 并实现 forward。\n    pass\n\n\ndef main():\n    # TODO: 初始化模型并训练。\n    pass\n\n\nif __name__ == \"__main__\":\n    main()\n",
        "answer": "import torch\n+from torch import nn\n\n\nclass MLP(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.net = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 2))\n        for layer in self.net:\n            if isinstance(layer, nn.Linear):\n                nn.init.kaiming_normal_(layer.weight)\n                nn.init.zeros_(layer.bias)\n\n    def forward(self, x):\n        return self.net(x)\n\n\ndef main():\n    torch.manual_seed(0)\n    x = torch.randn(400, 2)\n    y = ((x[:, 0] ** 2 + x[:, 1] ** 2) > 1.2).long()\n    model = MLP()\n    opt = torch.optim.Adam(model.parameters(), lr=0.02)\n    loss_fn = nn.CrossEntropyLoss()\n    for _ in range(120):\n        loss = loss_fn(model(x), y)\n        opt.zero_grad()\n        loss.backward()\n        opt.step()\n    print(\"accuracy:\", (model(x).argmax(1) == y).float().mean().item())\n\n\nif __name__ == \"__main__\":\n    main()\n",
    },
    {
        "slug": "regularization",
        "title": "正则化：weight decay 与 Dropout",
        "principles": [
            "正则化的目标是降低过拟合，让模型不要只记住训练数据。",
            "weight_decay 相当于惩罚过大的权重，常直接传给 optimizer。",
            "Dropout 在训练时随机丢弃部分激活，eval 时自动关闭。",
        ],
        "tasks": [
            "构建包含 Dropout 的 MLP。",
            "给 Adam 设置 weight_decay。",
            "比较 train/eval 模式下同一个输入的输出是否稳定。",
        ],
        "starter": "def main():\n    # TODO: 构建带 Dropout 和 weight_decay 的训练示例。\n    pass\n\n\nif __name__ == \"__main__\":\n    main()\n",
        "answer": "import torch\n+from torch import nn\n\n\ndef main():\n    torch.manual_seed(0)\n    x = torch.randn(300, 10)\n    y = (x[:, :3].sum(dim=1) > 0).long()\n    model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Dropout(0.4), nn.Linear(32, 2))\n    opt = torch.optim.Adam(model.parameters(), lr=0.02, weight_decay=1e-3)\n    loss_fn = nn.CrossEntropyLoss()\n    for _ in range(80):\n        model.train()\n        loss = loss_fn(model(x), y)\n        opt.zero_grad()\n        loss.backward()\n        opt.step()\n    sample = x[:1]\n    model.train(); a = model(sample)\n    model.train(); b = model(sample)\n    model.eval(); c = model(sample); d = model(sample)\n    print(\"accuracy:\", (model(x).argmax(1) == y).float().mean().item())\n    print(\"train outputs equal:\", torch.allclose(a, b))\n    print(\"eval outputs equal:\", torch.allclose(c, d))\n\n\nif __name__ == \"__main__\":\n    main()\n",
    },
    {
        "slug": "save_load",
        "title": "保存与加载模型参数",
        "principles": [
            "推荐保存 model.state_dict()，它只包含参数和 buffer，结构清晰、可迁移。",
            "加载时需要先创建相同结构的模型，再 load_state_dict。",
            "推理前记得 model.eval()，避免 Dropout 和 BatchNorm 处在训练行为。",
        ],
        "tasks": [
            "训练一个小模型。",
            "把 state_dict 保存到 pytorch_tutorials/tmp_linear.pt。",
            "创建新模型加载参数，验证两者输出一致。",
        ],
        "starter": "def main():\n    # TODO: 训练、保存、加载，并比较输出。\n    pass\n\n\nif __name__ == \"__main__\":\n    main()\n",
        "answer": "from pathlib import Path\n+import torch\n+from torch import nn\n\n\ndef main():\n    torch.manual_seed(0)\n    x = torch.randn(100, 3)\n    y = x @ torch.tensor([[2.0], [-1.0], [0.5]]) + 0.3\n    model = nn.Linear(3, 1)\n    opt = torch.optim.SGD(model.parameters(), lr=0.1)\n    loss_fn = nn.MSELoss()\n    for _ in range(80):\n        loss = loss_fn(model(x), y)\n        opt.zero_grad(); loss.backward(); opt.step()\n    path = Path(__file__).resolve().parents[1] / \"tmp_linear.pt\"\n    torch.save(model.state_dict(), path)\n    loaded = nn.Linear(3, 1)\n    loaded.load_state_dict(torch.load(path, map_location=\"cpu\"))\n    model.eval(); loaded.eval()\n    print(\"saved to:\", path)\n    print(\"outputs equal:\", torch.allclose(model(x[:5]), loaded(x[:5])))\n\n\nif __name__ == \"__main__\":\n    main()\n",
    },
    {
        "slug": "device_agnostic_training",
        "title": "设备无关训练：CPU/GPU 通用代码",
        "principles": [
            "设备无关代码会先选择 device，然后把模型和每个 batch 都移动到这个 device。",
            "不要只移动模型或只移动数据，否则会出现 device mismatch 错误。",
            "保存模型时通常仍保存 state_dict；加载到 CPU 可使用 map_location='cpu'。",
        ],
        "tasks": [
            "选择 cuda 或 cpu。",
            "把模型、输入和标签都移动到 device。",
            "训练一个小分类器并打印当前 device。",
        ],
        "starter": "def main():\n    # TODO: 写出 CPU/GPU 都能运行的训练代码。\n    pass\n\n\nif __name__ == \"__main__\":\n    main()\n",
        "answer": "import torch\n+from torch import nn\n\n\ndef main():\n    device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n    torch.manual_seed(0)\n    x = torch.randn(256, 6).to(device)\n    y = (x[:, 0] - x[:, 1] > 0).long().to(device)\n    model = nn.Sequential(nn.Linear(6, 16), nn.ReLU(), nn.Linear(16, 2)).to(device)\n    opt = torch.optim.Adam(model.parameters(), lr=0.02)\n    loss_fn = nn.CrossEntropyLoss()\n    for _ in range(60):\n        loss = loss_fn(model(x), y)\n        opt.zero_grad(); loss.backward(); opt.step()\n    print(\"device:\", device)\n    print(\"accuracy:\", (model(x).argmax(1) == y).float().mean().item())\n\n\nif __name__ == \"__main__\":\n    main()\n",
    },
    {
        "slug": "custom_dataset",
        "title": "自定义 Dataset",
        "principles": [
            "自定义 Dataset 至少实现 __len__ 和 __getitem__。",
            "__getitem__ 返回一个样本，可以是张量、标签、字典等结构。",
            "把数据生成或预处理封装到 Dataset 后，训练循环会更稳定清晰。",
        ],
        "tasks": [
            "写一个 SineDataset，输入 x，标签 y=sin(x)。",
            "用 DataLoader 批量读取。",
            "训练一个小 MLP 做回归。",
        ],
        "starter": "class SineDataset:\n    # TODO: 继承 Dataset，实现 __len__ 和 __getitem__。\n    pass\n\n\ndef main():\n    # TODO: 使用 DataLoader 训练回归模型。\n    pass\n\n\nif __name__ == \"__main__\":\n    main()\n",
        "answer": "import math\n+import torch\n+from torch import nn\n+from torch.utils.data import Dataset, DataLoader\n\n\nclass SineDataset(Dataset):\n    def __init__(self, n=300):\n        self.x = torch.linspace(-math.pi, math.pi, n).unsqueeze(1)\n        self.y = torch.sin(self.x)\n\n    def __len__(self):\n        return len(self.x)\n\n    def __getitem__(self, idx):\n        return self.x[idx], self.y[idx]\n\n\ndef main():\n    torch.manual_seed(0)\n    loader = DataLoader(SineDataset(), batch_size=32, shuffle=True)\n    model = nn.Sequential(nn.Linear(1, 32), nn.Tanh(), nn.Linear(32, 1))\n    opt = torch.optim.Adam(model.parameters(), lr=0.02)\n    loss_fn = nn.MSELoss()\n    for _ in range(80):\n        for xb, yb in loader:\n            loss = loss_fn(model(xb), yb)\n            opt.zero_grad(); loss.backward(); opt.step()\n    print(\"final batch loss:\", loss.item())\n\n\nif __name__ == \"__main__\":\n    main()\n",
    },
    {
        "slug": "conv2d_shapes",
        "title": "图像张量与 Conv2d 形状",
        "principles": [
            "PyTorch 图像张量常用 NCHW：batch、channel、height、width。",
            "Conv2d 会用卷积核在空间维度滑动，输出通道数由 out_channels 决定。",
            "padding、stride、kernel_size 会共同决定输出高宽。",
        ],
        "tasks": [
            "创建形状 [8,1,28,28] 的随机图片 batch。",
            "用 Conv2d(1,4,kernel_size=3,padding=1) 处理。",
            "接 MaxPool2d(2)，打印每一步 shape。",
        ],
        "starter": "def main():\n    # TODO: 创建图像 batch，经过卷积和池化，打印 shape。\n    pass\n\n\nif __name__ == \"__main__\":\n    main()\n",
        "answer": "import torch\n+from torch import nn\n\n\ndef main():\n    x = torch.randn(8, 1, 28, 28)\n    conv = nn.Conv2d(1, 4, kernel_size=3, padding=1)\n    pool = nn.MaxPool2d(2)\n    y = conv(x)\n    z = pool(y)\n    print(\"input:\", x.shape)\n    print(\"after conv:\", y.shape)\n    print(\"after pool:\", z.shape)\n\n\nif __name__ == \"__main__\":\n    main()\n",
    },
    {
        "slug": "cnn_synthetic_images",
        "title": "CNN 小项目：识别条纹方向",
        "principles": [
            "CNN 适合处理局部空间模式，例如边缘、纹理、形状。",
            "合成数据能帮助你先掌握训练流程，不依赖下载数据集。",
            "Flatten 前要确认卷积输出形状，避免 Linear 的输入维度写错。",
        ],
        "tasks": [
            "生成 16x16 灰度图：竖条纹为 0 类，横条纹为 1 类。",
            "写一个小 CNN 分类器。",
            "训练并打印准确率。",
        ],
        "starter": "def main():\n    # TODO: 生成条纹图像，训练 CNN 做二分类。\n    pass\n\n\nif __name__ == \"__main__\":\n    main()\n",
        "answer": "import torch\n+from torch import nn\n\n\ndef make_data(n=300):\n    imgs = torch.zeros(n, 1, 16, 16)\n    y = torch.arange(n) % 2\n    for i in range(n):\n        if y[i] == 0:\n            imgs[i, 0, :, ::4] = 1.0\n        else:\n            imgs[i, 0, ::4, :] = 1.0\n    imgs += 0.1 * torch.randn_like(imgs)\n    return imgs, y.long()\n\n\ndef main():\n    torch.manual_seed(0)\n    x, y = make_data()\n    model = nn.Sequential(\n        nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),\n        nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),\n        nn.Flatten(), nn.Linear(16 * 4 * 4, 2)\n    )\n    opt = torch.optim.Adam(model.parameters(), lr=0.01)\n    loss_fn = nn.CrossEntropyLoss()\n    for _ in range(40):\n        loss = loss_fn(model(x), y)\n        opt.zero_grad(); loss.backward(); opt.step()\n    print(\"accuracy:\", (model(x).argmax(1) == y).float().mean().item())\n\n\nif __name__ == \"__main__\":\n    main()\n",
    },
    {
        "slug": "batchnorm_dropout",
        "title": "BatchNorm 与 Dropout 的 train/eval 差异",
        "principles": [
            "BatchNorm 在训练时使用当前 batch 统计，并更新 running_mean/running_var。",
            "eval 模式下 BatchNorm 使用累计统计，Dropout 则关闭随机丢弃。",
            "这就是为什么验证和推理阶段一定要调用 model.eval()。",
        ],
        "tasks": [
            "构建 Linear-BatchNorm-ReLU-Dropout-Linear 网络。",
            "喂入同一个 batch，比较 train 和 eval 下输出差异。",
            "打印 BatchNorm 的 running_mean。",
        ],
        "starter": "def main():\n    # TODO: 演示 BatchNorm/Dropout 在 train 和 eval 下的差异。\n    pass\n\n\nif __name__ == \"__main__\":\n    main()\n",
        "answer": "import torch\n+from torch import nn\n\n\ndef main():\n    torch.manual_seed(0)\n    bn = nn.BatchNorm1d(8)\n    model = nn.Sequential(nn.Linear(4, 8), bn, nn.ReLU(), nn.Dropout(0.5), nn.Linear(8, 2))\n    x = torch.randn(16, 4)\n    model.train()\n    train_a = model(x)\n    train_b = model(x)\n    model.eval()\n    eval_a = model(x)\n    eval_b = model(x)\n    print(\"train equal:\", torch.allclose(train_a, train_b))\n    print(\"eval equal:\", torch.allclose(eval_a, eval_b))\n    print(\"running_mean:\", bn.running_mean)\n\n\nif __name__ == \"__main__\":\n    main()\n",
    },
    {
        "slug": "scheduler_early_stopping",
        "title": "学习率调度与早停思路",
        "principles": [
            "学习率太大可能震荡，太小可能收敛很慢。",
            "scheduler 可以在训练过程中调整学习率，例如 StepLR 定期衰减。",
            "早停会在验证指标长期不提升时停止训练，避免浪费时间和过拟合。",
        ],
        "tasks": [
            "使用 StepLR 每 20 个 epoch 把学习率乘 0.5。",
            "记录最好的验证 loss。",
            "如果验证 loss 连续若干轮没有提升就提前停止。",
        ],
        "starter": "def main():\n    # TODO: 写一个包含 scheduler 和 early stopping 的训练循环。\n    pass\n\n\nif __name__ == \"__main__\":\n    main()\n",
        "answer": "import torch\n+from torch import nn\n\n\ndef main():\n    torch.manual_seed(0)\n    x = torch.randn(500, 5)\n    y = (x[:, 0] + 0.5 * x[:, 1] > 0).long()\n    train_x, val_x = x[:400], x[400:]\n    train_y, val_y = y[:400], y[400:]\n    model = nn.Sequential(nn.Linear(5, 16), nn.ReLU(), nn.Linear(16, 2))\n    opt = torch.optim.Adam(model.parameters(), lr=0.05)\n    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=20, gamma=0.5)\n    loss_fn = nn.CrossEntropyLoss()\n    best = float(\"inf\")\n    patience, bad_epochs = 8, 0\n    for epoch in range(100):\n        model.train()\n        loss = loss_fn(model(train_x), train_y)\n        opt.zero_grad(); loss.backward(); opt.step(); scheduler.step()\n        model.eval()\n        with torch.no_grad():\n            val_loss = loss_fn(model(val_x), val_y).item()\n        if val_loss < best - 1e-4:\n            best, bad_epochs = val_loss, 0\n        else:\n            bad_epochs += 1\n        if bad_epochs >= patience:\n            break\n    print(\"stopped_epoch:\", epoch + 1, \"best_val_loss:\", best, \"lr:\", scheduler.get_last_lr()[0])\n\n\nif __name__ == \"__main__\":\n    main()\n",
    },
    {
        "slug": "metrics_confusion_matrix",
        "title": "指标：准确率、精确率、召回率、混淆矩阵",
        "principles": [
            "accuracy 适合类别均衡场景，但类别不均衡时可能误导。",
            "precision 关注预测为正的样本中有多少是真的正类。",
            "recall 关注真实正类中有多少被找出来，混淆矩阵能展示错误类型。",
        ],
        "tasks": [
            "给定 y_true 和 y_pred。",
            "计算二分类混淆矩阵 TP/FP/FN/TN。",
            "计算 accuracy、precision、recall。",
        ],
        "starter": "def main():\n    # TODO: 手写二分类指标计算。\n    pass\n\n\nif __name__ == \"__main__\":\n    main()\n",
        "answer": "import torch\n\n\ndef main():\n    y_true = torch.tensor([1, 0, 1, 1, 0, 0, 1, 0])\n    y_pred = torch.tensor([1, 0, 0, 1, 0, 1, 1, 0])\n    tp = ((y_true == 1) & (y_pred == 1)).sum().item()\n    fp = ((y_true == 0) & (y_pred == 1)).sum().item()\n    fn = ((y_true == 1) & (y_pred == 0)).sum().item()\n    tn = ((y_true == 0) & (y_pred == 0)).sum().item()\n    accuracy = (tp + tn) / len(y_true)\n    precision = tp / max(tp + fp, 1)\n    recall = tp / max(tp + fn, 1)\n    print({\"tp\": tp, \"fp\": fp, \"fn\": fn, \"tn\": tn})\n    print(\"accuracy:\", accuracy, \"precision:\", precision, \"recall:\", recall)\n\n\nif __name__ == \"__main__\":\n    main()\n",
    },
    {
        "slug": "embeddings_text",
        "title": "Embedding：把离散 id 变成向量",
        "principles": [
            "Embedding 本质是一个可训练查表矩阵，把 token id 映射成稠密向量。",
            "文本、类别 id、用户 id、商品 id 都常用 embedding 表示。",
            "简单文本分类可以先对词向量求平均，再接线性分类层。",
        ],
        "tasks": [
            "构造整数 token 序列，标签由某个关键词是否出现决定。",
            "使用 nn.Embedding 和 mean pooling。",
            "训练一个二分类模型。",
        ],
        "starter": "def main():\n    # TODO: 用 Embedding + 平均池化完成 toy 文本分类。\n    pass\n\n\nif __name__ == \"__main__\":\n    main()\n",
        "answer": "import torch\n+from torch import nn\n\n\nclass TextClassifier(nn.Module):\n    def __init__(self, vocab_size=20, dim=8):\n        super().__init__()\n        self.embedding = nn.Embedding(vocab_size, dim)\n        self.fc = nn.Linear(dim, 2)\n\n    def forward(self, tokens):\n        emb = self.embedding(tokens)\n        pooled = emb.mean(dim=1)\n        return self.fc(pooled)\n\n\ndef main():\n    torch.manual_seed(0)\n    tokens = torch.randint(0, 20, (300, 6))\n    y = (tokens.eq(7).any(dim=1) | tokens.eq(13).any(dim=1)).long()\n    model = TextClassifier()\n    opt = torch.optim.Adam(model.parameters(), lr=0.03)\n    loss_fn = nn.CrossEntropyLoss()\n    for _ in range(120):\n        loss = loss_fn(model(tokens), y)\n        opt.zero_grad(); loss.backward(); opt.step()\n    print(\"accuracy:\", (model(tokens).argmax(1) == y).float().mean().item())\n\n\nif __name__ == \"__main__\":\n    main()\n",
    },
    {
        "slug": "rnn_sequence",
        "title": "RNN：序列分类入门",
        "principles": [
            "RNN 会按时间步处理序列，隐藏状态携带之前的信息。",
            "输入形状常用 [batch, time, features]，需要设置 batch_first=True。",
            "序列分类常取最后一个时间步的隐藏状态接分类层。",
        ],
        "tasks": [
            "生成长度为 8 的一维序列。",
            "标签为序列后半段均值是否大于前半段均值。",
            "使用 nn.RNN 做二分类。",
        ],
        "starter": "def main():\n    # TODO: 用 nn.RNN 完成序列分类。\n    pass\n\n\nif __name__ == \"__main__\":\n    main()\n",
        "answer": "import torch\n+from torch import nn\n\n\nclass RNNClassifier(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.rnn = nn.RNN(input_size=1, hidden_size=12, batch_first=True)\n        self.fc = nn.Linear(12, 2)\n\n    def forward(self, x):\n        out, _ = self.rnn(x)\n        return self.fc(out[:, -1])\n\n\ndef main():\n    torch.manual_seed(0)\n    x = torch.randn(400, 8, 1)\n    y = (x[:, 4:].mean(dim=(1, 2)) > x[:, :4].mean(dim=(1, 2))).long()\n    model = RNNClassifier()\n    opt = torch.optim.Adam(model.parameters(), lr=0.02)\n    loss_fn = nn.CrossEntropyLoss()\n    for _ in range(100):\n        loss = loss_fn(model(x), y)\n        opt.zero_grad(); loss.backward(); opt.step()\n    print(\"accuracy:\", (model(x).argmax(1) == y).float().mean().item())\n\n\nif __name__ == \"__main__\":\n    main()\n",
    },
    {
        "slug": "lstm_forecasting",
        "title": "LSTM：预测正弦序列下一步",
        "principles": [
            "LSTM 用门控结构缓解普通 RNN 的长程依赖困难。",
            "时间序列预测常用过去若干步作为输入，预测下一步或未来多步。",
            "回归任务输出连续值，常用 MSELoss。",
        ],
        "tasks": [
            "生成 sin 曲线滑动窗口数据。",
            "输入过去 10 步，预测第 11 步。",
            "用 nn.LSTM 和 Linear 完成回归。",
        ],
        "starter": "def main():\n    # TODO: 构造滑动窗口数据，用 LSTM 预测下一步。\n    pass\n\n\nif __name__ == \"__main__\":\n    main()\n",
        "answer": "import math\n+import torch\n+from torch import nn\n\n\nclass Forecaster(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.lstm = nn.LSTM(input_size=1, hidden_size=16, batch_first=True)\n        self.fc = nn.Linear(16, 1)\n\n    def forward(self, x):\n        out, _ = self.lstm(x)\n        return self.fc(out[:, -1])\n\n\ndef main():\n    t = torch.linspace(0, 8 * math.pi, 260)\n    series = torch.sin(t)\n    xs, ys = [], []\n    for i in range(len(series) - 10):\n        xs.append(series[i:i + 10])\n        ys.append(series[i + 10])\n    x = torch.stack(xs).unsqueeze(-1)\n    y = torch.stack(ys).unsqueeze(-1)\n    model = Forecaster()\n    opt = torch.optim.Adam(model.parameters(), lr=0.02)\n    loss_fn = nn.MSELoss()\n    for _ in range(120):\n        loss = loss_fn(model(x), y)\n        opt.zero_grad(); loss.backward(); opt.step()\n    print(\"mse:\", loss.item())\n    print(\"first prediction/target:\", model(x[:1]).item(), y[:1].item())\n\n\nif __name__ == \"__main__\":\n    main()\n",
    },
    {
        "slug": "attention_basics",
        "title": "Attention：Query、Key、Value",
        "principles": [
            "Attention 用 query 和 key 的相似度作为权重，再对 value 做加权求和。",
            "缩放点积注意力会除以 sqrt(d_k)，避免维度变大时 logits 过大。",
            "注意力权重经过 softmax 后，每一行通常和为 1。",
        ],
        "tasks": [
            "创建 q、k、v 三个张量。",
            "计算 scaled dot-product attention。",
            "验证 attention weights 最后一维求和为 1。",
        ],
        "starter": "def scaled_dot_product_attention(q, k, v):\n    # TODO: 返回 output 和 weights。\n    pass\n\n\ndef main():\n    # TODO: 构造输入并测试函数。\n    pass\n\n\nif __name__ == \"__main__\":\n    main()\n",
        "answer": "import math\n+import torch\n\n\ndef scaled_dot_product_attention(q, k, v):\n    scores = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))\n    weights = torch.softmax(scores, dim=-1)\n    output = weights @ v\n    return output, weights\n\n\ndef main():\n    torch.manual_seed(0)\n    q = torch.randn(2, 4, 8)\n    k = torch.randn(2, 4, 8)\n    v = torch.randn(2, 4, 8)\n    output, weights = scaled_dot_product_attention(q, k, v)\n    print(\"output shape:\", output.shape)\n    print(\"weights shape:\", weights.shape)\n    print(\"row sums:\", weights.sum(dim=-1))\n\n\nif __name__ == \"__main__\":\n    main()\n",
    },
    {
        "slug": "transformer_encoder",
        "title": "TransformerEncoder：序列建模小例子",
        "principles": [
            "TransformerEncoder 用自注意力让序列中每个位置都能看见其他位置。",
            "它通常需要 embedding、位置编码或某种位置信息。",
            "nn.TransformerEncoderLayer 可以快速搭出标准 encoder block。",
        ],
        "tasks": [
            "构造 token 序列，标签为第一个 token 是否大于最后一个 token。",
            "使用 Embedding + TransformerEncoder。",
            "取序列表示平均后分类。",
        ],
        "starter": "def main():\n    # TODO: 用 TransformerEncoder 完成 toy 序列分类。\n    pass\n\n\nif __name__ == \"__main__\":\n    main()\n",
        "answer": "import torch\n+from torch import nn\n\n\nclass TinyTransformer(nn.Module):\n    def __init__(self, vocab=30, dim=16):\n        super().__init__()\n        self.embed = nn.Embedding(vocab, dim)\n        self.pos = nn.Parameter(torch.randn(1, 6, dim) * 0.02)\n        layer = nn.TransformerEncoderLayer(d_model=dim, nhead=4, batch_first=True)\n        self.encoder = nn.TransformerEncoder(layer, num_layers=1)\n        self.fc = nn.Linear(dim, 2)\n\n    def forward(self, tokens):\n        x = self.embed(tokens) + self.pos\n        x = self.encoder(x)\n        return self.fc(x.mean(dim=1))\n\n\ndef main():\n    torch.manual_seed(0)\n    tokens = torch.randint(0, 30, (400, 6))\n    y = (tokens[:, 0] > tokens[:, -1]).long()\n    model = TinyTransformer()\n    opt = torch.optim.Adam(model.parameters(), lr=0.02)\n    loss_fn = nn.CrossEntropyLoss()\n    for _ in range(120):\n        loss = loss_fn(model(tokens), y)\n        opt.zero_grad(); loss.backward(); opt.step()\n    print(\"accuracy:\", (model(tokens).argmax(1) == y).float().mean().item())\n\n\nif __name__ == \"__main__\":\n    main()\n",
    },
    {
        "slug": "autoencoder",
        "title": "AutoEncoder：压缩与重建",
        "principles": [
            "自编码器由 encoder 和 decoder 组成，目标是重建输入。",
            "瓶颈层维度较小时，模型被迫学习压缩表示。",
            "重建任务通常使用 MSELoss 或 BCE 类损失，取决于数据范围。",
        ],
        "tasks": [
            "生成二维圆环数据。",
            "用 encoder 把 2 维压到 1 维，再 decoder 重建到 2 维。",
            "训练并打印重建误差。",
        ],
        "starter": "def main():\n    # TODO: 实现并训练一个小 AutoEncoder。\n    pass\n\n\nif __name__ == \"__main__\":\n    main()\n",
        "answer": "import math\n+import torch\n+from torch import nn\n\n\nclass AutoEncoder(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.encoder = nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 1))\n        self.decoder = nn.Sequential(nn.Linear(1, 8), nn.Tanh(), nn.Linear(8, 2))\n\n    def forward(self, x):\n        z = self.encoder(x)\n        return self.decoder(z)\n\n\ndef main():\n    torch.manual_seed(0)\n    t = torch.linspace(0, 2 * math.pi, 300)\n    x = torch.stack([torch.cos(t), torch.sin(t)], dim=1)\n    model = AutoEncoder()\n    opt = torch.optim.Adam(model.parameters(), lr=0.02)\n    loss_fn = nn.MSELoss()\n    for _ in range(250):\n        recon = model(x)\n        loss = loss_fn(recon, x)\n        opt.zero_grad(); loss.backward(); opt.step()\n    print(\"reconstruction_mse:\", loss.item())\n\n\nif __name__ == \"__main__\":\n    main()\n",
    },
    {
        "slug": "vae_reparameterization",
        "title": "VAE：均值、方差与重参数化",
        "principles": [
            "VAE 的 encoder 输出潜变量分布参数 mu 和 logvar，而不是单个确定向量。",
            "重参数化 z=mu+std*eps 让采样过程仍能反向传播到 encoder。",
            "VAE loss 通常由重建误差和 KL 散度组成。",
        ],
        "tasks": [
            "实现 reparameterize(mu, logvar)。",
            "写一个极小 VAE 处理二维数据。",
            "训练时同时计算 recon loss 和 KL loss。",
        ],
        "starter": "def reparameterize(mu, logvar):\n    # TODO: 实现重参数化采样。\n    pass\n\n\ndef main():\n    # TODO: 训练一个 tiny VAE。\n    pass\n\n\nif __name__ == \"__main__\":\n    main()\n",
        "answer": "import torch\n+from torch import nn\n\n\ndef reparameterize(mu, logvar):\n    std = torch.exp(0.5 * logvar)\n    eps = torch.randn_like(std)\n    return mu + std * eps\n\n\nclass VAE(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.enc = nn.Sequential(nn.Linear(2, 16), nn.ReLU())\n        self.mu = nn.Linear(16, 1)\n        self.logvar = nn.Linear(16, 1)\n        self.dec = nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 2))\n\n    def forward(self, x):\n        h = self.enc(x)\n        mu, logvar = self.mu(h), self.logvar(h)\n        z = reparameterize(mu, logvar)\n        return self.dec(z), mu, logvar\n\n\ndef main():\n    torch.manual_seed(0)\n    x = torch.randn(300, 2) * torch.tensor([2.0, 0.5])\n    model = VAE()\n    opt = torch.optim.Adam(model.parameters(), lr=0.02)\n    for _ in range(160):\n        recon, mu, logvar = model(x)\n        recon_loss = ((recon - x) ** 2).mean()\n        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())\n        loss = recon_loss + 0.05 * kl\n        opt.zero_grad(); loss.backward(); opt.step()\n    print(\"loss:\", loss.item(), \"recon:\", recon_loss.item(), \"kl:\", kl.item())\n\n\nif __name__ == \"__main__\":\n    main()\n",
    },
    {
        "slug": "gan_1d",
        "title": "GAN：一维分布生成",
        "principles": [
            "GAN 包含生成器 G 和判别器 D，二者交替训练。",
            "D 学会区分真实样本和生成样本，G 学会骗过 D。",
            "GAN 训练不稳定，本节只做最小可运行例子，理解训练步骤即可。",
        ],
        "tasks": [
            "真实数据来自均值 2、标准差 0.5 的一维高斯。",
            "Generator 把噪声映射成一维样本。",
            "交替训练 D 和 G，并打印生成样本均值。",
        ],
        "starter": "def main():\n    # TODO: 训练一个极小的一维 GAN。\n    pass\n\n\nif __name__ == \"__main__\":\n    main()\n",
        "answer": "import torch\n+from torch import nn\n\n\ndef main():\n    torch.manual_seed(0)\n    g = nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 1))\n    d = nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 1))\n    opt_g = torch.optim.Adam(g.parameters(), lr=0.01)\n    opt_d = torch.optim.Adam(d.parameters(), lr=0.01)\n    loss_fn = nn.BCEWithLogitsLoss()\n    for _ in range(250):\n        real = torch.randn(64, 1) * 0.5 + 2.0\n        noise = torch.randn(64, 1)\n        fake = g(noise).detach()\n        d_loss = loss_fn(d(real), torch.ones(64, 1)) + loss_fn(d(fake), torch.zeros(64, 1))\n        opt_d.zero_grad(); d_loss.backward(); opt_d.step()\n\n        noise = torch.randn(64, 1)\n        fake = g(noise)\n        g_loss = loss_fn(d(fake), torch.ones(64, 1))\n        opt_g.zero_grad(); g_loss.backward(); opt_g.step()\n    samples = g(torch.randn(1000, 1)).detach()\n    print(\"generated mean/std:\", samples.mean().item(), samples.std().item())\n\n\nif __name__ == \"__main__\":\n    main()\n",
    },
    {
        "slug": "gradient_clipping_amp",
        "title": "梯度裁剪与混合精度入口",
        "principles": [
            "梯度裁剪可以限制梯度范数，缓解梯度爆炸，常见于 RNN/Transformer。",
            "混合精度能在 GPU 上用更少显存和更快速度训练，CPU 上通常保持普通精度。",
            "torch.amp 的启用应根据 cuda 是否可用来决定。",
        ],
        "tasks": [
            "训练一个小网络。",
            "反向传播后使用 clip_grad_norm_。",
            "使用 autocast/GradScaler 写出兼容 CPU 的训练步骤。",
        ],
        "starter": "def main():\n    # TODO: 实现带梯度裁剪和可选 AMP 的训练步骤。\n    pass\n\n\nif __name__ == \"__main__\":\n    main()\n",
        "answer": "import torch\n+from torch import nn\n\n\ndef main():\n    device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n    use_amp = device.type == \"cuda\"\n    x = torch.randn(256, 20, device=device)\n    y = (x[:, :5].sum(1) > 0).long().to(device)\n    model = nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 2)).to(device)\n    opt = torch.optim.Adam(model.parameters(), lr=0.02)\n    loss_fn = nn.CrossEntropyLoss()\n    scaler = torch.amp.GradScaler(\"cuda\", enabled=use_amp)\n    for _ in range(50):\n        opt.zero_grad()\n        with torch.amp.autocast(device_type=device.type, enabled=use_amp):\n            loss = loss_fn(model(x), y)\n        scaler.scale(loss).backward()\n        scaler.unscale_(opt)\n        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)\n        scaler.step(opt)\n        scaler.update()\n    print(\"device:\", device, \"amp:\", use_amp, \"last_grad_norm:\", float(grad_norm))\n\n\nif __name__ == \"__main__\":\n    main()\n",
    },
    {
        "slug": "end_to_end_project",
        "title": "端到端小项目：从数据到保存模型",
        "principles": [
            "完整项目通常包含：数据准备、模型、训练循环、验证指标、保存最好模型。",
            "把 train_one_epoch 和 evaluate 拆成函数，会让代码更容易维护。",
            "最终你应该能独立搭起一个小型监督学习项目骨架。",
        ],
        "tasks": [
            "创建一个非线性二分类数据集。",
            "写 train_one_epoch 与 evaluate。",
            "保存验证准确率最高的模型参数。",
        ],
        "starter": "def train_one_epoch(model, x, y, optimizer, loss_fn):\n    # TODO: 完成一个训练 epoch。\n    pass\n\n\ndef evaluate(model, x, y, loss_fn):\n    # TODO: 返回 loss 和 accuracy。\n    pass\n\n\ndef main():\n    # TODO: 串起数据、模型、训练、验证、保存。\n    pass\n\n\nif __name__ == \"__main__\":\n    main()\n",
        "answer": "from pathlib import Path\n+import torch\n+from torch import nn\n\n\ndef train_one_epoch(model, x, y, optimizer, loss_fn):\n    model.train()\n    loss = loss_fn(model(x), y)\n    optimizer.zero_grad(); loss.backward(); optimizer.step()\n    return loss.item()\n\n\ndef evaluate(model, x, y, loss_fn):\n    model.eval()\n    with torch.no_grad():\n        logits = model(x)\n        loss = loss_fn(logits, y).item()\n        acc = (logits.argmax(1) == y).float().mean().item()\n    return loss, acc\n\n\ndef main():\n    torch.manual_seed(0)\n    x = torch.randn(800, 6)\n    y = ((x[:, 0] * x[:, 1] + x[:, 2] - x[:, 3].abs()) > 0).long()\n    train_x, val_x = x[:650], x[650:]\n    train_y, val_y = y[:650], y[650:]\n    model = nn.Sequential(nn.Linear(6, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 2))\n    opt = torch.optim.Adam(model.parameters(), lr=0.02)\n    loss_fn = nn.CrossEntropyLoss()\n    best_acc = 0.0\n    path = Path(__file__).resolve().parents[1] / \"best_project_model.pt\"\n    for epoch in range(120):\n        train_loss = train_one_epoch(model, train_x, train_y, opt, loss_fn)\n        val_loss, val_acc = evaluate(model, val_x, val_y, loss_fn)\n        if val_acc > best_acc:\n            best_acc = val_acc\n            torch.save(model.state_dict(), path)\n    print(\"train_loss:\", train_loss, \"val_loss:\", val_loss, \"best_val_acc:\", best_acc)\n    print(\"saved:\", path)\n\n\nif __name__ == \"__main__\":\n    main()\n",
    },
]


README = """# PyTorch 与深度学习渐进教程

这套材料按从浅到深排列。每节课由三部分组成：

- `docs/tutorial_XX_*.md`：详细教程和关键写法，适合先读和复习。
- `lessons/lesson_XX_*.py`：轻量练习文件，保留题目、路线和代码练习区。
- `answers/answer_XX_*.py`：对应答案。

建议学习节奏：

1. 先读对应的 docs 教程。
2. 在 lesson 文件底部把 `TODO` 补成能运行的 PyTorch 代码。
3. 运行自己的 lesson 文件。
4. 再打开对应 answer 文件对照。
5. 能独立复写答案后，再进入下一节。

运行示例：

```powershell
python .\\pytorch_tutorials\\lessons\\lesson_01_tensor_basics.py
python .\\pytorch_tutorials\\answers\\answer_01_tensor_basics.py
```

课程路线：

{lesson_list}
"""


API_HINTS = {
    "tensor_basics": [
        "`torch.tensor(..., dtype=torch.float32)` 用来创建浮点张量。",
        "`x @ w` 是矩阵乘法，要求左边最后一维等于右边倒数第二维。",
        "`tensor.shape`、`tensor.mean()`、`tensor.max()` 是最常用的检查手段。",
    ],
    "dtype_device_seed": [
        "`torch.manual_seed(seed)` 固定随机数，方便复现实验。",
        "`tensor.to(dtype)` 或 `tensor.float()` 可以转换数据类型。",
        "`torch.device('cuda' if torch.cuda.is_available() else 'cpu')` 是常见设备选择写法。",
    ],
    "reshape_broadcast": [
        "`reshape` 改形状，`unsqueeze` 增加维度，`squeeze` 删除长度为 1 的维度。",
        "广播会从尾部维度开始对齐，长度为 1 的维度可以自动扩展。",
        "遇到广播不确定时，先打印两个张量的 shape。",
    ],
    "indexing_masking": [
        "`x[row, col]` 取具体位置，`:` 表示保留某个维度上的一段范围。",
        "`x[:, :3]` 常用于取所有行的前几列。",
        "`x[x > 10]` 这种写法会用布尔 mask 筛选元素。",
    ],
    "reductions_norms": [
        "`mean(dim=0)` 按行方向聚合，得到每一列的均值。",
        "`keepdim=True` 会保留被聚合的维度，方便后续广播。",
        "`torch.linalg.vector_norm(x, dim=...)` 用来计算向量范数。",
    ],
    "autograd_basics": [
        "`requires_grad=True` 让 PyTorch 跟踪这个张量的计算历史。",
        "`loss.backward()` 从标量 loss 开始反向传播。",
        "梯度会保存在叶子张量的 `.grad` 上。",
    ],
    "manual_gradient_descent": [
        "训练循环一般是 forward -> loss -> backward -> update -> zero grad。",
        "手动改参数时要放在 `with torch.no_grad():` 里。",
        "PyTorch 梯度默认累加，所以每轮更新后要清零。",
    ],
    "nn_linear_regression": [
        "`nn.Linear(in_features, out_features)` 是最基础的全连接层。",
        "`nn.MSELoss()` 常用于回归任务。",
        "`optimizer.zero_grad(); loss.backward(); optimizer.step()` 是标准三连。",
    ],
    "dataloader_basics": [
        "`TensorDataset(x, y)` 可以把两个张量包装成样本集。",
        "`DataLoader(..., batch_size=..., shuffle=True)` 会自动切 batch。",
        "训练循环里通常写 `for xb, yb in loader:`。",
    ],
    "binary_classification": [
        "二分类模型可以输出一个 logit，shape 通常是 `[batch, 1]`。",
        "`nn.BCEWithLogitsLoss()` 直接吃 logits，不要提前 sigmoid。",
        "推理时再用 `torch.sigmoid(logits) > 0.5` 得到类别。",
    ],
    "multiclass_mlp": [
        "多分类输出 shape 是 `[batch, num_classes]`。",
        "`nn.CrossEntropyLoss()` 要求标签是 `torch.long` 的类别编号。",
        "`logits.argmax(dim=1)` 可以取预测类别。",
    ],
    "train_eval_split": [
        "`model.train()` 用于训练阶段，`model.eval()` 用于验证和推理阶段。",
        "`torch.no_grad()` 会关闭梯度记录，让验证更省资源。",
        "训练集更新参数，验证集只评估效果。",
    ],
    "initialization_activation": [
        "自定义模型通常继承 `nn.Module` 并实现 `forward`。",
        "`nn.ReLU()` 提供非线性，避免多层线性层退化成一个线性层。",
        "`nn.init.kaiming_normal_` 常用于 ReLU 网络的权重初始化。",
    ],
    "regularization": [
        "`weight_decay` 可以直接传给优化器。",
        "`nn.Dropout(p)` 只在 train 模式随机丢弃激活。",
        "比较 train/eval 输出时，要记得显式切换模式。",
    ],
    "save_load": [
        "`torch.save(model.state_dict(), path)` 保存参数。",
        "`model.load_state_dict(torch.load(path, map_location='cpu'))` 加载参数。",
        "加载后调用 `eval()`，再做推理比较。",
    ],
    "device_agnostic_training": [
        "模型和数据都要 `.to(device)`，只移动其中一个会报错。",
        "标签也属于数据，同样需要移动到 device。",
        "写 CPU/GPU 通用代码时，不要在中间硬编码 `.cuda()`。",
    ],
    "custom_dataset": [
        "自定义 Dataset 至少实现 `__len__` 和 `__getitem__`。",
        "`__getitem__` 返回一个样本，DataLoader 会把多个样本拼成 batch。",
        "回归任务输出通常是浮点张量，损失常用 MSELoss。",
    ],
    "conv2d_shapes": [
        "PyTorch 图像 batch 常用 `[N, C, H, W]`。",
        "`nn.Conv2d(in_channels, out_channels, kernel_size, padding=...)` 改变通道数。",
        "`nn.MaxPool2d(2)` 常把高宽各减半。",
    ],
    "cnn_synthetic_images": [
        "CNN 通常是 Conv -> ReLU -> Pool 的组合。",
        "`nn.Flatten()` 把卷积特征展平成 Linear 可以接收的向量。",
        "Linear 的输入维度要根据卷积/池化后的 shape 计算。",
    ],
    "batchnorm_dropout": [
        "`nn.BatchNorm1d` 会维护 running_mean 和 running_var。",
        "BatchNorm 和 Dropout 都会受 train/eval 模式影响。",
        "同一个输入在 train 模式下可能产生不同输出。",
    ],
    "scheduler_early_stopping": [
        "`torch.optim.lr_scheduler.StepLR` 可以定期衰减学习率。",
        "验证 loss 变好时更新 best，否则累计 bad_epochs。",
        "达到 patience 后 break，就是最小版 early stopping。",
    ],
    "metrics_confusion_matrix": [
        "TP/FP/FN/TN 可以用布尔条件组合后 `.sum()` 统计。",
        "precision = TP / (TP + FP)，recall = TP / (TP + FN)。",
        "分母可能为 0，实际代码里要做保护。",
    ],
    "embeddings_text": [
        "`nn.Embedding(vocab_size, dim)` 把 token id 变成向量。",
        "`emb.mean(dim=1)` 是最简单的句子级池化方法。",
        "Embedding 输入必须是整数 id，通常是 long 类型。",
    ],
    "rnn_sequence": [
        "`nn.RNN(..., batch_first=True)` 接收 `[batch, time, features]`。",
        "RNN 输出 `out` 包含每个时间步的隐藏状态。",
        "序列分类常用 `out[:, -1]` 取最后一步表示。",
    ],
    "lstm_forecasting": [
        "`nn.LSTM` 的输入形状和 RNN 类似。",
        "滑动窗口数据可以用过去若干步预测下一步。",
        "时间序列回归常用 `nn.MSELoss()`。",
    ],
    "attention_basics": [
        "注意力分数常写成 `q @ k.transpose(-2, -1)`。",
        "除以 `sqrt(d_k)` 可以控制分数尺度。",
        "`softmax(scores, dim=-1)` 得到每一行和为 1 的权重。",
    ],
    "transformer_encoder": [
        "`nn.TransformerEncoderLayer(..., batch_first=True)` 是基本 encoder block。",
        "token 先经过 Embedding，再加位置信息。",
        "分类时可以对序列维度求平均，也可以取某个特殊位置。",
    ],
    "autoencoder": [
        "encoder 把输入压缩到低维 latent，decoder 从 latent 重建输入。",
        "训练目标是让 `recon` 接近原始 `x`。",
        "重建误差可以用 MSELoss。",
    ],
    "vae_reparameterization": [
        "encoder 输出 `mu` 和 `logvar`。",
        "`std = exp(0.5 * logvar)`，再用 `mu + std * eps` 采样。",
        "VAE loss 通常是重建误差加 KL 散度。",
    ],
    "gan_1d": [
        "Generator 从噪声生成假样本，Discriminator 判断真假。",
        "训练 D 时要对 fake 使用 `.detach()`，避免更新 G。",
        "训练 G 时希望 D 把 fake 判断为真。",
    ],
    "gradient_clipping_amp": [
        "`nn.utils.clip_grad_norm_` 用于限制梯度范数。",
        "`torch.amp.autocast` 和 `torch.amp.GradScaler` 是新版混合精度入口。",
        "CPU 上通常关闭 AMP，CUDA 可用时再启用。",
    ],
    "end_to_end_project": [
        "把训练和评估拆成函数，项目会更容易扩展。",
        "验证指标变好时保存 `state_dict`，这叫保存最好模型。",
        "完整流程要包含数据、模型、损失、优化器、训练、验证、保存。",
    ],
}


API_HINTS.update({
    "tensor_basics": [
        "`import torch` 是所有 PyTorch 脚本的起点，通常先放在文件顶部。",
        "`torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)` 从 Python 列表创建张量，并指定浮点类型。",
        "`torch.zeros(shape)`、`torch.ones(shape)`、`torch.full(shape, value)` 分别创建全 0、全 1、指定值张量。",
        "`torch.arange(start, end, step)` 创建等差整数序列；`torch.linspace(start, end, steps)` 创建等距浮点序列。",
        "`torch.randn(shape)` 创建标准正态随机张量；`torch.rand(shape)` 创建 0 到 1 的均匀随机张量。",
        "`x + y`、`x - y`、`x * y`、`x / y` 是逐元素运算，要求形状相同或可以广播。",
        "`x @ w` 或 `torch.matmul(x, w)` 是矩阵乘法；二维时要求 `x.shape[1] == w.shape[0]`。",
        "`x.T` 可以转置二维矩阵；高维张量更常用 `transpose(dim0, dim1)` 或 `permute(...)`。",
        "`x.shape`、`x.ndim`、`x.numel()` 分别查看形状、维度数、元素总数。",
        "`x.mean()`、`x.sum()`、`x.max()`、`x.min()`、`x.std()` 是最常见统计操作。",
        "`x.item()` 把只含一个元素的张量转成 Python 数字，常用于打印 loss。",
        "`print(x, x.shape, x.dtype, x.device)` 是排查 PyTorch 代码最朴素也最有效的方法。",
    ],
    "dtype_device_seed": [
        "`torch.manual_seed(42)` 固定 CPU 随机数，让初始化和随机数据更容易复现。",
        "如果使用 CUDA，也常见写法是 `torch.cuda.manual_seed_all(42)`，用于多 GPU 随机种子。",
        "`dtype` 决定数值类型；模型输入常用 `torch.float32`，分类标签常用 `torch.long`。",
        "`x.float()`、`x.long()`、`x.double()` 是常见 dtype 快捷转换。",
        "`x.to(torch.float32)` 可以转换 dtype；`x.to(device)` 可以移动设备；`x.to(device=device, dtype=torch.float32)` 可同时处理。",
        "`device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')` 是 CPU/GPU 通用代码入口。",
        "模型和数据必须在同一个 device 上，`model.to(device)` 和 `batch.to(device)` 通常要配套出现。",
        "`torch.randn(2, 3, device=device, dtype=torch.float32)` 可以创建时直接指定 device 和 dtype。",
        "`x.cpu()` 把张量移回 CPU；转 NumPy 前通常需要 `x.detach().cpu().numpy()`。",
        "常见错误 `Expected all tensors to be on the same device` 基本就是模型、输入、标签有东西没搬到同一个设备。",
        "常见错误 `expected scalar type Long` 多半是分类标签 dtype 不对，CrossEntropyLoss 要 long 类别编号。",
    ],
    "reshape_broadcast": [
        "`x.reshape(new_shape)` 改变形状，元素总数必须不变，例如 12 个元素可以变成 `(3, 4)`。",
        "`x.view(...)` 也能改形状，但要求内存连续；不确定时优先用 `reshape`。",
        "`x.flatten()` 展平成一维；`nn.Flatten()` 常放在 CNN 进入 Linear 前。",
        "`x.unsqueeze(dim)` 增加长度为 1 的维度，例如 `[3] -> [1, 3]` 或 `[3, 1]`。",
        "`x.squeeze(dim)` 删除指定的长度为 1 的维度；不指定 dim 会删掉所有长度为 1 的维度，使用时要小心 batch 维。",
        "`x.permute(0, 2, 3, 1)` 按任意顺序重排维度；图像数据经常在 NHWC 和 NCHW 间转换。",
        "`x.transpose(dim0, dim1)` 交换两个维度；二维矩阵也可以用 `x.T`。",
        "广播规则从最后一维开始对齐；两个维度相等，或其中一个为 1，才可以广播。",
        "`bias` 形状是 `[features]` 时，可以自动加到 `[batch, features]` 的每一行。",
        "`keepdim=True` 可以保留聚合后的维度，常用于 `(x - mean) / std` 这种标准化写法。",
        "如果广播报错，先打印参与运算的两个 shape，从最后一维往前对齐检查。",
    ],
    "indexing_masking": [
        "`x[i]` 取第 i 行或第 i 个元素；索引从 0 开始，`x[-1]` 表示最后一个。",
        "`x[:, 0]` 取所有行的第 0 列；`x[1, :]` 取第 1 行所有列。",
        "`x[:, :3]` 取所有行的前 3 列；`x[::2]` 每隔一个取一次。",
        "`x[start:end]` 左闭右开，不包含 `end` 位置。",
        "`mask = x > 10` 会得到布尔张量；`x[mask]` 会筛出满足条件的元素。",
        "`torch.where(condition, a, b)` 按条件从 a/b 中选值，适合做条件替换。",
        "`x.argmax(dim=1)` 常用于从分类 logits 得到预测类别。",
        "`x.gather(dim, index)` 可以按 index 收集元素，常见于更复杂的分类或序列任务。",
        "`x.scatter_(dim, index, value)` 可以按 index 写入，函数名带 `_` 表示原地修改。",
        "原地修改如 `x[mask] = 0` 很方便，但在 autograd 计算图里要谨慎，可能破坏梯度需要的中间值。",
    ],
    "reductions_norms": [
        "`x.mean()` 对所有元素求均值；`x.mean(dim=0)` 沿第 0 维聚合，常得到每列均值。",
        "`x.sum(dim=1)` 对每一行求和，常用于统计每个样本的特征总量。",
        "`x.max(dim=1)` 返回 `(values, indices)`，多分类里更常用 `x.argmax(dim=1)`。",
        "`keepdim=True` 保留被聚合维度，方便后续和原张量广播运算。",
        "`x.std(dim=0)` 计算标准差；标准化常写成 `(x - mean) / (std + 1e-8)`。",
        "`torch.linalg.vector_norm(x, dim=1)` 计算每行向量范数。",
        "`torch.clamp(x, min=0, max=1)` 可以把数值限制在区间内。",
        "`torch.nan_to_num(x)` 可以把 NaN/Inf 替换成有限数，在排查坏数据时有用。",
        "`dim` 是 PyTorch 最重要的参数之一；不知道该写几，就先把 shape 按维度编号写在纸上。",
        "统计值参与训练时仍然是张量；只打印时再 `.item()`，不要过早把它变成 Python 数字。",
    ],
    "autograd_basics": [
        "`x = torch.tensor(1.0, requires_grad=True)` 会让 PyTorch 记录 x 参与的计算。",
        "只有浮点或复数张量能 `requires_grad=True`，整数标签不能求梯度。",
        "`loss.backward()` 要求 loss 通常是标量；非标量需要传入 `gradient` 参数。",
        "叶子张量的梯度保存在 `.grad`，例如 `w.grad`。",
        "PyTorch 默认梯度累加，所以训练循环里必须清空旧梯度。",
        "`with torch.no_grad():` 里面的计算不会被 autograd 记录，常用于手动更新参数和推理。",
        "`x.detach()` 得到一个不再连接当前计算图的新张量，常用于停止梯度传播。",
        "`loss.item()` 只用于日志，不要用 `.item()` 后的 Python 数字继续参与反向传播。",
        "如果看到 `element 0 of tensors does not require grad`，说明 loss 和可训练参数之间的计算图断了。",
        "如果遇到原地操作报错，检查是否对需要梯度的中间张量用了带 `_` 的方法或切片赋值。",
    ],
    "manual_gradient_descent": [
        "最小训练循环是 `pred -> loss -> zero_grad -> backward -> step`，手写版则把 `step` 换成手动减梯度。",
        "手动参数需要 `requires_grad=True`，例如 `w = torch.randn(1, 1, requires_grad=True)`。",
        "线性回归预测常写成 `pred = x @ w + b`，其中 `x` 是 `[batch, features]`。",
        "MSE 可手写为 `((pred - y) ** 2).mean()`。",
        "参数更新要放进 `with torch.no_grad(): w -= lr * w.grad`，否则更新本身也会被记录进计算图。",
        "更新后要 `w.grad.zero_()` 和 `b.grad.zero_()`，否则下一轮梯度会叠加。",
        "`lr` 太大 loss 可能变成 NaN 或震荡；太小则收敛很慢。",
        "训练时可以每隔若干 epoch 打印 loss，观察是否下降。",
        "真实项目更常用 optimizer，但手写一次能帮你理解 optimizer 到底替你做了什么。",
    ],
    "nn_linear_regression": [
        "`from torch import nn` 后，可以用 `nn.Module`、`nn.Linear`、各种 loss 和层。",
        "`nn.Linear(in_features, out_features)` 输入 shape 是 `[batch, in_features]`，输出是 `[batch, out_features]`。",
        "`model.parameters()` 会返回模型里需要优化的参数，直接交给 optimizer。",
        "`nn.MSELoss()` 适合回归，预测和标签 shape 通常要一致。",
        "`torch.optim.SGD(model.parameters(), lr=...)` 是最基础优化器；Adam 通常更省调参。",
        "标准顺序是 `opt.zero_grad(); loss.backward(); opt.step()`。",
        "`model.weight` 和 `model.bias` 可以查看 Linear 的参数。",
        "用 `nn.Sequential(...)` 可以快速堆简单模型；复杂模型建议自定义 `nn.Module`。",
        "训练前可用 `print(model)` 查看模型结构。",
        "回归输出层一般不加 sigmoid/softmax，直接输出连续值。",
    ],
    "dataloader_basics": [
        "`TensorDataset(x, y)` 把多个张量按第 0 维对齐，组成样本集。",
        "`DataLoader(dataset, batch_size=32, shuffle=True)` 自动切 batch 并打乱训练数据。",
        "`for xb, yb in loader:` 是 mini-batch 训练的基本写法。",
        "Dataset 里的第 0 维通常是样本数，`len(dataset)` 返回样本数量。",
        "`shuffle=True` 常用于训练集；验证/测试集一般设为 False。",
        "每个 batch 内仍要做完整训练三连：`zero_grad -> backward -> step`。",
        "如果使用 GPU，要在循环里把每个 `xb`、`yb` 移到 device。",
        "`drop_last=True` 可以丢弃最后一个不满 batch 的小批次，BatchNorm 或固定形状场景可能有用。",
        "Windows 上自定义多进程 DataLoader 时要注意 `if __name__ == '__main__':` 保护。",
        "刚入门时 `num_workers=0` 最稳，等流程跑通再考虑加速数据读取。",
    ],
    "binary_classification": [
        "二分类可输出一个 logit，shape `[batch, 1]`，标签也整理成 `[batch, 1]` 的 float。",
        "`nn.BCEWithLogitsLoss()` 内部已经包含 sigmoid，训练时不要手动 sigmoid。",
        "推理阶段用 `probs = torch.sigmoid(logits)` 得到正类概率。",
        "`pred = probs > 0.5` 是默认阈值，也可以按业务需求调阈值。",
        "标签要是 0/1 浮点数；如果是 bool 或 long，通常用 `.float()` 转换。",
        "准确率可写成 `((pred == y.bool()).float().mean())`。",
        "如果输出两个 logits，也可以把二分类当多分类，用 `CrossEntropyLoss` 和 long 标签。",
        "类别极不均衡时，accuracy 可能虚高，要关注 precision、recall 或 AUC。",
        "`pos_weight` 参数可以让 BCEWithLogitsLoss 更重视正类，适合正负样本不均衡。",
        "logits 不是概率，可以小于 0 或大于 1；只有 sigmoid 后才是 0 到 1。",
    ],
    "multiclass_mlp": [
        "多分类输出 shape 是 `[batch, num_classes]`，每一列对应一个类别 logit。",
        "`nn.CrossEntropyLoss()` 输入 raw logits，不要先 softmax。",
        "CrossEntropyLoss 的标签 shape 通常是 `[batch]`，dtype 必须是 `torch.long`。",
        "`pred = logits.argmax(dim=1)` 得到预测类别编号。",
        "MLP 常见结构是 `Linear -> ReLU -> Linear`，隐藏层可以重复堆叠。",
        "`nn.Sequential` 适合顺序结构；需要分支或多输入时自定义 `nn.Module`。",
        "隐藏层宽度如 16、32、64 是常见起点，不必一开始就很大。",
        "分类准确率可写成 `(pred == y).float().mean().item()`。",
        "如果 loss 不降，先检查标签是否从 0 到 `num_classes-1`，以及标签 dtype 是否 long。",
        "评估概率时再用 `torch.softmax(logits, dim=1)`。",
    ],
    "train_eval_split": [
        "训练集用于 `backward` 和 `optimizer.step()`，验证集只用于评估。",
        "`model.train()` 会打开 Dropout，并让 BatchNorm 使用当前 batch 统计。",
        "`model.eval()` 会关闭 Dropout，并让 BatchNorm 使用 running statistics。",
        "验证代码通常写在 `with torch.no_grad():` 中，避免构建计算图。",
        "切分数据可先用简单切片；正式项目可用 `random_split` 或 sklearn 的 train_test_split。",
        "每个 epoch 后记录 train loss、val loss、val accuracy，更容易发现过拟合。",
        "训练 loss 降、验证 loss 升，通常是过拟合信号。",
        "验证时不要调用 `optimizer.step()`，也不要对验证 loss 做 backward。",
        "如果使用 DataLoader，训练 loader 可 shuffle，验证 loader 不需要 shuffle。",
        "保存模型时通常保存验证集表现最好的版本，而不是最后一轮。",
    ],
    "initialization_activation": [
        "自定义模型写法：`class MLP(nn.Module): __init__` 定义层，`forward` 定义计算。",
        "`super().__init__()` 必须调用，否则 PyTorch 不能正确注册参数和子模块。",
        "激活函数如 `ReLU`、`Tanh`、`GELU` 负责引入非线性。",
        "没有激活函数的多层 Linear 仍等价于一个 Linear，表达能力有限。",
        "`nn.init.kaiming_normal_(layer.weight)` 常配合 ReLU。",
        "`nn.init.xavier_uniform_(layer.weight)` 常配合 Tanh/Sigmoid 或一般全连接网络。",
        "`nn.init.zeros_(layer.bias)` 是常见 bias 初始化。",
        "遍历模块可用 `for layer in model.modules():`，再用 `isinstance(layer, nn.Linear)` 筛选。",
        "初始化通常在模型创建后、训练开始前做一次。",
        "过深网络还会涉及残差连接、归一化层和更谨慎的初始化策略。",
    ],
    "regularization": [
        "`weight_decay=1e-4` 可以直接写进 Adam/SGD，给权重加 L2 正则效果。",
        "`nn.Dropout(p=0.5)` 在训练时随机把一部分激活置 0。",
        "Dropout 放在隐藏层后更常见，通常不放在最终输出 logits 后。",
        "`model.train()` 时 Dropout 随机；`model.eval()` 时 Dropout 关闭。",
        "过拟合表现通常是 train 指标很好、val 指标明显差。",
        "减小模型、增加数据、数据增强、weight decay、Dropout 都是常见正则化手段。",
        "weight decay 太大可能欠拟合，表现为训练集也学不好。",
        "Dropout 太大也会欠拟合，小模型上不一定需要 Dropout。",
        "正则化不是越多越好，要看验证集表现。",
        "调参时一次只改一两个因素，否则很难判断是谁起作用。",
    ],
    "save_load": [
        "`torch.save(model.state_dict(), path)` 是推荐保存方式，只保存参数和 buffer。",
        "加载时先创建同结构模型：`model = MyModel(); model.load_state_dict(torch.load(path))`。",
        "`map_location='cpu'` 可以把 GPU 保存的参数加载到 CPU。",
        "推理前调用 `model.eval()`，保证 Dropout/BatchNorm 行为正确。",
        "如果还要恢复训练，应同时保存 optimizer state、epoch、best metric。",
        "checkpoint 常见结构：`{'model': model.state_dict(), 'optimizer': opt.state_dict(), 'epoch': epoch}`。",
        "`strict=False` 可以在模型结构略有变化时加载部分参数，但要仔细检查 missing/unexpected keys。",
        "只保存整个 `model` 对象不够稳，代码结构变化后更容易加载失败。",
        "保存路径用 `pathlib.Path` 更跨平台，也方便创建目录。",
        "加载后可用同一输入比较两个模型输出，确认参数恢复正确。",
    ],
    "device_agnostic_training": [
        "统一写 `device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')`。",
        "模型创建后调用 `model.to(device)`。",
        "每个 batch 进入模型前调用 `xb = xb.to(device); yb = yb.to(device)`。",
        "新创建的临时张量也要注意 device，可用 `torch.zeros_like(x)` 或指定 `device=device`。",
        "不要在通用代码里硬写 `.cuda()`，没有 GPU 的机器会直接失败。",
        "日志里的 loss 可用 `loss.item()`，不用把整个模型或 batch 搬回 CPU。",
        "转 NumPy 前必须在 CPU 上：`x.detach().cpu().numpy()`。",
        "保存 state_dict 时不需要特别处理 device；加载时可用 `map_location` 控制。",
        "device mismatch 错误几乎都能通过打印 `tensor.device` 定位。",
        "多 GPU、分布式训练是更高阶主题，单机入门先把单 device 写规范。",
    ],
    "custom_dataset": [
        "自定义 Dataset 继承 `torch.utils.data.Dataset`。",
        "`__len__(self)` 返回样本数，DataLoader 依赖它判断一轮有多少数据。",
        "`__getitem__(self, idx)` 返回第 idx 个样本，可以是 `(x, y)`、字典或更多字段。",
        "数据可以在 `__init__` 中提前准备，也可以在 `__getitem__` 中按需读取。",
        "返回的数值最好转换成张量，避免训练循环里到处做类型转换。",
        "`DataLoader` 会把多个样本自动 collate 成 batch；形状不一致时需要自定义 `collate_fn`。",
        "回归标签通常是 float，分类标签通常是 long。",
        "数据增强常放在 Dataset 的 `__getitem__` 中。",
        "Dataset 不负责训练，只负责稳定地提供样本。",
        "先用小数据和 `loader = DataLoader(dataset, batch_size=4)` 打印一个 batch，确认 shape。",
    ],
    "conv2d_shapes": [
        "PyTorch 卷积输入是 `[N, C, H, W]`，不是 `[N, H, W, C]`。",
        "`nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)` 是基本写法。",
        "`out_channels` 决定输出通道数，也就是学多少个卷积核。",
        "`padding=1, kernel_size=3, stride=1` 常保持高宽不变。",
        "`nn.MaxPool2d(2)` 会把高宽各减半，通道数不变。",
        "`nn.AdaptiveAvgPool2d((1, 1))` 可把任意高宽压到 1x1，便于接分类头。",
        "卷积输出高宽公式大致是 `(H + 2P - K) / S + 1`，向下取整。",
        "进入 Linear 前用 `nn.Flatten()`，并确认 flatten 后维度。",
        "图像从 PIL/NumPy 来时常是 HWC，需要转换到 CHW。",
        "排查 CNN 第一件事：打印每层输出 shape。",
    ],
    "cnn_synthetic_images": [
        "典型 CNN 分类器：`Conv2d -> ReLU -> Pool -> Conv2d -> ReLU -> Pool -> Flatten -> Linear`。",
        "小图像任务可以先用 8、16、32 个通道，不必一开始很大。",
        "卷积层学习局部模式，池化层降低空间尺寸并扩大感受野。",
        "`nn.Sequential` 很适合写直线型 CNN。",
        "Flatten 后 Linear 输入维度等于 `channels * height * width`。",
        "如果不想手算维度，可以先跑一个 dummy input 打印 shape。",
        "分类输出层维度等于类别数。",
        "CNN 输入一般要归一化到合理范围，例如 0-1 或均值方差标准化。",
        "合成数据适合练流程，但真实图像还要考虑数据增强和 train/val/test 切分。",
        "训练准确率 1.0 不代表泛化好，要看独立验证集。",
    ],
    "batchnorm_dropout": [
        "`nn.BatchNorm1d(num_features)` 常用于 MLP 的 `[batch, features]`。",
        "`nn.BatchNorm2d(num_channels)` 常用于 CNN 的 `[N, C, H, W]`。",
        "BatchNorm 训练时用当前 batch 统计，同时更新 running_mean/running_var。",
        "BatchNorm eval 时使用 running_mean/running_var，不再用当前 batch 统计。",
        "Dropout train 时随机，eval 时关闭。",
        "因此训练前 `model.train()`，验证/推理前 `model.eval()` 是必须习惯。",
        "小 batch 下 BatchNorm 统计可能不稳定，可考虑 LayerNorm 或 GroupNorm。",
        "BatchNorm 一般放在 Linear/Conv 后、激活函数前或后，具体结构可按常见架构习惯。",
        "比较两个输出是否一样时要先固定模式，否则 Dropout 会让结果随机。",
        "不要在验证阶段忘记 eval，否则指标会飘。",
    ],
    "scheduler_early_stopping": [
        "`StepLR(optimizer, step_size=20, gamma=0.5)` 每 20 轮把学习率乘 0.5。",
        "常见调用位置是每个 epoch 训练结束后 `scheduler.step()`。",
        "`scheduler.get_last_lr()` 可以查看当前学习率。",
        "早停需要记录 `best_val_loss` 或 `best_val_acc`。",
        "如果指标提升，保存模型并把 `bad_epochs = 0`；否则 `bad_epochs += 1`。",
        "`patience` 表示允许多少轮不提升。",
        "比较浮点指标时常加一个最小改善阈值，比如 `1e-4`。",
        "val loss 适合早停，val accuracy 也可以，但 accuracy 可能更抖。",
        "ReduceLROnPlateau 是另一种常见调度器，它根据验证指标自动降学习率。",
        "scheduler 和 early stopping 都属于训练策略，不改变模型 forward 结构。",
    ],
    "metrics_confusion_matrix": [
        "`tp = ((y_true == 1) & (y_pred == 1)).sum()` 统计真正例。",
        "`fp` 是真实为 0 但预测为 1；`fn` 是真实为 1 但预测为 0；`tn` 是真实为 0 且预测为 0。",
        "`accuracy = (tp + tn) / total` 表示总体预测正确比例。",
        "`precision = tp / (tp + fp)` 表示预测为正的样本里多少是真的正。",
        "`recall = tp / (tp + fn)` 表示真实正样本里多少被找出来。",
        "`f1 = 2 * precision * recall / (precision + recall)` 平衡 precision 和 recall。",
        "类别不均衡时 accuracy 可能误导，要同时看 precision/recall/F1。",
        "多分类混淆矩阵可以建立 `[num_classes, num_classes]` 的计数矩阵。",
        "指标计算通常放在 `torch.no_grad()` 和 `model.eval()` 下。",
        "实际项目要明确哪个类别是正类，否则 precision/recall 的含义会混乱。",
    ],
    "embeddings_text": [
        "`nn.Embedding(vocab_size, embedding_dim)` 输入 token id，输出 token 向量。",
        "Embedding 输入必须是整数张量，dtype 通常是 `torch.long`。",
        "输入 shape `[batch, seq_len]` 经过 Embedding 后变成 `[batch, seq_len, dim]`。",
        "`emb.mean(dim=1)` 是平均池化，可把序列变成句向量。",
        "有 padding 时不能直接 mean，需要用 mask 排除 padding 位置。",
        "`padding_idx=0` 可以让某个 token 作为 padding，并让它的 embedding 不更新。",
        "文本分类头常写成 `Embedding -> pooling -> Linear`。",
        "词表大小 vocab_size 必须大于最大 token id。",
        "Embedding 也可用于用户 id、商品 id、类别 id，不只用于文本。",
        "预训练词向量可以赋给 `embedding.weight.data`，但入门先练随机初始化即可。",
    ],
    "rnn_sequence": [
        "`nn.RNN(input_size, hidden_size, batch_first=True)` 接收 `[batch, time, features]`。",
        "RNN 返回 `(out, h_n)`；`out` 是每个时间步输出，`h_n` 是最后隐藏状态。",
        "序列分类常用 `out[:, -1, :]` 或 `h_n[-1]` 作为整体表示。",
        "输入一维数值序列时，features 维也要保留，例如 `[batch, time, 1]`。",
        "`hidden_size` 决定隐藏状态维度，也是分类头 Linear 的输入维度。",
        "普通 RNN 容易梯度消失，长序列更常用 LSTM/GRU/Transformer。",
        "变长序列可以用 padding + mask，或 `pack_padded_sequence`。",
        "RNN 训练也使用普通的 optimizer/loss/backward/step。",
        "序列任务排错重点是 `[batch, time, features]` 维度顺序。",
        "如果 `batch_first=False`，输入顺序会变成 `[time, batch, features]`，初学建议设 True。",
    ],
    "lstm_forecasting": [
        "`nn.LSTM(input_size, hidden_size, batch_first=True)` 是门控循环网络。",
        "LSTM 返回 `(out, (h_n, c_n))`，其中 c_n 是 cell state。",
        "预测下一步常取 `out[:, -1, :]` 接 `nn.Linear(hidden_size, output_size)`。",
        "滑动窗口构造：`x[i:i+window]` 作为输入，`x[i+window]` 作为标签。",
        "时间序列输入 shape 常是 `[samples, window, features]`。",
        "回归标签 shape 要和模型输出一致，例如 `[samples, 1]`。",
        "损失常用 `nn.MSELoss()` 或 `nn.L1Loss()`。",
        "序列数值通常需要标准化，真实项目里尤其重要。",
        "预测多步未来可以让输出层输出多个值，或递归地一步步预测。",
        "LSTM 也可能梯度爆炸，必要时配合梯度裁剪。",
    ],
    "attention_basics": [
        "Q/K/V 的最后一维是特征维，常记作 `d_k` 或 `d_model`。",
        "`scores = q @ k.transpose(-2, -1)` 得到每个 query 对每个 key 的相似度。",
        "`scores / math.sqrt(q.size(-1))` 是缩放点积注意力的标准写法。",
        "`weights = torch.softmax(scores, dim=-1)` 得到注意力权重。",
        "`output = weights @ v` 用权重对 value 加权求和。",
        "输入可为 `[batch, seq, dim]`，输出通常也是 `[batch, seq, dim]`。",
        "mask 可在 softmax 前把不可见位置填成很小的数，如 `-1e9`。",
        "自注意力是 q/k/v 都来自同一个序列；交叉注意力是 q 和 k/v 来自不同序列。",
        "多头注意力会把特征维拆成多个 head 并行计算。",
        "排查 attention 时检查 weights 最后一维求和是否接近 1。",
    ],
    "transformer_encoder": [
        "`nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)` 创建一个 encoder 层。",
        "`nn.TransformerEncoder(layer, num_layers=...)` 堆叠多个 encoder 层。",
        "输入到 Transformer 前通常先经过 `nn.Embedding`，shape `[batch, seq] -> [batch, seq, dim]`。",
        "Transformer 本身不知道顺序，需要加位置编码或可学习位置参数。",
        "`d_model` 必须能被 `nhead` 整除。",
        "分类任务可用平均池化、最后位置、或专门的 CLS token 得到序列表示。",
        "padding 序列时要传 `src_key_padding_mask`，避免模型关注 padding。",
        "Transformer 输出 shape 和输入 embedding shape 通常一致。",
        "小数据上 Transformer 不一定比 MLP/RNN 好，先理解机制更重要。",
        "如果训练很慢，可以先减小 dim、层数和序列长度。",
    ],
    "autoencoder": [
        "AutoEncoder 通常写成 `encoder` 和 `decoder` 两个子网络。",
        "`z = encoder(x)` 是压缩表示，`recon = decoder(z)` 是重建结果。",
        "瓶颈层 latent 维度越小，压缩压力越大，重建可能更难。",
        "连续值重建常用 `nn.MSELoss()`；0-1 图像也可考虑 BCE 类损失。",
        "decoder 最后一层是否加激活取决于数据范围，例如 0-1 可加 Sigmoid。",
        "自编码器不需要标签，是自监督/无监督风格的训练。",
        "训练循环和普通回归类似，只是 label 就是输入 x 自己。",
        "可以用 latent 做可视化、异常检测或下游特征。",
        "如果 recon 很差，尝试加大 latent 维度或模型容量。",
        "如果只会复制输入，说明压缩或正则约束可能不够。",
    ],
    "vae_reparameterization": [
        "VAE encoder 输出 `mu` 和 `logvar`，表示潜变量高斯分布参数。",
        "`std = torch.exp(0.5 * logvar)` 把 log 方差转成标准差。",
        "`eps = torch.randn_like(std)` 采样标准正态噪声。",
        "`z = mu + std * eps` 是重参数化技巧，使采样过程可反向传播。",
        "重建损失衡量 decoder 输出和输入的接近程度。",
        "KL loss 让潜变量分布接近标准正态，常写成 `-0.5 * mean(1 + logvar - mu^2 - exp(logvar))`。",
        "总 loss 通常是 `recon_loss + beta * kl`，beta 可调节压缩/生成约束强度。",
        "logvar 比直接预测 variance 更稳定，也更不容易出现负方差。",
        "训练 VAE 时 recon 和 KL 要分别打印，方便判断哪一项主导。",
        "VAE 训练好后可以从标准正态采样 z，再用 decoder 生成数据。",
    ],
    "gan_1d": [
        "GAN 有两个模型：Generator 负责造假，Discriminator 负责辨别真假。",
        "训练 D：真实样本标签是 1，生成样本标签是 0。",
        "训练 D 时 `fake = G(noise).detach()`，避免 D 的 loss 更新到 G。",
        "训练 G：希望 `D(G(noise))` 被判断为真，所以目标标签是 1。",
        "D 和 G 通常各有自己的 optimizer。",
        "`BCEWithLogitsLoss` 适合判别器输出 raw logit 的写法。",
        "GAN 训练可能不稳定，loss 不一定像监督学习那样单调下降。",
        "可以观察生成样本的均值、方差或可视化分布判断效果。",
        "D 太强时 G 学不到东西；G/D 学习率和训练次数需要平衡。",
        "真实图像 GAN 会复杂很多，本节先掌握交替训练流程。",
    ],
    "gradient_clipping_amp": [
        "`loss.backward()` 后、`optimizer.step()` 前进行梯度裁剪。",
        "`nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` 限制整体梯度范数。",
        "梯度裁剪常用于 RNN、LSTM、Transformer，缓解梯度爆炸。",
        "`torch.amp.autocast(device_type='cuda', enabled=True)` 会在合适操作上使用混合精度。",
        "`torch.amp.GradScaler('cuda', enabled=use_amp)` 用于缩放 loss，减少 float16 下梯度下溢。",
        "AMP 典型顺序：autocast 前向 -> scaler.scale(loss).backward() -> scaler.step(opt) -> scaler.update()。",
        "如果要裁剪梯度，先 `scaler.unscale_(optimizer)`，再 clip。",
        "CPU 上通常 `enabled=False`，保持普通 float32 训练。",
        "不是所有模型都适合 AMP；遇到 NaN 时先关 AMP 排查。",
        "即使不用 AMP，梯度裁剪仍然可以单独使用。",
    ],
    "end_to_end_project": [
        "端到端脚本通常包含：配置、数据、模型、loss、optimizer、train、evaluate、save。",
        "`train_one_epoch` 负责 train 模式、前向、loss、反向、更新，并返回训练日志。",
        "`evaluate` 负责 eval 模式、no_grad、计算 loss 和指标，不更新参数。",
        "主循环里每个 epoch 调一次训练和验证。",
        "用 `best_val_acc` 或 `best_val_loss` 判断是否保存模型。",
        "`torch.save(model.state_dict(), path)` 保存验证集最好的参数。",
        "实验要固定随机种子，便于比较不同改动。",
        "每次改模型/学习率/batch size 后，都用同一验证集比较。",
        "日志至少打印 epoch、train_loss、val_loss、val_acc。",
        "项目变大后，可以把 dataset、model、train utils 拆成不同文件，但先把单文件流程写熟。",
    ],
})


def lesson_text(index, lesson):
    lines = []
    tutorial_name = f"tutorial_{index:02d}_{lesson['slug']}.md"
    lines.append(f"# 第 {index:02d} 课：{lesson['title']}")
    lines.append("#")
    lines.append("# 核心讲解")
    lines.append("# 这里保留最短版本，详细教程请先读：")
    lines.append(f"# ../docs/{tutorial_name}")
    lines.append("#")
    for i, item in enumerate(lesson["principles"], start=1):
        lines.append(f"# {i}. {item}")
    lines.append("#")
    lines.append("# 操作路线")
    lines.append("# 写本节代码时，可以按这个顺序推进：")
    for i, item in enumerate(lesson["tasks"], start=1):
        lines.append(f"# {i}. {item}")
    lines.append("#")
    lines.append("# 小检查")
    lines.append("# - 先把关键张量 print 出来，确认数值和 shape 符合预期。")
    lines.append("# - 如果报错，优先检查 shape、dtype、device 这三件事。")
    lines.append("# - 如果训练结果不动，优先检查是否 zero_grad、backward、step 都写了。")
    lines.append("#")
    lines.append("# 学习目标")
    lines.append("# - 能用自己的话解释本节核心概念。")
    lines.append("# - 能不看答案写出一个可以运行的最小 PyTorch 程序。")
    lines.append("# - 能通过输出的 shape、loss 或 accuracy 判断代码是否基本正确。")
    lines.append("#")
    lines.append("# 练习任务")
    for item in lesson["tasks"]:
        lines.append(f"# - {item}")
    lines.append("#")
    lines.append("# 做题要求")
    lines.append("# - 尽量先不看 answers 目录。")
    lines.append("# - 代码需要能直接运行。")
    lines.append("# - 打印关键 shape、loss 或 accuracy，确认自己真的跑通了。")
    lines.append("#")
    lines.append("# ===== 练习区：从这里开始写代码 =====")
    lines.append("")
    return "\n".join(lines) + lesson["starter"]


def split_hint_for_table(item):
    code_spans = re.findall(r"`[^`]+`", item)
    if code_spans:
        left = "<br>".join(code_spans[:3])
        if len(code_spans) > 3:
            left += "<br>..."
        return left, item
    return "概念 / 经验", item


def hint_heading(item):
    code_spans = re.findall(r"`[^`]+`", item)
    if code_spans:
        first = code_spans[0].strip("`")
        if first.startswith("["):
            return "形状约定"
        if " / " in first or "/" in first:
            return "公式写法"
        if len(first) > 48:
            return "常用写法"
        return first
    return "概念和经验"


def hint_code_lines(item):
    code_spans = [code.strip("`") for code in re.findall(r"`[^`]+`", item)]
    seen = []
    for code in code_spans:
        if code not in seen:
            seen.append(code)
    return seen


def hint_extra_notes(item):
    notes = []
    lower = item.lower()
    if "shape" in lower or "[" in item:
        notes.append("写这类代码时，第一步先确认张量形状。PyTorch 的很多报错不是公式错了，而是某一维没有对齐。")
    if "dtype" in lower or "float" in lower or "long" in lower:
        notes.append("同时注意 dtype：模型输入通常是浮点张量，分类标签通常是 `torch.long`，二分类 BCE 标签通常是浮点 0/1。")
    if "device" in lower or "cuda" in lower or "cpu" in lower:
        notes.append("如果代码要兼容 CPU/GPU，模型、输入、标签和新建临时张量都要放在同一个 device。")
    if "loss" in lower or "backward" in lower or "optimizer" in lower or "step()" in lower or "opt.step" in lower:
        notes.append("训练时关注顺序：先前向得到输出，再算 loss，然后清空旧梯度、反向传播、更新参数。顺序乱了通常不会得到正确训练。")
    if "linear" in lower or "conv" in lower or "rnn" in lower or "lstm" in lower or "transformer" in lower:
        notes.append("涉及模型层时，把每一层都看成一次 shape 变换；不确定时在 forward 中临时打印中间结果 shape。")
    if "softmax" in lower or "sigmoid" in lower or "logit" in lower or "crossentropy" in lower or "bce" in lower:
        notes.append("分类任务要分清 logits 和概率。大多数 PyTorch loss 直接接收 logits，只有推理或展示结果时才需要 sigmoid/softmax。")
    if "grad" in lower or "requires_grad" in lower or "detach" in lower:
        notes.append("和梯度有关的写法要小心计算图是否被断开。用于打印日志时可以 `.item()`，但参与训练的张量不要过早转成 Python 数字。")
    if "dataset" in lower or "dataloader" in lower or "batch" in lower:
        notes.append("数据管道的重点是第 0 维样本数一致。进入训练循环后，每个 batch 都应该能直接喂给模型。")
    return notes


def tutorial_text(index, lesson):
    hints = API_HINTS.get(lesson["slug"], [])

    lines = []
    lines.append(f"# 第 {index:02d} 课：{lesson['title']}")
    lines.append("")
    lines.append("> 这一页只讲本节知识点和常用写法。练习题在 `lessons` 目录，答案在 `answers` 目录。")
    lines.append("")
    lines.append("## 核心概念")
    lines.append("")
    for item in lesson["principles"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## 关键写法详解")
    lines.append("")
    if hints:
        for i, item in enumerate(hints, start=1):
            lines.append(f"### {i}. {hint_heading(item)}")
            lines.append("")
            code_lines = hint_code_lines(item)
            if code_lines:
                lines.append("```python")
                for code in code_lines:
                    lines.append(code)
                lines.append("```")
                lines.append("")
            lines.append("说明：")
            lines.append("")
            lines.append(item)
            lines.append("")
    else:
        lines.append("- 本节暂无额外 API 清单，先按 lesson 文件里的操作路线完成练习。")
    lines.append("")
    lines.append("## 本节任务")
    lines.append("")
    for item in lesson["tasks"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def answer_text(index, lesson):
    header = f"# 第 {index:02d} 课答案：{lesson['title']}\n# 先独立完成 lesson 文件，再打开这个答案对照。\n\n"
    return header + lesson["answer"].replace("\n+", "\n")


def main():
    LESSONS_DIR.mkdir(parents=True, exist_ok=True)
    ANSWERS_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    lesson_list = []
    for i, lesson in enumerate(LESSONS, start=1):
        lesson_name = f"lesson_{i:02d}_{lesson['slug']}.py"
        answer_name = f"answer_{i:02d}_{lesson['slug']}.py"
        tutorial_name = f"tutorial_{i:02d}_{lesson['slug']}.md"
        (LESSONS_DIR / lesson_name).write_text(lesson_text(i, lesson), encoding="utf-8")
        (ANSWERS_DIR / answer_name).write_text(answer_text(i, lesson), encoding="utf-8")
        (DOCS_DIR / tutorial_name).write_text(tutorial_text(i, lesson), encoding="utf-8")
        lesson_list.append(
            f"{i:02d}. {lesson['title']} - `docs/{tutorial_name}` / `lessons/{lesson_name}` / `answers/{answer_name}`"
        )

    readme = README.format(lesson_list="\n".join(lesson_list))
    (BASE / "README.md").write_text(readme, encoding="utf-8")

    runner = textwrap.dedent(
        """\
        import subprocess
        import sys
        from pathlib import Path


        ROOT = Path(__file__).resolve().parent
        answers = sorted((ROOT / "answers").glob("answer_*.py"))

        for file in answers:
            print(f"RUN {file.name}", flush=True)
            result = subprocess.run([sys.executable, str(file)], cwd=ROOT.parent)
            if result.returncode != 0:
                raise SystemExit(result.returncode)

        print(f"OK: {len(answers)} answer files ran successfully.")
        """
    )
    (BASE / "run_all_answers.py").write_text(runner, encoding="utf-8")


if __name__ == "__main__":
    main()
