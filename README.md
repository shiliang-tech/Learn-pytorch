# PyTorch 深度学习自学教程

这是一个面向 PyTorch 和深度学习入门的自学仓库。课程按从浅到深排列，每一节都包含详细 Markdown 教程、练习 Python 文件和对应答案文件。

## 仓库结构

```text
.
├── pytorch_tutorials/
│   ├── docs/       # 每节课的详细教程和关键写法速查
│   ├── lessons/    # 练习题文件，在 TODO 处自己补代码
│   ├── answers/    # 对应答案文件
│   ├── README.md   # 课程内部说明和完整课程路线
│   └── run_all_answers.py
├── scripts/
│   └── generate_pytorch_lessons.py
└── .gitignore
```

## 学习方式

推荐每节课按这个顺序学习：

1. 先读 `pytorch_tutorials/docs/tutorial_XX_*.md`。
2. 再打开 `pytorch_tutorials/lessons/lesson_XX_*.py`，补全 TODO。
3. 运行自己的练习文件，观察输出、shape、loss 或 accuracy。
4. 最后对照 `pytorch_tutorials/answers/answer_XX_*.py`。
5. 能不看答案复写一遍，再进入下一课。

示例：

```powershell
python .\pytorch_tutorials\lessons\lesson_01_tensor_basics.py
python .\pytorch_tutorials\answers\answer_01_tensor_basics.py
```

## 课程内容

课程目前包含 32 节，覆盖：

- Tensor 基础、dtype、device、shape、索引、广播
- autograd、手写梯度下降、`nn.Module`、优化器
- DataLoader、训练集/验证集、保存和加载模型
- 二分类、多分类、MLP、正则化、指标
- CNN、BatchNorm、Dropout、学习率调度
- Embedding、RNN、LSTM、Attention、Transformer
- AutoEncoder、VAE、GAN
- 梯度裁剪、混合精度、端到端小项目

完整课程路线见：

[pytorch_tutorials/README.md](pytorch_tutorials/README.md)

## 批量验证答案

如果已经安装 PyTorch，可以运行所有答案文件：

```powershell
python .\pytorch_tutorials\run_all_answers.py
```

## 说明

`pytorch_tutorials/lessons_sl/` 是个人练习副本目录，已在 `.gitignore` 中排除。
