# 第 11 课：多分类 MLP 与 CrossEntropyLoss
#
# 核心讲解
# 这里保留最短版本，详细教程请先读：
# ../docs/tutorial_11_multiclass_mlp.md
#
# 1. 多分类模型输出每一类的 logit，shape 通常是 [batch, num_classes]。
# 2. CrossEntropyLoss 内部包含 log_softmax，标签必须是类别索引 long 类型。
# 3. MLP 通过 Linear + 非线性激活堆叠，能学习比线性模型更复杂的边界。
#
# 操作路线
# 写本节代码时，可以按这个顺序推进：
# 1. 生成 3 类二维点。
# 2. 搭建 Linear-ReLU-Linear 的 MLP。
# 3. 使用 CrossEntropyLoss 训练并打印准确率。
#
# 小检查
# - 先把关键张量 print 出来，确认数值和 shape 符合预期。
# - 如果报错，优先检查 shape、dtype、device 这三件事。
# - 如果训练结果不动，优先检查是否 zero_grad、backward、step 都写了。
#
# 学习目标
# - 能用自己的话解释本节核心概念。
# - 能不看答案写出一个可以运行的最小 PyTorch 程序。
# - 能通过输出的 shape、loss 或 accuracy 判断代码是否基本正确。
#
# 练习任务
# - 生成 3 类二维点。
# - 搭建 Linear-ReLU-Linear 的 MLP。
# - 使用 CrossEntropyLoss 训练并打印准确率。
#
# 做题要求
# - 尽量先不看 answers 目录。
# - 代码需要能直接运行。
# - 打印关键 shape、loss 或 accuracy，确认自己真的跑通了。
#
# ===== 练习区：从这里开始写代码 =====
def main():
    # TODO: 完成三分类 MLP。
    pass


if __name__ == "__main__":
    main()
