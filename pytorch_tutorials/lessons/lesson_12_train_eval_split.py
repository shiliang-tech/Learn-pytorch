# 第 12 课：训练集、验证集与 eval 模式
#
# 核心讲解
# 这里保留最短版本，详细教程请先读：
# ../docs/tutorial_12_train_eval_split.md
#
# 1. 训练集用来更新参数，验证集用来观察泛化效果。
# 2. model.train() 与 model.eval() 会影响 Dropout、BatchNorm 等层的行为。
# 3. 验证阶段通常配合 torch.no_grad()，避免构建计算图，节省显存和时间。
#
# 操作路线
# 写本节代码时，可以按这个顺序推进：
# 1. 把合成分类数据切成训练集和验证集。
# 2. 训练 MLP，并在每个 epoch 后计算验证准确率。
# 3. 验证时使用 model.eval() 和 torch.no_grad()。
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
# - 把合成分类数据切成训练集和验证集。
# - 训练 MLP，并在每个 epoch 后计算验证准确率。
# - 验证时使用 model.eval() 和 torch.no_grad()。
#
# 做题要求
# - 尽量先不看 answers 目录。
# - 代码需要能直接运行。
# - 打印关键 shape、loss 或 accuracy，确认自己真的跑通了。
#
# ===== 练习区：从这里开始写代码 =====
def main():
    # TODO: 写一个带验证集评估的训练循环。
    pass


if __name__ == "__main__":
    main()
