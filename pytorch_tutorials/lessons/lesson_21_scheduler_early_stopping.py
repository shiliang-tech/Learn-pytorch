# 第 21 课：学习率调度与早停思路
#
# 核心讲解
# 这里保留最短版本，详细教程请先读：
# ../docs/tutorial_21_scheduler_early_stopping.md
#
# 1. 学习率太大可能震荡，太小可能收敛很慢。
# 2. scheduler 可以在训练过程中调整学习率，例如 StepLR 定期衰减。
# 3. 早停会在验证指标长期不提升时停止训练，避免浪费时间和过拟合。
#
# 操作路线
# 写本节代码时，可以按这个顺序推进：
# 1. 使用 StepLR 每 20 个 epoch 把学习率乘 0.5。
# 2. 记录最好的验证 loss。
# 3. 如果验证 loss 连续若干轮没有提升就提前停止。
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
# - 使用 StepLR 每 20 个 epoch 把学习率乘 0.5。
# - 记录最好的验证 loss。
# - 如果验证 loss 连续若干轮没有提升就提前停止。
#
# 做题要求
# - 尽量先不看 answers 目录。
# - 代码需要能直接运行。
# - 打印关键 shape、loss 或 accuracy，确认自己真的跑通了。
#
# ===== 练习区：从这里开始写代码 =====
def main():
    # TODO: 写一个包含 scheduler 和 early stopping 的训练循环。
    pass


if __name__ == "__main__":
    main()
