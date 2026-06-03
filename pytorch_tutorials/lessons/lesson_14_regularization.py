# 第 14 课：正则化：weight decay 与 Dropout
#
# 核心讲解
# 这里保留最短版本，详细教程请先读：
# ../docs/tutorial_14_regularization.md
#
# 1. 正则化的目标是降低过拟合，让模型不要只记住训练数据。
# 2. weight_decay 相当于惩罚过大的权重，常直接传给 optimizer。
# 3. Dropout 在训练时随机丢弃部分激活，eval 时自动关闭。
#
# 操作路线
# 写本节代码时，可以按这个顺序推进：
# 1. 构建包含 Dropout 的 MLP。
# 2. 给 Adam 设置 weight_decay。
# 3. 比较 train/eval 模式下同一个输入的输出是否稳定。
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
# - 构建包含 Dropout 的 MLP。
# - 给 Adam 设置 weight_decay。
# - 比较 train/eval 模式下同一个输入的输出是否稳定。
#
# 做题要求
# - 尽量先不看 answers 目录。
# - 代码需要能直接运行。
# - 打印关键 shape、loss 或 accuracy，确认自己真的跑通了。
#
# ===== 练习区：从这里开始写代码 =====
def main():
    # TODO: 构建带 Dropout 和 weight_decay 的训练示例。
    pass


if __name__ == "__main__":
    main()
