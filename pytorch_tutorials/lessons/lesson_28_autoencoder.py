# 第 28 课：AutoEncoder：压缩与重建
#
# 核心讲解
# 这里保留最短版本，详细教程请先读：
# ../docs/tutorial_28_autoencoder.md
#
# 1. 自编码器由 encoder 和 decoder 组成，目标是重建输入。
# 2. 瓶颈层维度较小时，模型被迫学习压缩表示。
# 3. 重建任务通常使用 MSELoss 或 BCE 类损失，取决于数据范围。
#
# 操作路线
# 写本节代码时，可以按这个顺序推进：
# 1. 生成二维圆环数据。
# 2. 用 encoder 把 2 维压到 1 维，再 decoder 重建到 2 维。
# 3. 训练并打印重建误差。
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
# - 生成二维圆环数据。
# - 用 encoder 把 2 维压到 1 维，再 decoder 重建到 2 维。
# - 训练并打印重建误差。
#
# 做题要求
# - 尽量先不看 answers 目录。
# - 代码需要能直接运行。
# - 打印关键 shape、loss 或 accuracy，确认自己真的跑通了。
#
# ===== 练习区：从这里开始写代码 =====
def main():
    # TODO: 实现并训练一个小 AutoEncoder。
    pass


if __name__ == "__main__":
    main()
