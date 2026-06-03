# 第 09 课：TensorDataset 与 DataLoader
#
# 核心讲解
# 这里保留最短版本，详细教程请先读：
# ../docs/tutorial_09_dataloader_basics.md
#
# 1. Dataset 定义如何取一个样本，DataLoader 负责批量、打乱、多进程读取。
# 2. batch 训练比一次喂全部数据更常见，也更接近真实项目。
# 3. shuffle=True 常用于训练集，验证集和测试集通常不需要 shuffle。
#
# 操作路线
# 写本节代码时，可以按这个顺序推进：
# 1. 用 TensorDataset 包装 x 和 y。
# 2. 用 DataLoader 每次取 16 条样本。
# 3. 写一个 mini-batch 训练循环拟合线性回归。
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
# - 用 TensorDataset 包装 x 和 y。
# - 用 DataLoader 每次取 16 条样本。
# - 写一个 mini-batch 训练循环拟合线性回归。
#
# 做题要求
# - 尽量先不看 answers 目录。
# - 代码需要能直接运行。
# - 打印关键 shape、loss 或 accuracy，确认自己真的跑通了。
#
# ===== 练习区：从这里开始写代码 =====
def main():
    # TODO: 使用 TensorDataset 和 DataLoader 训练模型。
    pass


if __name__ == "__main__":
    main()
