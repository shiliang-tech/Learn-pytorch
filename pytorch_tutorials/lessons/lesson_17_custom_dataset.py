# 第 17 课：自定义 Dataset
#
# 核心讲解
# 这里保留最短版本，详细教程请先读：
# ../docs/tutorial_17_custom_dataset.md
#
# 1. 自定义 Dataset 至少实现 __len__ 和 __getitem__。
# 2. __getitem__ 返回一个样本，可以是张量、标签、字典等结构。
# 3. 把数据生成或预处理封装到 Dataset 后，训练循环会更稳定清晰。
#
# 操作路线
# 写本节代码时，可以按这个顺序推进：
# 1. 写一个 SineDataset，输入 x，标签 y=sin(x)。
# 2. 用 DataLoader 批量读取。
# 3. 训练一个小 MLP 做回归。
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
# - 写一个 SineDataset，输入 x，标签 y=sin(x)。
# - 用 DataLoader 批量读取。
# - 训练一个小 MLP 做回归。
#
# 做题要求
# - 尽量先不看 answers 目录。
# - 代码需要能直接运行。
# - 打印关键 shape、loss 或 accuracy，确认自己真的跑通了。
#
# ===== 练习区：从这里开始写代码 =====
class SineDataset:
    # TODO: 继承 Dataset，实现 __len__ 和 __getitem__。
    pass


def main():
    # TODO: 使用 DataLoader 训练回归模型。
    pass


if __name__ == "__main__":
    main()
