# 第 13 课：初始化与激活函数
#
# 核心讲解
# 这里保留最短版本，详细教程请先读：
# ../docs/tutorial_13_initialization_activation.md
#
# 1. 激活函数引入非线性，否则多层 Linear 仍等价于一个 Linear。
# 2. ReLU 常配合 Kaiming 初始化，Tanh/Sigmoid 更常配合 Xavier 初始化。
# 3. 初始化会影响早期梯度大小，过大或过小都可能让训练不稳定。
#
# 操作路线
# 写本节代码时，可以按这个顺序推进：
# 1. 定义一个两层 MLP 类。
# 2. 对 Linear 层使用 kaiming_normal_ 初始化。
# 3. 训练一个非线性二分类任务。
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
# - 定义一个两层 MLP 类。
# - 对 Linear 层使用 kaiming_normal_ 初始化。
# - 训练一个非线性二分类任务。
#
# 做题要求
# - 尽量先不看 answers 目录。
# - 代码需要能直接运行。
# - 打印关键 shape、loss 或 accuracy，确认自己真的跑通了。
#
# ===== 练习区：从这里开始写代码 =====
class MLP:
    # TODO: 继承 nn.Module 并实现 forward。
    pass


def main():
    # TODO: 初始化模型并训练。
    pass


if __name__ == "__main__":
    main()
