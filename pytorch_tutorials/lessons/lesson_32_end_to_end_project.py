# 第 32 课：端到端小项目：从数据到保存模型
#
# 核心讲解
# 这里保留最短版本，详细教程请先读：
# ../docs/tutorial_32_end_to_end_project.md
#
# 1. 完整项目通常包含：数据准备、模型、训练循环、验证指标、保存最好模型。
# 2. 把 train_one_epoch 和 evaluate 拆成函数，会让代码更容易维护。
# 3. 最终你应该能独立搭起一个小型监督学习项目骨架。
#
# 操作路线
# 写本节代码时，可以按这个顺序推进：
# 1. 创建一个非线性二分类数据集。
# 2. 写 train_one_epoch 与 evaluate。
# 3. 保存验证准确率最高的模型参数。
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
# - 创建一个非线性二分类数据集。
# - 写 train_one_epoch 与 evaluate。
# - 保存验证准确率最高的模型参数。
#
# 做题要求
# - 尽量先不看 answers 目录。
# - 代码需要能直接运行。
# - 打印关键 shape、loss 或 accuracy，确认自己真的跑通了。
#
# ===== 练习区：从这里开始写代码 =====
def train_one_epoch(model, x, y, optimizer, loss_fn):
    # TODO: 完成一个训练 epoch。
    pass


def evaluate(model, x, y, loss_fn):
    # TODO: 返回 loss 和 accuracy。
    pass


def main():
    # TODO: 串起数据、模型、训练、验证、保存。
    pass


if __name__ == "__main__":
    main()
