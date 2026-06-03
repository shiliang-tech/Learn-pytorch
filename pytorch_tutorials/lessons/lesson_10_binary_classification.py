# 第 10 课：二分类：logits、Sigmoid 与 BCEWithLogitsLoss
#
# 核心讲解
# 这里保留最短版本，详细教程请先读：
# ../docs/tutorial_10_binary_classification.md
#
# 1. 二分类模型常输出一个 logit，logit 经过 sigmoid 后变成属于正类的概率。
# 2. BCEWithLogitsLoss 内部包含 sigmoid，比手动 sigmoid 后再 BCE 更数值稳定。
# 3. 预测时一般用 sigmoid(logit)>0.5 得到类别。
#
# 操作路线
# 写本节代码时，可以按这个顺序推进：
# 1. 生成二维点，标签为 x1+x2>0。
# 2. 训练 nn.Linear(2,1) 二分类器。
# 3. 计算训练准确率。
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
# - 生成二维点，标签为 x1+x2>0。
# - 训练 nn.Linear(2,1) 二分类器。
# - 计算训练准确率。
#
# 做题要求
# - 尽量先不看 answers 目录。
# - 代码需要能直接运行。
# - 打印关键 shape、loss 或 accuracy，确认自己真的跑通了。
#
# ===== 练习区：从这里开始写代码 =====
def main():
    # TODO: 训练一个二分类线性模型，并打印 accuracy。
    pass


if __name__ == "__main__":
    main()
