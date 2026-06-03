# 第 23 课：Embedding：把离散 id 变成向量
#
# 核心讲解
# 这里保留最短版本，详细教程请先读：
# ../docs/tutorial_23_embeddings_text.md
#
# 1. Embedding 本质是一个可训练查表矩阵，把 token id 映射成稠密向量。
# 2. 文本、类别 id、用户 id、商品 id 都常用 embedding 表示。
# 3. 简单文本分类可以先对词向量求平均，再接线性分类层。
#
# 操作路线
# 写本节代码时，可以按这个顺序推进：
# 1. 构造整数 token 序列，标签由某个关键词是否出现决定。
# 2. 使用 nn.Embedding 和 mean pooling。
# 3. 训练一个二分类模型。
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
# - 构造整数 token 序列，标签由某个关键词是否出现决定。
# - 使用 nn.Embedding 和 mean pooling。
# - 训练一个二分类模型。
#
# 做题要求
# - 尽量先不看 answers 目录。
# - 代码需要能直接运行。
# - 打印关键 shape、loss 或 accuracy，确认自己真的跑通了。
#
# ===== 练习区：从这里开始写代码 =====
def main():
    # TODO: 用 Embedding + 平均池化完成 toy 文本分类。
    pass


if __name__ == "__main__":
    main()
