# 第 27 课：TransformerEncoder：序列建模小例子
#
# 核心讲解
# 这里保留最短版本，详细教程请先读：
# ../docs/tutorial_27_transformer_encoder.md
#
# 1. TransformerEncoder 用自注意力让序列中每个位置都能看见其他位置。
# 2. 它通常需要 embedding、位置编码或某种位置信息。
# 3. nn.TransformerEncoderLayer 可以快速搭出标准 encoder block。
#
# 操作路线
# 写本节代码时，可以按这个顺序推进：
# 1. 构造 token 序列，标签为第一个 token 是否大于最后一个 token。
# 2. 使用 Embedding + TransformerEncoder。
# 3. 取序列表示平均后分类。
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
# - 构造 token 序列，标签为第一个 token 是否大于最后一个 token。
# - 使用 Embedding + TransformerEncoder。
# - 取序列表示平均后分类。
#
# 做题要求
# - 尽量先不看 answers 目录。
# - 代码需要能直接运行。
# - 打印关键 shape、loss 或 accuracy，确认自己真的跑通了。
#
# ===== 练习区：从这里开始写代码 =====
def main():
    # TODO: 用 TransformerEncoder 完成 toy 序列分类。
    pass


if __name__ == "__main__":
    main()
