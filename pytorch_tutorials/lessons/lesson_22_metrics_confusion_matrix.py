# 第 22 课：指标：准确率、精确率、召回率、混淆矩阵
#
# 核心讲解
# 这里保留最短版本，详细教程请先读：
# ../docs/tutorial_22_metrics_confusion_matrix.md
#
# 1. accuracy 适合类别均衡场景，但类别不均衡时可能误导。
# 2. precision 关注预测为正的样本中有多少是真的正类。
# 3. recall 关注真实正类中有多少被找出来，混淆矩阵能展示错误类型。
#
# 操作路线
# 写本节代码时，可以按这个顺序推进：
# 1. 给定 y_true 和 y_pred。
# 2. 计算二分类混淆矩阵 TP/FP/FN/TN。
# 3. 计算 accuracy、precision、recall。
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
# - 给定 y_true 和 y_pred。
# - 计算二分类混淆矩阵 TP/FP/FN/TN。
# - 计算 accuracy、precision、recall。
#
# 做题要求
# - 尽量先不看 answers 目录。
# - 代码需要能直接运行。
# - 打印关键 shape、loss 或 accuracy，确认自己真的跑通了。
#
# ===== 练习区：从这里开始写代码 =====
def main():
    # TODO: 手写二分类指标计算。
    pass


if __name__ == "__main__":
    main()
