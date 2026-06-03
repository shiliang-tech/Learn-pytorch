# 第 20 课：BatchNorm 与 Dropout 的 train/eval 差异
#
# 核心讲解
# 这里保留最短版本，详细教程请先读：
# ../docs/tutorial_20_batchnorm_dropout.md
#
# 1. BatchNorm 在训练时使用当前 batch 统计，并更新 running_mean/running_var。
# 2. eval 模式下 BatchNorm 使用累计统计，Dropout 则关闭随机丢弃。
# 3. 这就是为什么验证和推理阶段一定要调用 model.eval()。
#
# 操作路线
# 写本节代码时，可以按这个顺序推进：
# 1. 构建 Linear-BatchNorm-ReLU-Dropout-Linear 网络。
# 2. 喂入同一个 batch，比较 train 和 eval 下输出差异。
# 3. 打印 BatchNorm 的 running_mean。
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
# - 构建 Linear-BatchNorm-ReLU-Dropout-Linear 网络。
# - 喂入同一个 batch，比较 train 和 eval 下输出差异。
# - 打印 BatchNorm 的 running_mean。
#
# 做题要求
# - 尽量先不看 answers 目录。
# - 代码需要能直接运行。
# - 打印关键 shape、loss 或 accuracy，确认自己真的跑通了。
#
# ===== 练习区：从这里开始写代码 =====
def main():
    # TODO: 演示 BatchNorm/Dropout 在 train 和 eval 下的差异。
    pass


if __name__ == "__main__":
    main()
