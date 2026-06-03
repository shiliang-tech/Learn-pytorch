# 第 06 课：自动求导 autograd
#
# 核心讲解
# 这里保留最短版本，详细教程请先读：
# ../docs/tutorial_06_autograd_basics.md
#
# 1. requires_grad=True 会让 PyTorch 记录张量参与的计算图。
# 2. loss.backward() 会从标量 loss 反向传播，计算叶子张量的 grad。
# 3. 每次反向传播前通常要清空旧梯度，因为 PyTorch 默认会累加梯度。
#
# 操作路线
# 写本节代码时，可以按这个顺序推进：
# 1. 创建一个可求导的标量 w。
# 2. 令 loss=(w-3)^2，调用 backward。
# 3. 打印 w.grad，并解释为什么梯度等于 2*(w-3)。
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
# - 创建一个可求导的标量 w。
# - 令 loss=(w-3)^2，调用 backward。
# - 打印 w.grad，并解释为什么梯度等于 2*(w-3)。
#
# 做题要求
# - 尽量先不看 answers 目录。
# - 代码需要能直接运行。
# - 打印关键 shape、loss 或 accuracy，确认自己真的跑通了。
#
# ===== 练习区：从这里开始写代码 =====
def main():
    # TODO: 创建 w，计算 loss，然后 backward。
    pass


if __name__ == "__main__":
    main()
