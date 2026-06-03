# 第 31 课：梯度裁剪与混合精度入口
#
# 核心讲解
# 这里保留最短版本，详细教程请先读：
# ../docs/tutorial_31_gradient_clipping_amp.md
#
# 1. 梯度裁剪可以限制梯度范数，缓解梯度爆炸，常见于 RNN/Transformer。
# 2. 混合精度能在 GPU 上用更少显存和更快速度训练，CPU 上通常保持普通精度。
# 3. torch.amp 的启用应根据 cuda 是否可用来决定。
#
# 操作路线
# 写本节代码时，可以按这个顺序推进：
# 1. 训练一个小网络。
# 2. 反向传播后使用 clip_grad_norm_。
# 3. 使用 autocast/GradScaler 写出兼容 CPU 的训练步骤。
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
# - 训练一个小网络。
# - 反向传播后使用 clip_grad_norm_。
# - 使用 autocast/GradScaler 写出兼容 CPU 的训练步骤。
#
# 做题要求
# - 尽量先不看 answers 目录。
# - 代码需要能直接运行。
# - 打印关键 shape、loss 或 accuracy，确认自己真的跑通了。
#
# ===== 练习区：从这里开始写代码 =====
def main():
    # TODO: 实现带梯度裁剪和可选 AMP 的训练步骤。
    pass


if __name__ == "__main__":
    main()
