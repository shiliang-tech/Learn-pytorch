# 第 15 课：保存与加载模型参数
#
# 核心讲解
# 这里保留最短版本，详细教程请先读：
# ../docs/tutorial_15_save_load.md
#
# 1. 推荐保存 model.state_dict()，它只包含参数和 buffer，结构清晰、可迁移。
# 2. 加载时需要先创建相同结构的模型，再 load_state_dict。
# 3. 推理前记得 model.eval()，避免 Dropout 和 BatchNorm 处在训练行为。
#
# 操作路线
# 写本节代码时，可以按这个顺序推进：
# 1. 训练一个小模型。
# 2. 把 state_dict 保存到 pytorch_tutorials/tmp_linear.pt。
# 3. 创建新模型加载参数，验证两者输出一致。
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
# - 训练一个小模型。
# - 把 state_dict 保存到 pytorch_tutorials/tmp_linear.pt。
# - 创建新模型加载参数，验证两者输出一致。
#
# 做题要求
# - 尽量先不看 answers 目录。
# - 代码需要能直接运行。
# - 打印关键 shape、loss 或 accuracy，确认自己真的跑通了。
#
# ===== 练习区：从这里开始写代码 =====
def main():
    # TODO: 训练、保存、加载，并比较输出。
    pass


if __name__ == "__main__":
    main()
