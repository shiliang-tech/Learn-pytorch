# 第 02 课：dtype、device 与随机种子
#
# 核心讲解
# 这里保留最短版本，详细教程请先读：
# ../docs/tutorial_02_dtype_device_seed.md
#
# 1. dtype 决定数值类型，常见训练默认用 torch.float32；分类标签常用 torch.long。
# 2. device 决定张量在 CPU 还是 GPU。模型和输入必须在同一个 device 上。
# 3. 随机种子让实验更容易复现，尤其是初始化、随机数据、DataLoader shuffle。
#
# 操作路线
# 写本节代码时，可以按这个顺序推进：
# 1. 设置 torch.manual_seed(42)。
# 2. 创建一个随机张量，把它转换成 float64，再转换回 float32。
# 3. 选择 cuda 或 cpu 作为 device，并把张量移动过去。
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
# - 设置 torch.manual_seed(42)。
# - 创建一个随机张量，把它转换成 float64，再转换回 float32。
# - 选择 cuda 或 cpu 作为 device，并把张量移动过去。
#
# 做题要求
# - 尽量先不看 answers 目录。
# - 代码需要能直接运行。
# - 打印关键 shape、loss 或 accuracy，确认自己真的跑通了。
#
# ===== 练习区：从这里开始写代码 =====
def main():
    # TODO: 设置随机种子，练习 dtype 与 device 转换。
    pass


if __name__ == "__main__":
    main()
