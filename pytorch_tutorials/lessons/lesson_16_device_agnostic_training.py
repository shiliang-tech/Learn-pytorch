# 第 16 课：设备无关训练：CPU/GPU 通用代码
#
# 核心讲解
# 这里保留最短版本，详细教程请先读：
# ../docs/tutorial_16_device_agnostic_training.md
#
# 1. 设备无关代码会先选择 device，然后把模型和每个 batch 都移动到这个 device。
# 2. 不要只移动模型或只移动数据，否则会出现 device mismatch 错误。
# 3. 保存模型时通常仍保存 state_dict；加载到 CPU 可使用 map_location='cpu'。
#
# 操作路线
# 写本节代码时，可以按这个顺序推进：
# 1. 选择 cuda 或 cpu。
# 2. 把模型、输入和标签都移动到 device。
# 3. 训练一个小分类器并打印当前 device。
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
# - 选择 cuda 或 cpu。
# - 把模型、输入和标签都移动到 device。
# - 训练一个小分类器并打印当前 device。
#
# 做题要求
# - 尽量先不看 answers 目录。
# - 代码需要能直接运行。
# - 打印关键 shape、loss 或 accuracy，确认自己真的跑通了。
#
# ===== 练习区：从这里开始写代码 =====
def main():
    # TODO: 写出 CPU/GPU 都能运行的训练代码。
    pass


if __name__ == "__main__":
    main()
