# 第 18 课：图像张量与 Conv2d 形状
#
# 核心讲解
# 这里保留最短版本，详细教程请先读：
# ../docs/tutorial_18_conv2d_shapes.md
#
# 1. PyTorch 图像张量常用 NCHW：batch、channel、height、width。
# 2. Conv2d 会用卷积核在空间维度滑动，输出通道数由 out_channels 决定。
# 3. padding、stride、kernel_size 会共同决定输出高宽。
#
# 操作路线
# 写本节代码时，可以按这个顺序推进：
# 1. 创建形状 [8,1,28,28] 的随机图片 batch。
# 2. 用 Conv2d(1,4,kernel_size=3,padding=1) 处理。
# 3. 接 MaxPool2d(2)，打印每一步 shape。
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
# - 创建形状 [8,1,28,28] 的随机图片 batch。
# - 用 Conv2d(1,4,kernel_size=3,padding=1) 处理。
# - 接 MaxPool2d(2)，打印每一步 shape。
#
# 做题要求
# - 尽量先不看 answers 目录。
# - 代码需要能直接运行。
# - 打印关键 shape、loss 或 accuracy，确认自己真的跑通了。
#
# ===== 练习区：从这里开始写代码 =====
def main():
    # TODO: 创建图像 batch，经过卷积和池化，打印 shape。
    pass


if __name__ == "__main__":
    main()
