# 第 29 课：VAE：均值、方差与重参数化
#
# 核心讲解
# 这里保留最短版本，详细教程请先读：
# ../docs/tutorial_29_vae_reparameterization.md
#
# 1. VAE 的 encoder 输出潜变量分布参数 mu 和 logvar，而不是单个确定向量。
# 2. 重参数化 z=mu+std*eps 让采样过程仍能反向传播到 encoder。
# 3. VAE loss 通常由重建误差和 KL 散度组成。
#
# 操作路线
# 写本节代码时，可以按这个顺序推进：
# 1. 实现 reparameterize(mu, logvar)。
# 2. 写一个极小 VAE 处理二维数据。
# 3. 训练时同时计算 recon loss 和 KL loss。
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
# - 实现 reparameterize(mu, logvar)。
# - 写一个极小 VAE 处理二维数据。
# - 训练时同时计算 recon loss 和 KL loss。
#
# 做题要求
# - 尽量先不看 answers 目录。
# - 代码需要能直接运行。
# - 打印关键 shape、loss 或 accuracy，确认自己真的跑通了。
#
# ===== 练习区：从这里开始写代码 =====
def reparameterize(mu, logvar):
    # TODO: 实现重参数化采样。
    pass


def main():
    # TODO: 训练一个 tiny VAE。
    pass


if __name__ == "__main__":
    main()
