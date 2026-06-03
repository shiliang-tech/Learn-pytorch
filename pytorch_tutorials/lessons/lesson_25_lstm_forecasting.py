# 第 25 课：LSTM：预测正弦序列下一步
#
# 核心讲解
# 这里保留最短版本，详细教程请先读：
# ../docs/tutorial_25_lstm_forecasting.md
#
# 1. LSTM 用门控结构缓解普通 RNN 的长程依赖困难。
# 2. 时间序列预测常用过去若干步作为输入，预测下一步或未来多步。
# 3. 回归任务输出连续值，常用 MSELoss。
#
# 操作路线
# 写本节代码时，可以按这个顺序推进：
# 1. 生成 sin 曲线滑动窗口数据。
# 2. 输入过去 10 步，预测第 11 步。
# 3. 用 nn.LSTM 和 Linear 完成回归。
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
# - 生成 sin 曲线滑动窗口数据。
# - 输入过去 10 步，预测第 11 步。
# - 用 nn.LSTM 和 Linear 完成回归。
#
# 做题要求
# - 尽量先不看 answers 目录。
# - 代码需要能直接运行。
# - 打印关键 shape、loss 或 accuracy，确认自己真的跑通了。
#
# ===== 练习区：从这里开始写代码 =====
def main():
    # TODO: 构造滑动窗口数据，用 LSTM 预测下一步。
    pass


if __name__ == "__main__":
    main()
