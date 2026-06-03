# 第 22 课答案：指标：准确率、精确率、召回率、混淆矩阵
# 先独立完成 lesson 文件，再打开这个答案对照。

import torch


def main():
    y_true = torch.tensor([1, 0, 1, 1, 0, 0, 1, 0])
    y_pred = torch.tensor([1, 0, 0, 1, 0, 1, 1, 0])
    tp = ((y_true == 1) & (y_pred == 1)).sum().item()
    fp = ((y_true == 0) & (y_pred == 1)).sum().item()
    fn = ((y_true == 1) & (y_pred == 0)).sum().item()
    tn = ((y_true == 0) & (y_pred == 0)).sum().item()
    accuracy = (tp + tn) / len(y_true)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    print({"tp": tp, "fp": fp, "fn": fn, "tn": tn})
    print("accuracy:", accuracy, "precision:", precision, "recall:", recall)


if __name__ == "__main__":
    main()
