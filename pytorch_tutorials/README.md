# PyTorch 与深度学习渐进教程

这套材料按从浅到深排列。每节课由三部分组成：

- `docs/tutorial_XX_*.md`：详细教程和关键写法，适合先读和复习。
- `lessons/lesson_XX_*.py`：轻量练习文件，保留题目、路线和代码练习区。
- `answers/answer_XX_*.py`：对应答案。

建议学习节奏：

1. 先读对应的 docs 教程。
2. 在 lesson 文件底部把 `TODO` 补成能运行的 PyTorch 代码。
3. 运行自己的 lesson 文件。
4. 再打开对应 answer 文件对照。
5. 能独立复写答案后，再进入下一节。

运行示例：

```powershell
python .\pytorch_tutorials\lessons\lesson_01_tensor_basics.py
python .\pytorch_tutorials\answers\answer_01_tensor_basics.py
```

课程路线：

01. Tensor 入门：创建、运算、矩阵乘法 - `docs/tutorial_01_tensor_basics.md` / `lessons/lesson_01_tensor_basics.py` / `answers/answer_01_tensor_basics.py`
02. dtype、device 与随机种子 - `docs/tutorial_02_dtype_device_seed.md` / `lessons/lesson_02_dtype_device_seed.py` / `answers/answer_02_dtype_device_seed.py`
03. 形状变换与广播机制 - `docs/tutorial_03_reshape_broadcast.md` / `lessons/lesson_03_reshape_broadcast.py` / `answers/answer_03_reshape_broadcast.py`
04. 索引、切片与布尔 mask - `docs/tutorial_04_indexing_masking.md` / `lessons/lesson_04_indexing_masking.py` / `answers/answer_04_indexing_masking.py`
05. 聚合统计、范数与标准化 - `docs/tutorial_05_reductions_norms.md` / `lessons/lesson_05_reductions_norms.py` / `answers/answer_05_reductions_norms.py`
06. 自动求导 autograd - `docs/tutorial_06_autograd_basics.md` / `lessons/lesson_06_autograd_basics.py` / `answers/answer_06_autograd_basics.py`
07. 手写梯度下降：拟合 y=2x+1 - `docs/tutorial_07_manual_gradient_descent.md` / `lessons/lesson_07_manual_gradient_descent.py` / `answers/answer_07_manual_gradient_descent.py`
08. nn.Module 版线性回归 - `docs/tutorial_08_nn_linear_regression.md` / `lessons/lesson_08_nn_linear_regression.py` / `answers/answer_08_nn_linear_regression.py`
09. TensorDataset 与 DataLoader - `docs/tutorial_09_dataloader_basics.md` / `lessons/lesson_09_dataloader_basics.py` / `answers/answer_09_dataloader_basics.py`
10. 二分类：logits、Sigmoid 与 BCEWithLogitsLoss - `docs/tutorial_10_binary_classification.md` / `lessons/lesson_10_binary_classification.py` / `answers/answer_10_binary_classification.py`
11. 多分类 MLP 与 CrossEntropyLoss - `docs/tutorial_11_multiclass_mlp.md` / `lessons/lesson_11_multiclass_mlp.py` / `answers/answer_11_multiclass_mlp.py`
12. 训练集、验证集与 eval 模式 - `docs/tutorial_12_train_eval_split.md` / `lessons/lesson_12_train_eval_split.py` / `answers/answer_12_train_eval_split.py`
13. 初始化与激活函数 - `docs/tutorial_13_initialization_activation.md` / `lessons/lesson_13_initialization_activation.py` / `answers/answer_13_initialization_activation.py`
14. 正则化：weight decay 与 Dropout - `docs/tutorial_14_regularization.md` / `lessons/lesson_14_regularization.py` / `answers/answer_14_regularization.py`
15. 保存与加载模型参数 - `docs/tutorial_15_save_load.md` / `lessons/lesson_15_save_load.py` / `answers/answer_15_save_load.py`
16. 设备无关训练：CPU/GPU 通用代码 - `docs/tutorial_16_device_agnostic_training.md` / `lessons/lesson_16_device_agnostic_training.py` / `answers/answer_16_device_agnostic_training.py`
17. 自定义 Dataset - `docs/tutorial_17_custom_dataset.md` / `lessons/lesson_17_custom_dataset.py` / `answers/answer_17_custom_dataset.py`
18. 图像张量与 Conv2d 形状 - `docs/tutorial_18_conv2d_shapes.md` / `lessons/lesson_18_conv2d_shapes.py` / `answers/answer_18_conv2d_shapes.py`
19. CNN 小项目：识别条纹方向 - `docs/tutorial_19_cnn_synthetic_images.md` / `lessons/lesson_19_cnn_synthetic_images.py` / `answers/answer_19_cnn_synthetic_images.py`
20. BatchNorm 与 Dropout 的 train/eval 差异 - `docs/tutorial_20_batchnorm_dropout.md` / `lessons/lesson_20_batchnorm_dropout.py` / `answers/answer_20_batchnorm_dropout.py`
21. 学习率调度与早停思路 - `docs/tutorial_21_scheduler_early_stopping.md` / `lessons/lesson_21_scheduler_early_stopping.py` / `answers/answer_21_scheduler_early_stopping.py`
22. 指标：准确率、精确率、召回率、混淆矩阵 - `docs/tutorial_22_metrics_confusion_matrix.md` / `lessons/lesson_22_metrics_confusion_matrix.py` / `answers/answer_22_metrics_confusion_matrix.py`
23. Embedding：把离散 id 变成向量 - `docs/tutorial_23_embeddings_text.md` / `lessons/lesson_23_embeddings_text.py` / `answers/answer_23_embeddings_text.py`
24. RNN：序列分类入门 - `docs/tutorial_24_rnn_sequence.md` / `lessons/lesson_24_rnn_sequence.py` / `answers/answer_24_rnn_sequence.py`
25. LSTM：预测正弦序列下一步 - `docs/tutorial_25_lstm_forecasting.md` / `lessons/lesson_25_lstm_forecasting.py` / `answers/answer_25_lstm_forecasting.py`
26. Attention：Query、Key、Value - `docs/tutorial_26_attention_basics.md` / `lessons/lesson_26_attention_basics.py` / `answers/answer_26_attention_basics.py`
27. TransformerEncoder：序列建模小例子 - `docs/tutorial_27_transformer_encoder.md` / `lessons/lesson_27_transformer_encoder.py` / `answers/answer_27_transformer_encoder.py`
28. AutoEncoder：压缩与重建 - `docs/tutorial_28_autoencoder.md` / `lessons/lesson_28_autoencoder.py` / `answers/answer_28_autoencoder.py`
29. VAE：均值、方差与重参数化 - `docs/tutorial_29_vae_reparameterization.md` / `lessons/lesson_29_vae_reparameterization.py` / `answers/answer_29_vae_reparameterization.py`
30. GAN：一维分布生成 - `docs/tutorial_30_gan_1d.md` / `lessons/lesson_30_gan_1d.py` / `answers/answer_30_gan_1d.py`
31. 梯度裁剪与混合精度入口 - `docs/tutorial_31_gradient_clipping_amp.md` / `lessons/lesson_31_gradient_clipping_amp.py` / `answers/answer_31_gradient_clipping_amp.py`
32. 端到端小项目：从数据到保存模型 - `docs/tutorial_32_end_to_end_project.md` / `lessons/lesson_32_end_to_end_project.py` / `answers/answer_32_end_to_end_project.py`
