# Surgival Fine-tuning improves adaptation to distribution shifts
---
## CIFAR-10 数据集， ResNet-26结构 input level的修改

**corrupted_test_data.py:** 划分测试数据集为偏移版本和原始版本  
**train.py:**    利用ResNet-26对CIFAR-10整体数据进行训练，使用全部test data进行结果测试，保存结果权重文件和训练过程loss变化

### 数据处理收藏
**test_raw_batch.pkl:** 50%特征偏移的测试集合，用作微调阶段的训练集合
**test_c_batch.pkl:** 50%特征偏移的测试集合，用作微调阶段的测试集合
