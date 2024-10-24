# Surgival Fine-tuning improves adaptation to distribution shifts
---
## CIFAR-10 数据集， ResNet-26结构 input level的修改
**ResNet.py:** 确定ResNet26的基本结构  
**corrupted_test_data.py:** 划分测试数据集为偏移版本和原始版本  
**train.py:** 利用ResNet-26对CIFAR-10整体数据进行训练，使用全部test data进行结果测试，保存结果权重文件和训练过程loss变化
**test.py:** 测试训练结果

### 数据处理
**blur_image.py:** 包括14种不同的方法来模糊图像，这可以提供imput_level分布偏移  
**test_raw_batch.pkl:** 50%特征偏移的测试集合，用作微调阶段的训练集合
**test_c_batch.pkl:** 50%特征偏移的测试集合，用作微调阶段的测试集合
**output_level_data_shift.ipynb：** 提供output_level分布偏移，并保存数据集
**corrupted_test_data.py：** 模糊图像，并且保存数据集

---
# 使用方式
训练例子  
```
python train.py --train_data autodl-tmp/cifar-10-batches-py/test_train_batch.pkl --test_data autodl-tmp/cifar-10-batches-py/test_test_batch.pkl --weights checkpoints/resnet26.pth --save_path checkpoints/ --weights_name full_fine_tune --epochs 15 --lr 1e-3
```
测试例子  
```
python test.py --data autodl-tmp/cifar-10-batches-py/test_test_batch.pkl --weights checkpoints/inputlevel_b1_lr_0.100000.pth
```
