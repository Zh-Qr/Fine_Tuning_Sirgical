import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
from blur_image import apply_gaus_blur,apply_frost,apply_gaussian_noise,apply_glass_blur,apply_impulse_noise,apply_jpeg_compression,apply_motion_blur,apply_pixelate,apply_saturate,apply_shot_noise,apply_snow,apply_spatter,apply_speckle_noise,apply_zoom_blur

# 读取文件
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

#保存文件
def save_data(file_name, data):
    """保存数据到 pickle 文件"""
    with open(file_name, 'wb') as f:
        pickle.dump(data, f)



def main():
    file_name = "autodl-tmp/cifar-10-batches-py/"
    # 读取原始测试数据
    test_batch = unpickle(file_name+'test_batch')

    # 拆分数据和标签
    test_data = test_batch[b'data']
    test_labels = test_batch[b'labels']

    # 转换形状为 (10000, 32, 32, 3)
    test_data = test_data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

    # 计算原始数据和变换数据的数量
    total_samples = len(test_data)
    num_raw_samples = int(total_samples * 0.1)  # 10% 原始数据,1000个用于微调
    num_c_samples = total_samples - num_raw_samples  # 90% 变换数据，9000个

    # 随机选择原始数据的索引
    np.random.seed(42)  # 设置随机种子以确保可重复性
    raw_indices = np.random.choice(total_samples, num_raw_samples, replace=False)

    # 保存原始数据
    test_raw_data = []
    test_raw_labels = []

    # 保存变换数据
    test_c_data = []
    test_c_labels = []

    for i in range(total_samples):
        if i not in raw_indices:
            # 对剩余的样本应用模糊和偏移
            transformed_img = apply_gaus_blur(test_data[i], blur_ksize=5)
            test_c_data.append(transformed_img)
            test_c_labels.append(test_labels[i])
        else:
            transformed_img = apply_gaus_blur(test_data[i], blur_ksize=5)
            test_raw_data.append(transformed_img)
            test_raw_labels.append(test_labels[i])

    # 转换回 CIFAR-10 数据格式 (N, 3, 32, 32)
    test_raw_data = np.array(test_raw_data).transpose(0, 3, 1, 2).reshape(-1, 3072)
    test_c_data = np.array(test_c_data).transpose(0, 3, 1, 2).reshape(-1, 3072)

    # 创建新的数据批次字典
    test_raw_batch = {b'data': test_raw_data, b'labels': test_raw_labels}
    test_c_batch = {b'data': test_c_data, b'labels': test_c_labels}

    # 保存为 pickle 文件
    save_data(file_name+'/test_train_batch.pkl', test_raw_batch)
    save_data(file_name+'/test_test_batch.pkl', test_c_batch)

    print("test_raw 和 test_c 数据已成功保存。")
    
if __name__ == '__main__':
    main()