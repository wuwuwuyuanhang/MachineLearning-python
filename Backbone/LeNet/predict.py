# @Auther : wuwuwu 
# @Time : 2021/8/11 
# @File : predict.py
# @Description : 网络预测

import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
from matplotlib import pyplot as plt
from model import LeNet


def plot_sample(net, sample_data, sample_label, classes, load_path, device):
    """
    使用 matplotlib 库绘制预测结果
    :param net: 网络模型
    :param sample_data: 输入数据
    :param sample_label: 输入标签
    :param classes: 所有标签
    :param load_path: 训练模型
    :param device: 预测环境
    :return:
    """
    net = net.to(device)
    print("load weights : " + load_path)
    net.load_state_dict(torch.load(load_path))
    print("finished load")

    predict_y = net(sample_data.to(device))
    predict_label = torch.max(predict_y, dim=1)[1]  # 预测结果
    predict_label = predict_label.cpu().numpy()  # torch.tensor 转为 numpy.ndarray

    show_img = sample_data.numpy()
    show_label = sample_label.numpy()

    plt.figure(figsize=(8, 8))  # 设置显示框大小
    length = len(sample_data)  # 一组数据长度，即 batch_size 大小
    font2 = {
        'family': 'Times New Roman',  # 字体格式为 Times New Roman
        'weight': 'normal',  # 字体粗细为正常大小
        'size': 20  # 字体大小为 20 pt
    }
    for i in range(length):
        plt.subplot(2, length / 2, i + 1)  # 分开显示
        plt.xticks([])  # 不显示 x 轴坐标
        plt.yticks([])  # 不显示 y 轴坐标
        img = np.transpose(show_img[i, ...], (1, 2, 0))  # 转换维度, [C, H, W]-->[H, W, C]
        img = img / 2 + 0.5  # 去归一化
        plt.imshow(img)
        if predict_label[i] == show_label[i]:
            font2['color'] = 'green'  # 正确分类用绿色显示
            plt.xlabel(classes[predict_label[i]], font2)
        else:
            font2['color'] = 'red'  # 错误分类用红色显示
            plt.xlabel(classes[predict_label[i]], font2)
        font2['color'] = 'black'
        plt.title('Predict', font2)
    plt.savefig('./result/predict.jpg')
    plt.show()


def main():
    # 参数
    load_path = './logs/LeNet.pth'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    # 测试集
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=True, num_workers=0)

    # 所有标签
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    net = LeNet(classes_num=10)

    test_data_iter = iter(test_loader)  # 迭代器
    test_data, test_label = test_data_iter.next()   # 取下一组数据
    plot_sample(net, test_data, test_label, classes, load_path, device)


if __name__ == '__main__':
    main()
