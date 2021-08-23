# @Auther : wuwuwu 
# @Time : 2021/8/23 
# @File : predict.py
# @Description :


import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import os
import numpy as np
from matplotlib import pyplot as plt
from model import GoogLeNet


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
    missing_keys, unexpected_keys = net.load_state_dict(torch.load(load_path), strict=False)  # 非强制加载，返回缺失和冗余
    print("finished load")

    predict_y = net(sample_data.to(device))
    predict_label = torch.max(predict_y, dim=1)[1]  # 预测结果
    predict_label = predict_label.cpu().numpy()  # torch.tensor 转为 numpy.ndarray

    show_img = sample_data.numpy()
    show_label = sample_label.numpy()

    plt.figure(figsize=(15, 8))  # 设置显示框大小
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
        plt.title(classes[show_label[i]], font2)
    plt.savefig('./result/predict.jpg')
    plt.show()


def main():
    load_path = './logs/googlenet.pth'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    root_data = os.path.join(os.getcwd(), '../../dataset/birds')

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 测试集
    test_dataset = datasets.ImageFolder(root=os.path.join(root_data, 'test'), transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=True, num_workers=0)

    # 所有标签
    classes = []
    with open('class_indices.txt', 'r') as f:
        cla_list = f.readlines()
    for cla in cla_list:
        class_name = cla.split(' ')[-1]
        class_name = class_name[:-1]
        classes.append(class_name)

    net = GoogLeNet(num_classes=5, aux_logits=False)

    test_data_iter = iter(test_loader)  # 迭代器
    test_data, test_label = test_data_iter.next()  # 取下一组数据
    plot_sample(net, test_data, test_label, classes, load_path, device)


if __name__ == '__main__':
    main()
