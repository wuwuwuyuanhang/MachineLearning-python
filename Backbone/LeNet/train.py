# @Auther : wuwuwu 
# @Time : 2021/8/11 
# @File : train.py
# @Description : 使用 CIFAR-10 数据集进行训练

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from model import LeNet
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt


def main():
    # train parameters
    batch_size = 16     # 批量处理数量
    num_workers = 0     # 训练子进程数
    epochs = 10         # 训练轮数
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 训练设备，如果有 GPU 就使用第一块，没有就用CPU
    save_path = './logs/LeNet.pth'  # 权重保存位置
    load_path = './logs/LeNet.pth'  # 加载权重位置，首次训练时可设置为空

    # prepocess
    transform = transforms.Compose([
            transforms.ToTensor(),  # convert a PIL image or numpy.ndarray to tensor
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # data standardization
        ])

    # load dataset
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    train_num = len(train_loader)
    test_num = len(test_dataset)

    # LeNet model
    net = LeNet().to(device)
    if load_path is not None:
        print("load weights : " + load_path)
        net.load_state_dict(torch.load(load_path))
        print("finished load")

    # loss function
    loss_function = nn.CrossEntropyLoss()    # 交叉熵
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

    best_acc = 0.0
    losses, accuracies = [], []
    for epoch in range(epochs):
        # train
        net.train()  # 设置为训练模式
        running_loss = 0.0
        train_bar = tqdm(train_loader)   # 使用 tqdm 显示训练进度条
        for step, data in enumerate(train_bar):
            inputs, labels = data   # data 格式 [inputs, ]
            optimizer.zero_grad()   # 梯度清零，重新计算
            output = net(inputs.to(device))  # 前向传播，注意数据的环境
            loss = loss_function(output, labels.to(device))  # 计算损失函数
            loss.backward()  # 反向传播
            optimizer.step()    # 更新参数

            running_loss += loss.item()  # 损失函数值叠加

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch+1, epochs, loss)

        net.eval()  # 设置为测试模式， 关闭 Dropout
        acc = 0.0
        with torch.no_grad():   # 验证集和测试集不需要梯度更新
            val_bar = tqdm(test_loader)     # 使用 tqdm 显示训练进度条
            for data in val_bar:
                inputs, labels = data
                output = net(inputs.to(device))
                predict_y = torch.max(output, dim=1)[1]
                acc += torch.eq(predict_y, labels.to(device)).sum().item()

        val_accurate = acc / test_num

        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' % (epoch + 1, running_loss / train_num, val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

        losses.append(running_loss / train_num)
        accuracies.append(val_accurate)

    # 绘制 loss 和 accuracy 曲线
    fig = plt.figure(figsize=(6, 6))
    font_xy = {
        'family': 'Times New Roman',  # 字体格式为 Times New Roman
        'weight': 'normal',  # 字体粗细为正常大小
        'size': 20,  # 字体大小为 20 pt
        'color': 'black'  # 字体颜色为黑色
    }
    xlims = np.array(range(1, epochs + 1))  # 横坐标
    ax1 = fig.add_subplot(111)
    ax1.plot(xlims, losses, 'b')
    ax1.set_xlabel('epoch', font_xy)
    ax1.set_ylabel('training losses', font_xy)

    ax2 = ax1.twinx()   # 设置双 y 轴
    ax2.plot(xlims, accuracies, 'r')
    ax2.set_xlabel('epoch', font_xy)
    ax2.set_ylabel('testing accuracies', font_xy)

    fig.legend(('training losses', 'testing accuracies'))
    plt.savefig('./result/train.jpg')
    plt.show()


if __name__ == '__main__':
    main()