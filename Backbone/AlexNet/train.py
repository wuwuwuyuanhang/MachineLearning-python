# @Auther : wuwuwu 
# @Time : 2021/8/16 
# @File : train.py
# @Description :

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
import os
from model import AlexNet


def main():
    # 参数设置
    Batch_size = 8
    Epochs = 50
    save_path = './logs/AlexNet.pth'
    load_path = ''

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    transform = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),  # 随机剪裁图片，并调整大小到 224 * 224
            transforms.RandomHorizontalFlip(),  # 图片随机水平翻转
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]),
        'test': transforms.Compose([
            transforms.Resize((224, 224)),  # 测试集图片不剪裁，直接转换为 224 * 224 大小
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    }

    data_path = './data'

    train_dataset = datasets.ImageFolder(root=os.path.join(data_path, 'train'), transform=transform['train'])
    train_loader = DataLoader(train_dataset, batch_size=Batch_size, shuffle=True, num_workers=0)
    train_num = len(train_dataset)

    # 字典格式转换标签
    label_list = train_dataset.class_to_idx
    label_dict = dict((val, key) for key, val in label_list.items())  # 转换关键字和键值，即 数字 ：标签

    with open('class_indices.txt', 'w') as f:
        for key, val in label_dict.items():
            f.write(str(key) + " " + val + "\n")

    test_dataset = datasets.ImageFolder(root=os.path.join(data_path, 'test'), transform=transform['test'])
    test_loader = DataLoader(test_dataset, batch_size=Batch_size, shuffle=False, num_workers=0)
    test_num = len(test_dataset)

    # 实例化
    net = AlexNet(class_num=5)
    net = net.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)

    # if load_path is not None:
    #     print("load weights : " + load_path)
    #     net.load_state_dict(torch.load(load_path))
    #     print("finished load")

    best_acc = 0.0
    losses, accuracies = [], []
    for epoch in range(Epochs):
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader)
        for step, data in enumerate(train_bar):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, Epochs, loss)

        net.eval()
        acc = 0.0
        with torch.no_grad():
            val_bar = tqdm(test_loader)
            for data in val_bar:
                inputs, labels = data
                outputs = net(inputs.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, labels.to(device)).sum().item()

        val_accuracy = acc / test_num

        print("[epoch: %d] train_loss: %.3f, test_accuracy: %.3f" % (epoch + 1, running_loss / train_num, val_accuracy))

        if val_accuracy > best_acc:
            best_acc = val_accuracy
            torch.save(net.state_dict(), save_path)

        losses.append(running_loss / train_num)
        accuracies.append(val_accuracy)

    fig = plt.figure(figsize=(7, 7))
    font_xy = {
        'family': 'Times New Roman',
        'weight': 'normal',
        'size': 20,
        'color': 'black'
    }
    xlims = np.arange(1, Epochs + 1)
    ax1 = fig.add_subplot(111)
    ax1.plot(xlims, losses, 'b')
    ax1.set_xlabel('epoch', font_xy)
    ax1.set_ylabel('training loss', font_xy)

    ax2 = ax1.twinx()
    ax2.plot(xlims, accuracies, 'r')
    ax2.set_xlabel('epoch', font_xy)
    ax2.set_ylabel('test accuracy', font_xy)

    fig.legend(('training loss', 'test_accuracy'))
    plt.savefig('./result/train.jpg')
    plt.show()


if __name__ == '__main__':
    main()
