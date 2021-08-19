# @Auther : wuwuwu 
# @Time : 2021/8/18 
# @File : trian.py
# @Description :


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import os
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
from model import vgg


def plot_loss_accuracy(losses, accuracy):
    xlims = np.arange(1, len(losses) + 1)
    font_xy = {
        'family': 'Times New Roman',
        'weight': 'normal',
        'size': 20,
        'color': 'black'
    }
    fig = plt.figure(figsize=(8, 7))
    ax1 = fig.add_subplot(111)
    ax1.plot(xlims, losses, 'b')
    ax1.set_xlabel('epoch', font_xy)
    ax1.set_ylabel('training loss', font_xy)

    ax2 = ax1.twinx()
    ax2.plot(xlims, accuracy, 'r')
    ax2.set_xlabel('epoch', font_xy)
    ax2.set_ylabel('testing accuracy', font_xy)

    fig.legend(('training loss', 'testing accuracy'))
    plt.savefig('./result/train.jpg')
    plt.show()


def main():
    # 参数设置
    batch_size = 4
    freeze_epochs = 10  # 冻结前面的权重训练 10 轮
    unfreeze_epochs = 40  # 释放权重后整体训练 10 轮
    load_path = './logs/vgg16_bn-6c64b313.pth'
    save_path = './logs/vgg16_bn_birds.pth'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    transform = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]),
        'test': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    }

    data_root = './data'

    train_dataset = datasets.ImageFolder(root=os.path.join(data_root, 'train'), transform=transform['train'])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    train_num = len(train_dataset)

    label_dict = train_dataset.class_to_idx
    label_dict = dict((val, key) for key, val in label_dict.items())
    with open('class_indices.txt', 'w') as f:
        for key, val in label_dict.items():
            f.write(str(key) + ' ' + val + '\n')

    test_dataset = datasets.ImageFolder(root=os.path.join(data_root, 'test'), transform=transform['test'])
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_num = len(test_dataset)

    net = vgg('vgg16', batch_norm=True, class_num=5)
    net = net.to(device)

    # 加载预训练模型 dict
    pretrained_state_dict = torch.load(load_path)
    # 获取当前模型的结构 dict
    net_state_dict = net.state_dict()
    # 提取二者相同部分的权重 (名字相同，且对应权重形状一样)
    pretrained_dict = {key: val for key, val in pretrained_state_dict.items()
                       if key in net_state_dict and val.shape == net_state_dict[key].shape}
    # print(list(pretrained_dict))
    # 更新当前模型结构中的权重
    net_state_dict.update(pretrained_dict)
    # 加载更新后的权重
    net.load_state_dict(net_state_dict)

    # 冻结 Backbone 部分，训练自己设置的网络层
    loss_function = nn.CrossEntropyLoss()  # 交叉熵损失函数
    for name, parameter in net.named_parameters():
        # 遍历 net 中的参数，如果其中有
        if name in pretrained_dict:
            # print("冻结参数{}".format(name))
            parameter.requires_grad = False
    # 使用 filter 函数迭代将 net 中的参数设置为是否训练
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=1e-3, weight_decay=1e-5)

    best_acc = 0.0
    losses, accuracies = [], []
    for epoch in range(0, freeze_epochs):
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

            train_bar.desc = 'Epoch[{}/{}], training loss:{:.3f}'.format(epoch + 1, freeze_epochs + unfreeze_epochs, loss)

        net.eval()
        acc = 0.0
        with torch.no_grad():
            test_bar = tqdm(test_loader)
            for data in test_bar:
                inputs, labels = data
                outputs = net(inputs.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, labels.to(device)).sum().item()

        test_accuracy = acc / test_num
        print("[epoch:{}]: training loss:{:.3f}, test accuracy:{:.3f}".format(epoch + 1, running_loss / train_num, test_accuracy))

        losses.append(running_loss / train_num)
        accuracies.append(test_accuracy)

        if test_accuracy > best_acc:
            best_acc = test_accuracy
            torch.save(net.state_dict(), save_path)

    print('Freeze Backbone!')
    for name, parameter in net.named_parameters():
        parameter.requires_grad = True
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=1e-3, weight_decay=1e-5)
    # 释放权重，整体训练 unfreeze_epochs 轮
    for epoch in range(freeze_epochs, freeze_epochs + unfreeze_epochs):
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

            train_bar.desc = 'Epoch[{}/{}], training loss:{:.3f}'.format(epoch + 1, freeze_epochs + unfreeze_epochs, loss)

        net.eval()
        acc = 0.0
        with torch.no_grad():
            test_bar = tqdm(test_loader)
            for data in test_bar:
                inputs, labels = data
                outputs = net(inputs.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, labels.to(device)).sum().item()

        test_accuracy = acc / test_num
        print("[epoch:{}]: training loss:{:.3f}, test accuracy:{:.3f}".format(epoch + 1, running_loss / train_num, test_accuracy))

        losses.append(running_loss / train_num)
        accuracies.append(test_accuracy)

        if test_accuracy > best_acc:
            best_acc = test_accuracy
            torch.save(net.state_dict(), save_path)

    plot_loss_accuracy(losses, accuracies)


if __name__ == '__main__':
    main()
