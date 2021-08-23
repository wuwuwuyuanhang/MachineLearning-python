# @Auther : wuwuwu 
# @Time : 2021/8/22 
# @File : train.py
# @Description :


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import os
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from model import GoogLeNet


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
    # 训练参数
    batch_size = 4  # 批量训练数
    freeze_epochs = 10  # 冻结参数训练 10 轮
    unfreeze_epochs = 40  # 释放参数训练 10 轮
    load_path = './logs/googlenet-1378be20.pth'  # 预训练权重
    save_path = './logs/googlenet.pth'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # 训练环境

    # 预处理
    transform = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),  # 随机剪裁成 224 * 224
            transforms.RandomHorizontalFlip(),  # 随机水平翻转
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]),
        'test': transforms.Compose([
            transforms.Resize((224, 224)),  # 将图片缩放到 224 * 224 ，不进行裁剪
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    }

    root_data = os.getcwd()  # 获取当前路径
    root_data = os.path.join(root_data, '../../dataset/birds')  # 返回上两级路径，再进入到数据集文件夹

    train_dataset = datasets.ImageFolder(root=os.path.join(root_data, 'train'), transform=transform['train'])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    train_num = len(train_dataset)

    cla_dict = train_dataset.class_to_idx
    label_dict = dict((val, key) for key, val in cla_dict.items())

    with open('class_indices.txt', 'w') as f:
        for k, v in label_dict.items():
            f.write(str(k) + ' ' + v + '\n')

    test_dataset = datasets.ImageFolder(root=os.path.join(root_data, 'test'), transform=transform['test'])
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_num = len(test_dataset)

    net = GoogLeNet(num_classes=5).to(device)

    # 获取预训练权重 字典
    pretrained_state_dict = torch.load(load_path)
    net_dict = net.state_dict()
    # 创建需要加载的预训练权重 字典
    pretrained_dict = {key: val for key, val in pretrained_state_dict.items()
                       if key in net_dict and net_dict[key].shape == pretrained_state_dict[key].shape}
    # print(list(pretrained_dict))
    net_dict.update(pretrained_dict)  # 更新权重
    net.load_state_dict(net_dict)  # 加载权重

    for name, parameter in net.named_parameters():
        if name in pretrained_dict:
            parameter.requires_grad = False

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=1e-3, weight_decay=1e-5)

    # 冻结权重训练 freeze_epochs 轮
    best_acc = 0.0
    losses, accuracies = [], []
    for epoch in range(freeze_epochs):
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader)
        for data in train_bar:
            inputs, labels = data
            optimizer.zero_grad()
            logits, aux_logits2, aux_logits1 = net(inputs.to(device))  # GoogLeNet 返回三个分类值
            loss0 = loss_function(logits, labels.to(device))
            loss1 = loss_function(aux_logits1, labels.to(device))
            loss2 = loss_function(aux_logits2, labels.to(device))
            loss = loss0 + loss1 * 0.3 + loss2 * 0.3  # 损失函数为主分类器加两个辅助分类器，辅助分类器权重为 0.3
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            train_bar.desc = 'train epoch[{}/{}], training loss: {:.3f}'.format(epoch + 1,
                                                                                freeze_epochs + unfreeze_epochs, loss)

        net.eval()
        acc = 0.0
        with torch.no_grad():
            val_bar = tqdm(test_loader)
            for data in val_bar:
                inputs, labels = data
                outputs = net(inputs.to(device))  # 预测时，去掉两个辅助分类器，只看主分类器
                predict_y = torch.max(outputs, 1)[1]
                acc += torch.eq(predict_y, labels.to(device)).sum().item()

        val_accuracy = acc / test_num
        print("[Epoch: %d] training loss: %.3f, test accuracy: %.3f"
              % (epoch + 1, running_loss / train_num, val_accuracy))

        losses.append(running_loss / train_num)
        accuracies.append(val_accuracy)

        if val_accuracy > best_acc:
            best_acc = val_accuracy
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
        for data in train_bar:
            inputs, labels = data
            optimizer.zero_grad()
            logits, aux_logits2, aux_logits1 = net(inputs.to(device))  # GoogLeNet 返回三个分类值
            loss0 = loss_function(logits, labels.to(device))
            loss1 = loss_function(aux_logits1, labels.to(device))
            loss2 = loss_function(aux_logits2, labels.to(device))
            loss = loss0 + loss1 * 0.3 + loss2 * 0.3  # 损失函数为主分类器加两个辅助分类器，辅助分类器权重为 0.3
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            train_bar.desc = 'train epoch[{}/{}], training loss: {:.3f}'.format(epoch + 1,
                                                                                freeze_epochs + unfreeze_epochs, loss)

        net.eval()
        acc = 0.0
        with torch.no_grad():
            val_bar = tqdm(test_loader)
            for data in val_bar:
                inputs, labels = data
                outputs = net(inputs.to(device))  # 预测时，去掉两个辅助分类器，只看主分类器
                predict_y = torch.max(outputs, 1)[1]
                acc += torch.eq(predict_y, labels.to(device)).sum().item()

        val_accuracy = acc / test_num
        print("[Epoch: %d] training loss: %.3f, test accuracy: %.3f"
              % (epoch + 1, running_loss / train_num, val_accuracy))

        losses.append(running_loss / train_num)
        accuracies.append(val_accuracy)

        if val_accuracy > best_acc:
            best_acc = val_accuracy
            torch.save(net.state_dict(), save_path)

    plot_loss_accuracy(losses, accuracies)


if __name__ == '__main__':
    main()
