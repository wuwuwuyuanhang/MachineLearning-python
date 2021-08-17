# @Auther : wuwuwu 
# @Time : 2021/8/14 
# @File : model.py
# @Description :

import torch
import torch.nn as nn
from torchsummary import summary


class AlexNet(nn.Module):
    def __init__(self, class_num, init_weight=False):
        super(AlexNet, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(3, 48, 11, stride=4, padding=2),  # [3, 224, 224]-->[48, 55, 55]
            nn.ReLU(inplace=True),  # 原地操作，减少内存开销
            nn.MaxPool2d(kernel_size=3, stride=2),  # [48, 55, 55]-->[48, 27, 27]
            nn.Conv2d(48, 128, 5, stride=1, padding=2),  # [48, 27, 27]-->[128, 27, 27]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # [128, 27, 27]-->[128, 13, 13]
            nn.Conv2d(128, 192, 3, stride=1, padding=1),  # [128, 13, 13]-->[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, 3, stride=1, padding=1),  # [192, 13, 13]-->[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, 3, stride=1, padding=1),  # [192, 13, 13]-->[128, 13, 13]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)  # [128, 13, 13]-->[128, 6, 6]
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),  # dropout
            nn.Linear(128 * 6 * 6, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, class_num)
        )

        if init_weight:
            self._initialize_weights()

    def forward(self, x):
        x = self.feature(x)
        x = torch.flatten(x, start_dim=1)  # 从第 1 维开始展平操作
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():  # 遍历每一层
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  # 卷积层：权重用 凯明归一化 初始化
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)  # 卷积层：偏置用 0 初始化
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)  # 全连接层：权重用 均值为0，标准差0.01 正态分布 初始化
                nn.init.constant_(m.bias, 0)  # 全连接层：偏置用 0 初始化


if __name__ == '__main__':
    net = AlexNet(class_num=5)
    if torch.cuda.is_available():
        net = net.cuda()
    summary(net, (3, 224, 224))
