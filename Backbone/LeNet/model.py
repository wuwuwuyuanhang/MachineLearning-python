# @Auther : wuwuwu 
# @Time : 2021/8/5 
# @File : LeNet.py
# @Description : LeNet-5

import torch
import torch.nn as nn
from torchsummary import summary


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 120, 5),
            nn.ReLU()
        )
        self.linear = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        output = self.conv(x)
        output = output.view(-1, 120*1*1)
        output = self.linear(output)
        return output


if __name__ == '__main__':
    lenet_model = LeNet()
    if torch.cuda.is_available():
        lenet_model = lenet_model.cuda()
    summary(lenet_model, (1, 32, 32))