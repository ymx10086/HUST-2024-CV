# Path: experiment2/net.py
import torch
import torch.nn as nn

# 设计一个卷积神经网络，并在其中使用ResNet模块，在MNIST数据集上实现10分类手写体数字识别。
class ResidualBlock(nn.Module):

    def __init__(self, channel):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channel, channel, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channel, channel, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out += identity
        out = self.relu(out)
        return out
    
# 输入为两张MNIST手写体数字图片，输出为0或1，0表示两张图片不是同一个数字，1表示两张图片是同一个数字
class Net(torch.nn.Module):
    def __init__(self, num_classes=2, num_channels=1, img_size=28, num_filters=16, kernel_size=3):
        super(Net, self).__init__()
        self.num_classes = num_classes
        self.num_channels = num_channels
        self.img_size = img_size
        self.num_filters = num_filters
        self.kernel_size = kernel_size

        self.conv1 = nn.Conv2d(self.num_channels, self.num_filters, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(self.num_filters, self.num_filters * 2, kernel_size=3, padding=1)
        self.residual_block1 = ResidualBlock(self.num_filters)
        self.residual_block2 = ResidualBlock(self.num_filters * 2)
        self.max_pool = nn.MaxPool2d(2)
        
        self.linear = nn.Linear(self.num_filters * 2 * (self.img_size // 4) * (self.img_size // 4) * 2, self.num_filters * 2 * (self.img_size // 4) * (self.img_size // 4))
        self.classifier = nn.Linear(self.num_filters * 2 * (self.img_size // 4) * (self.img_size // 4), self.num_classes)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x, y):
        out1 = self.conv1(x)
        out1 = self.relu(out1)
        out1 = self.residual_block1(out1)
        out1 = self.max_pool(out1)
        out1 = self.conv2(out1)
        out1 = self.relu(out1)
        out1 = self.residual_block2(out1)
        out1 = self.max_pool(out1)

        out2 = self.conv1(y)
        out2 = self.relu(out2)
        out2 = self.residual_block1(out2)
        out2 = self.max_pool(out2)
        out2 = self.conv2(out2)
        out2 = self.relu(out2)
        out2 = self.residual_block2(out2)
        out2 = self.max_pool(out2)

        out = torch.cat((out1, out2), 1)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.classifier(out)
        
        out = self.softmax(out)
        # print(out)

        return out

