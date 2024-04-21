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

class Net1(torch.nn.Module):
    def __init__(self, num_classes=10, num_channels=1, img_size=28, num_filters=16, kernel_size=3):
        super(Net1, self).__init__()
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
        self.linear = nn.Linear(self.num_filters * 2 * (self.img_size // 4) * (self.img_size // 4), self.num_classes)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):

        out = self.conv1(x)
        out = self.relu(out)
        out = self.residual_block1(out)
        out = self.max_pool(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.residual_block2(out)
        out = self.max_pool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        out = self.softmax(out)
        
        return out
    
# 设计一个卷积神经网络，实现MNIST数据集上的10分类手写体数字识别。
# 参考LeNet-5模型，构建一个卷积神经网络，实现MNIST数据集上的10分类手写体数字识别。
class Net2(torch.nn.Module):
    def __init__(self, num_classes=10, num_channels=1, img_size=28, num_filters=6, kernel_size=3):
        super(Net2, self).__init__()
        self.num_classes = num_classes
        self.num_channels = num_channels
        self.img_size = img_size
        self.num_filters = num_filters
        self.kernel_size = kernel_size

        self.conv1 = nn.Sequential(     #input_size=(1*28*28)
            #第一个卷积层，输入通道为1，输出通道为6，卷积核大小为5，步长为1，填充为2保证输入输出尺寸相同
            nn.Conv2d(1, 6, 5, 1, 2), #padding=2保证输入输出尺寸相同
            #激活函数,两个网络层之间加入，引入非线性

            nn.ReLU(),      #input_size=(6*28*28)
            #池化层，大小为2步长为2
            nn.MaxPool2d(kernel_size=2, stride=2),#output_size=(6*14*14)
        )
        self.residual_block1 = ResidualBlock(self.num_filters)
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),      #input_size=(16*10*10)
            nn.MaxPool2d(2, 2)  #output_size=(16*5*5)
        )
        #全连接层，输入是16*5*5特征图，神经元数目120
        self.fc1 = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU()
        )
        #全连接层神经元数目输入为上一层的120，输出为84
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU()
        )
        #最后一层全连接层神经元数目10，与上一个全连接层同理
        self.fc3 = nn.Linear(84, 10)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        
        out = self.conv1(x)
        out = self.residual_block1(out)
        out = self.conv2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.softmax(out)
        return out


        

