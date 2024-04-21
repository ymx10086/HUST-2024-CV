# Path: experiment2/main.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from tqdm import tqdm
import matplotlib.pyplot as plt

from logger import logger_to_file
from net import Net1, Net2

# 读取dataset中的MNIST数据集
def load_dataset():
    # 读取MNIST数据集
    mnist_train = MNIST(root="./dataset", train=True, download=True, transform=transforms.ToTensor())
    mnist_test = MNIST(root="./dataset", train=False, download=True, transform=transforms.ToTensor())

    # 构建DataLoader
    train_loader = DataLoader(mnist_train, batch_size=128, shuffle=True)
    test_loader = DataLoader(mnist_test, batch_size=128, shuffle=False)

    return train_loader, test_loader

# 训练模型, 并返回训练过程中的loss, acc, 用tqdm显示进度条
def train(net, dataloader, optimizer, criterion):
    loss_list = []
    acc_list = []
    for inputs, labels in tqdm(dataloader):
        inputs = inputs.cuda()
        labels = labels.cuda()

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())
        _, predicted = torch.max(outputs, 1)
        acc_list.append((predicted == labels).sum().item() / len(labels))
    # 打印loss和acc
    print('Train: ' + 'Loss:', loss.item(), 'Acc:', (predicted == labels).sum().item() / len(labels))
    return loss_list, acc_list

# 测试模型, 并返回测试过程中的loss, acc
def test(net, dataloader, criterion):
    loss_list = []
    acc_list = []
    for inputs, labels in tqdm(dataloader):
        inputs = inputs.cuda()
        labels = labels.cuda()
        
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss_list.append(loss.item())
        _, predicted = torch.max(outputs, 1)
        acc_list.append((predicted == labels).sum().item() / len(labels))
    return loss_list, acc_list

# 绘制loss和acc曲线
def plot_loss_acc(loss_list, acc_list, title, tag=False):
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(loss_list)
    plt.title(title + ' Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    if tag:
        plt.ylim(0, max(loss_list))  # 设置纵坐标从0开始
    
    plt.subplot(1, 2, 2)
    plt.plot(acc_list)
    plt.title(title + ' Accuracy')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.0)  # 如果准确度的范围是[0, 1]，你可以将最大值设置为1.0
    
    plt.show()

# 训练模型
def train_model(net, train_dataloader, test_dataloader, optimizer, criterion, epoch):
    train_loss_list = []
    train_acc_list = []
    test_loss_list = []
    test_acc_list = []
    for i in range(epoch):
        print('Epoch', i + 1)
        train_loss, train_acc = train(net, train_dataloader, optimizer, criterion)
        test_loss, test_acc = test(net, test_dataloader, criterion)
        train_loss_list += train_loss
        train_acc_list += train_acc
        test_loss_list += test_loss
        test_acc_list += test_acc
    return train_loss_list, train_acc_list, test_loss_list, test_acc_list

# 评价模型
def evaluate(net, dataloader):
    # 计算整体的准确率
    correct = 0
    total = 0
    # 计算每个标签的准确率
    label_correct = [0 for i in range(10)]
    label_total = [0 for i in range(10)]
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader):
            inputs = inputs.cuda()
            labels = labels.cuda()

            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            for i in range(len(labels)):
                label_total[labels[i]] += 1
                if predicted[i] == labels[i]:
                    label_correct[labels[i]] += 1
    # 准确率精确到小数点后两位
    print('Accuracy of the network on the test images: %.2f %%' % (100 * correct / total))
    for i in range(10):
        print('Accuracy of %d: %.2f %%' % (i, 100 * label_correct[i] / label_total[i]))


# 主函数
def main():
    # 读取MNIST数据集
    train_loader, test_loader = load_dataset()

    # 定义网络
    # net = Net1().to('cuda:0')
    net = Net2().cuda()

    # 定义损失函数
    criterion = nn.CrossEntropyLoss()

    # 定义优化器
    optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=0.0001)

    # 训练模型
    train_loss_list, train_acc_list, test_loss_list, test_acc_list = train_model(net, train_loader, test_loader, optimizer, criterion, epoch=4)

    # 绘制loss和acc曲线
    plot_loss_acc(train_loss_list, train_acc_list, 'Train')
    plot_loss_acc(test_loss_list, test_acc_list, 'Test', tag=True)
    
    # 评价模型
    print('Test:')
    evaluate(net, test_loader)

    # 评价模型
    print('Train:')
    evaluate(net, train_loader)

if __name__ == "__main__":
    logger_to_file()
    main()

