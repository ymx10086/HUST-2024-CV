# Path: experiment1/main.py
import torch
import torch.utils.data as Data
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.nn.functional as F
import torch.optim as optim
from net import Net
import tqdm
from logger import logger_to_file
import time
# import graphviz

# 下载“dataset.csv”数据集，其中包含四类二维高斯数据和它们的标签。
# 设计至少含有一层隐藏层的前馈神经网络来预测二维高斯样本(〖data〗_1,〖data〗_2)所属的分类label。
# 这个数据集需要先进行随机排序，然后选取90%用于训练，剩下的10%用于测试。

# 读取数据集
def read_dataset():
    dataset = pd.read_csv('dataset.csv')
    # 计算不同类别的数量
    print(dataset['label'].value_counts())
    dataset = dataset.sample(frac=1).reset_index(drop=True) # shuffle
    dataset = dataset.values
    return dataset

# 划分训练集和测试集
def split_dataset(dataset):
    train_size = int(len(dataset) * 0.9)
    train_set = dataset[:train_size]
    test_set = dataset[train_size:]
    return train_set, test_set

# 划分输入和标签
def split_input_label(dataset):
    input_set = dataset[:, :2]
    label_set = dataset[:, 2]
    # 对于所有的label，将其减1，使得label从0开始
    label_set = label_set - 1
    return input_set, label_set

# 将数据转换为Tensor
def convert_to_tensor(input_set, label_set):
    input_set = torch.from_numpy(input_set).float()
    label_set = torch.from_numpy(label_set).long()
    return input_set, label_set

# 将数据转换为TensorDataset
def convert_to_tensor_dataset(input_set, label_set):
    dataset = Data.TensorDataset(input_set, label_set)
    return dataset

# 将数据转换为DataLoader
def convert_to_dataloader(dataset, batch_size):
    dataloader = Data.DataLoader(dataset, batch_size=batch_size, shuffle=True) # shuffle
    return dataloader

# 训练模型, 并返回训练过程中的loss, acc, 用tqdm显示进度条
def train(net, dataloader, optimizer, criterion):
    loss_list = []
    acc_list = []
    for inputs, labels in tqdm.tqdm(dataloader):
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())
        acc_list.append((outputs.argmax(dim=1) == labels).sum().item() / len(labels))
    # 打印loss和acc
    print('Train: ' + 'Loss:', loss.item(), 'Acc:', (outputs.argmax(dim=1) == labels).sum().item() / len(labels))
    return loss_list, acc_list

# 测试模型, 并返回测试过程中的loss, acc
def test(net, dataloader, criterion):
    loss_list = []
    acc_list = []
    for inputs, labels in tqdm.tqdm(dataloader):
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss_list.append(loss.item())
        acc_list.append((outputs.argmax(dim=1) == labels).sum().item() / len(labels))
    return loss_list, acc_list

# 绘制loss和acc曲线
def plot_loss_acc(loss_list, acc_list, title):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(loss_list)
    plt.title(title + ' Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.subplot(1, 2, 2)
    plt.plot(acc_list)
    plt.title(title + ' Accuracy')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
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

# 画出决策边界
def plot_decision_boundary(net, dataset):
    # 画出决策边界
    x_min, x_max = dataset[:, 0].min() - 1, dataset[:, 0].max() + 1
    y_min, y_max = dataset[:, 1].min() - 1, dataset[:, 1].max() + 1
    h = 0.02
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))
    inputs = np.c_[xx.ravel(), yy.ravel()]
    inputs = torch.from_numpy(inputs).float()
    outputs = net(inputs)
    Z = outputs.argmax(dim=1).reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(dataset[:, 0], dataset[:, 1], c=dataset[:, 2], cmap=plt.cm.Spectral)
    plt.show()
    
# 评价模型
def evaluate(net, dataloader):
    # 计算整体的准确率
    correct = 0
    total = 0
    # 计算每个类别的准确率
    class_correct = list(0. for i in range(4))
    class_total = list(0. for i in range(4))
    with torch.no_grad():
        for inputs, labels in tqdm.tqdm(dataloader):
            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    print('Accuracy of the network on the images: %.2f %%' % (100 * correct / total))
    for i in range(4):
        print('Accuracy of %d : %.2f %%' % (i + 1, 100 * class_correct[i] / class_total[i]))


# 主函数
def main():

    # 读取数据集
    dataset = read_dataset()
    # 划分训练集和测试集
    train_set, test_set = split_dataset(dataset)
    # 划分输入和标签
    train_input_set, train_label_set = split_input_label(train_set)
    test_input_set, test_label_set = split_input_label(test_set)
    # 绘制训练集和标签在二维平面上的分布
    plt.scatter(train_input_set[:, 0], train_input_set[:, 1], c=train_label_set, cmap=plt.cm.Spectral)
    plt.show()

    # 将数据转换为Tensor
    train_input_set, train_label_set = convert_to_tensor(train_input_set, train_label_set)
    test_input_set, test_label_set = convert_to_tensor(test_input_set, test_label_set)
    # 将数据转换为TensorDataset
    train_dataset = convert_to_tensor_dataset(train_input_set, train_label_set)
    test_dataset = convert_to_tensor_dataset(test_input_set, test_label_set)
    # 将数据转换为DataLoader
    train_dataloader = convert_to_dataloader(train_dataset, 64)
    test_dataloader = convert_to_dataloader(test_dataset, 64)

    # 初始化网络
    net = Net(2, 4, 4)

    # 定义损失函数
    criterion = F.cross_entropy
    # 定义优化器
    optimizer = optim.SGD(net.parameters(), lr=0.1)
    # 训练模型
    # 计算训练时间
    start = time.time()
    train_loss_list, train_acc_list, test_loss_list, test_acc_list = train_model(net, train_dataloader, test_dataloader, optimizer, criterion, 100)
    end = time.time()
    print('Training time: ', end - start)
    # 绘制loss和acc曲线
    plot_loss_acc(train_loss_list, train_acc_list, 'Train')
    plot_loss_acc(test_loss_list, test_acc_list, 'Test')
    # 画出决策边界
    plot_decision_boundary(net, dataset)
    # 保存模型
    torch.save(net, 'net.pkl')
    
    # 评价模型
    print('Evaluation for test:')
    evaluate(net, test_dataloader)

    # 评价模型
    print('Evaluation for train:')
    evaluate(net, train_dataloader)
    
    

if __name__ == '__main__':
    logger_to_file()
    main()
                