# Path: experiment3/build_dataset.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import sys

# 读取dataset中的MNIST数据集
def load_dataset_ration():

    # 如果本地已经存在数据集，则直接读取本地数据集
    # if os.path.exists("./dataset/train_dataset1.pth") and os.path.exists("./dataset/test_dataset1.pth"):
    #     train_dataset = torch.load("./dataset/train_dataset1.pth")
    #     test_dataset = torch.load("./dataset/test_dataset1.pth")
    #     train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    #     test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    #     return train_loader, test_loader
    
    # 读取MNIST数据集
    mnist_train = MNIST(root="./dataset", train=True, download=True, transform=transforms.ToTensor())
    mnist_test = MNIST(root="./dataset", train=False, download=True, transform=transforms.ToTensor())

    # 分别获取其中标签为0，1，2，3，4，5，6，7，8，9的数据
    mnist_train_label = []
    minst_test_label = []
    for i in range(10):
        mnist_train_label.append(mnist_train.data[mnist_train.targets == i])
        minst_test_label.append(mnist_test.data[mnist_test.targets == i])

    # 绘制MNIST数据集中标签为0，1，2，3，4，5，6，7，8，9的数据的数量
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.bar(range(10), [len(mnist_train_label[0]), len(mnist_train_label[1]), len(mnist_train_label[2]), len(mnist_train_label[3]), len(mnist_train_label[4]), len(mnist_train_label[5]), len(mnist_train_label[6]), len(mnist_train_label[7]), len(mnist_train_label[8]), len(mnist_train_label[9])])
    plt.title('Train Count')
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.subplot(1, 2, 2)
    plt.bar(range(10), [len(minst_test_label[0]), len(minst_test_label[1]), len(minst_test_label[2]), len(minst_test_label[3]), len(minst_test_label[4]), len(minst_test_label[5]), len(minst_test_label[6]), len(minst_test_label[7]), len(minst_test_label[8]), len(minst_test_label[9])])
    plt.title('Test Count')
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.show()

    # 构建数据集，随机输入为两张MNIST手写体数字图片，输出为0或1，0表示两张图片不是同一个数字，1表示两张图片是同一个数字，数据量约为MNIST数据集
    train_dataset = []
    test_dataset = []

    # 首先构建两张图片的标签相同的数据集，数据量为500000，每个数字的数据量为50000
    for i in range(10):
        # 从MNIST数据集的训练集中随机选取两张图片
        for j in range(50000):
            index1 = torch.randint(0, len(mnist_train_label[i]), (1,)).item()
            index2 = torch.randint(0, len(mnist_train_label[i]), (1,)).item()
            # 如果两张图片的标签相同，则输出为1
            train_dataset.append([mnist_train_label[i][index1].unsqueeze(0), mnist_train_label[i][index2].unsqueeze(0), 1])

    # 然后构建两张图片的标签不同的数据集，数据量为450000，每种情况的数据量为10000
    for i in range(10):
        for j in range(i + 1, 10):
            # 从MNIST数据集的训练集中随机选取两张图片
            for k in range(10000):
                index1 = torch.randint(0, len(mnist_train_label[i]), (1,)).item()
                index2 = torch.randint(0, len(mnist_train_label[j]), (1,)).item()
                # 如果两张图片的标签不同，则输出为0
                train_dataset.append([mnist_train_label[i][index1].unsqueeze(0), mnist_train_label[j][index2].unsqueeze(0), 0])

    # 构建两张图片的标签相同的数据集，数据量为50000，每个数字的数据量为5000
    for i in range(10):
        # 从MNIST数据集的测试集中随机选取两张图片
        for j in range(5000):
            index1 = torch.randint(0, len(minst_test_label[i]), (1,)).item()
            index2 = torch.randint(0, len(minst_test_label[i]), (1,)).item()
            # 如果两张图片的标签相同，则输出为1
            test_dataset.append([minst_test_label[i][index1].unsqueeze(0), minst_test_label[i][index2].unsqueeze(0), 1])

    # 然后构建两张图片的标签不同的数据集，数据量为450000，每种情况的数据量为10000
    for i in range(10):
        for j in range(i + 1, 10):
            # 从MNIST数据集的测试集中随机选取两张图片
            for k in range(1000):
                index1 = torch.randint(0, len(minst_test_label[i]), (1,)).item()
                index2 = torch.randint(0, len(minst_test_label[j]), (1,)).item()
                # 如果两张图片的标签不同，则输出为0
                test_dataset.append([minst_test_label[i][index1].unsqueeze(0), minst_test_label[j][index2].unsqueeze(0), 0])

    # 构建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # 将数据集保存到本地
    torch.save(train_dataset, "./dataset/train_dataset1.pth")
    torch.save(test_dataset, "./dataset/test_dataset1.pth")

    return train_loader, test_loader

# 读取dataset中的MNIST数据集
def load_dataset_balanced():

    # 如果本地已经存在数据集，则直接读取本地数据集
    if os.path.exists("./dataset/train_dataset2.pth") and os.path.exists("./dataset/test_dataset2.pth"):
        train_dataset = torch.load("./dataset/train_dataset2.pth")
        test_dataset = torch.load("./dataset/test_dataset2.pth")
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
        return train_loader, test_loader
    
    # 读取MNIST数据集
    mnist_train = MNIST(root="./dataset", train=True, download=True, transform=transforms.ToTensor())
    mnist_test = MNIST(root="./dataset", train=False, download=True, transform=transforms.ToTensor())

    # 将mnist数据集打乱
    mnist_train.targets = torch.randperm(len(mnist_train.targets))
    mnist_test.targets = torch.randperm(len(mnist_test.targets))
    mnist_train_len = len(mnist_train)
    mnist_test_len = len(mnist_test)
    print("mnist_train: ", len(mnist_train))
    print("mnist_test: ", len(mnist_test))

    # 从MNIST数据集的训练集中随机选取10%作为本实验的训练图片，从MNIST数据集的测试集中随机选取10%作为本实验的测试图片
    mnist_train.data = mnist_train.data[:len(mnist_train.data) // 10]
    mnist_train.targets = mnist_train.targets[:len(mnist_train.targets) // 10]
    mnist_test.data = mnist_test.data[:len(mnist_test.data) // 10]
    mnist_test.targets = mnist_test.targets[:len(mnist_test.targets) // 10]
    print("mnist_train_cal: ", len(mnist_train))
    print("mnist_test_cal: ", len(mnist_test))

    # 构建数据集，随机输入为两张MNIST手写体数字图片，输出为0或1，0表示两张图片不是同一个数字，1表示两张图片是同一个数字，数据量约为MNIST数据集
    train_dataset = []
    # 首先构建 0表示两张图片不是同一个数字 的数据，数据量为MNIST数据集的一半
    while len(train_dataset) < mnist_train_len // 2:
        # 从MNIST数据集的训练集中随机选取两张图片
        index1 = torch.randint(0, len(mnist_train.data), (1,)).item()
        index2 = torch.randint(0, len(mnist_train.data), (1,)).item()
        # 如果两张图片的标签不同，则输出为0
        if mnist_train.targets[index1] != mnist_train.targets[index2]:
            train_dataset.append([mnist_train.data[index1].unsqueeze(0), mnist_train.data[index2].unsqueeze(0), 0])
    print("构建训练集完成一半！")
    # 然后构建 1表示两张图片是同一个数字 的数据，数据量为MNIST数据集的一半
    while len(train_dataset) < mnist_train_len:
        # 从MNIST数据集的训练集中随机选取两张图片
        index1 = torch.randint(0, len(mnist_train.data), (1,)).item()
        index2 = torch.randint(0, len(mnist_train.data), (1,)).item()
        # 如果两张图片的标签相同，则输出为1
        if mnist_train.targets[index1] == mnist_train.targets[index2]:
            train_dataset.append([mnist_train.data[index1].unsqueeze(0), mnist_train.data[index2].unsqueeze(0), 1])
            print(len(train_dataset))
    print("构建训练集完成！")

    test_dataset = []
    # 构建数据集，随机输入为两张MNIST手写体数字图片，输出为0或1，0表示两张图片不是同一个数字，1表示两张图片是同一个数字，数据量约为MNIST数据集
    while len(test_dataset) < mnist_test_len // 2:
        # 从MNIST数据集的测试集中随机选取两张图片
        index1 = torch.randint(0, len(mnist_test.data), (1,)).item()
        index2 = torch.randint(0, len(mnist_test.data), (1,)).item()
        # 如果两张图片的标签不同，则输出为0
        if mnist_test.targets[index1] != mnist_test.targets[index2]:
            test_dataset.append([mnist_test.data[index1].unsqueeze(0), mnist_test.data[index2].unsqueeze(0), 0])
    print("构建测试集完成一半！")
    while len(test_dataset) < mnist_test_len:
        # 从MNIST数据集的测试集中随机选取两张图片
        index1 = torch.randint(0, len(mnist_test.data), (1,)).item()
        index2 = torch.randint(0, len(mnist_test.data), (1,)).item()
        # 如果两张图片的标签相同，则输出为1
        if mnist_test.targets[index1] == mnist_test.targets[index2]:
            test_dataset.append([mnist_test.data[index1].unsqueeze(0), mnist_test.data[index2].unsqueeze(0), 1])
            print(len(test_dataset))
    print("构建测试集完成！")


    # 构建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # 打印数据集的大小
    print("train_dataset: ", len(train_dataset))
    print("test_dataset: ", len(test_dataset))

    # 分析数据集中的数据，统计每个标签的数量
    train_count = [0, 0]
    test_count = [0, 0]
    for i in range(len(train_dataset)):
        train_count[train_dataset[i][2]] += 1
    for i in range(len(test_dataset)):
        test_count[test_dataset[i][2]] += 1
    print("train_count: ", train_count)
    print("test_count: ", test_count)
    # 绘制数据集中每个标签的数量
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.bar(range(2), train_count)
    plt.title('Train Count')
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.subplot(1, 2, 2)
    plt.bar(range(2), test_count)
    plt.title('Test Count')
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.show()


    # 将数据集保存到本地
    torch.save(train_dataset, "./dataset/train_dataset2.pth")
    torch.save(test_dataset, "./dataset/test_dataset2.pth")

    return train_loader, test_loader
