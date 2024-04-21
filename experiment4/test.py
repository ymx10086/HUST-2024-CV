import torch


# 定义钩子操作
def grad_hook(grad):
    y_grad.append(grad)
    
# 创建一个list来保存钩子获取的梯度
y_grad = list()

# 创建输入变量
x = torch.tensor([[1., 2.], [3., 4.]], requires_grad=True)  # requires_grad=True表明该节点为叶子节点
y = x + 1  # 没有明确requires_grad=True，所以是非叶子节点

# 为非叶子节点y注册钩子
y.register_hook(grad_hook)  # 这是传入的是函数而非函数的调用

y.retain_grad()  # 如果想要将 y 设置为叶子节点，可以设置 y.retain_grad()

# 计算z节点(z 也是一个非叶子节点，因为它是通过对非叶子节点 y 进行操作而得到的)
z = torch.mean(y * y)

# 反向传播
z.backward()

print(f"y.type: {type(y)}")
print(f"y.grad: {y.grad}")
print(f"y_grad: {y_grad}")
