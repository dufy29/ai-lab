"""
关于pytorch 前向loss，反向梯度以及参数更新的demo

"""
import torch
import numpy as np
import torch.nn as nn

seed = 1234 #seed必须是int，可以自行设置
torch.manual_seed(seed)

x_values = [i for i in range(11)] # [0,1,2,3,4,5,6,7,8,9,10]
x_train = np.array(x_values, dtype=np.float32)
x_train = x_train.reshape(-1, 1) # 将x_train调整为11*1的矩阵

y_values = [2*i+0 for i in x_values] # y=2x + 0
y_train = np.array(y_values, dtype=np.float32)
y_train = y_train.reshape(-1, 1)
print(f'label: {y_train.T}')

class LinearRegressionModel(nn.Module): # 继承自nn包的Module类

    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModel, self).__init__() # 执行父类的构造函数
        self.linear = nn.Linear(input_dim, output_dim)
        # nn.Linear(输入数据维度, 输出数据维度) 全连接层

    def forward(self, x):
        out = self.linear(x)
        return out
    
## 进行模型参数初始化
def init_weights1(model):
    if isinstance(model, nn.Linear):
        torch.nn.init.xavier_uniform(model.weight)
        model.bias.data.fill_(1)

input_dim = 1
output_dim = 1
model = LinearRegressionModel(input_dim, output_dim)
model.apply(init_weights1)
print(model.state_dict()['linear.weight'])

epochs = 10 # 训练次数
learning_rate = 0.01 # 学习率
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)# 优化器
criterion = nn.MSELoss() # 回归任务可选用MSE等

# x_train和y_train均为numpy.ndarry格式，需要转换为tensor格式才可以传入框架
inputs = torch.from_numpy(x_train)
labels = torch.from_numpy(y_train)

for epoch in range(epochs):
    epoch += 1
    print(f'='*40)
    for name in model.state_dict():  # 打印参数权重
        print(name, model.state_dict()[name])

    # print(f'计算梯度??')
    # for name, parms in model.named_parameters():
    #     print('-->name:', name, ' -->grad_value:',parms.grad)
    # 每次迭代开始时 梯度需要清零
    optimizer.zero_grad()
    # print(f'计算梯度??')
    # for name, parms in model.named_parameters():
    #     print('-->name:', name, ' -->grad_value:',parms.grad)

    # 前向传播
    outputs = model.forward(inputs)

    # 计算损失
    loss = criterion(outputs, labels)
    # 每50个epoch输出一次，以显示训练进度
    if epoch % 1 == 0:
        print('epoch {}, loss {}'.format(epoch, loss.item()))
    
    # 反向传播
    loss.backward()
    print(f'计算梯度')
    for name, parms in model.named_parameters():
        print(f'{learning_rate * parms.grad.item()}')
        print('-->name:', name, ' -->grad_value:',parms.grad)

    # 更新权重参数
    optimizer.step()

    print(f'梯度更新后的参数：')
    for name in model.state_dict():  # 打印参数权重
        print(name, model.state_dict()[name])



y_predicted = model.forward(torch.from_numpy(x_train)).data.numpy()
print(f'y_predicted: {y_predicted.shape}')
print(y_predicted.T)
