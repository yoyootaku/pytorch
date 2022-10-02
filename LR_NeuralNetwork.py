import torch
import matplotlib.pyplot as plt
from torch import nn,optim
from time import perf_counter

x = torch.unsqueeze(torch.linspace(-3,3,100000),dim=1)
y = x + 1.2 * torch.rand(x.size())

class LR(nn.Module):
    #初始化
    def __init__(self):
        super(LR,self).__init__() #父类初始化函数
        self.linear = nn.Linear(1,1) #构造线性模型，输入输出均为1维
    #前向传播
    def forward(self,x):
        out = self.linear(x)
        return out

#创建对象
LR_model = LR()
inputs = x
target = y

#损失函数（nn模块中的均方误差）
criterion = nn.MSELoss()

#随机梯度下降SGD
optimizer = optim.SGD(LR_model.parameters(),lr=1e-4)

#训练函数,参数：model：被训练的神经网络模型、criterion:损失函数、optimizer:优化器、epochs:训练轮数
def train(model,criterion,optimizer,epochs):
    for epoch in range(epochs):
        output = model(inputs)#前向传播
        loss = criterion(output,target)#计算损失值
        optimizer.zero_grad()#反向传播清空权重
        loss.backward()#反向传播计算梯度
        optimizer.step()#权值更新
        if epoch % 80 == 0:
            draw(output,loss)
    return model,loss

#绘图
def draw(output,loss):
    plt.cla()
    plt.scatter(x.numpy(),y.numpy())#绘制散点图
    plt.plot(x.numpy(),output.data.numpy(),'r-',lw=5)
    plt.text(0.5,0,'loss=%s'%(loss.item()),fontdict={'size':20,'color':'red'})
    plt.pause(0.005)

#
start = perf_counter()
LR_model,loss = train(LR_model,criterion,optimizer,10000)
finish = perf_counter()
time = finish - start
print("计算时间：%s" % time)
print("final loss:",loss.item())
print("weights:",list(LR_model.parameters()))