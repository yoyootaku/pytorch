import torch
import matplotlib.pyplot as plt
from torch import nn,optim
import torch.nn.functional as F #激活函数集合
from time import perf_counter

x = torch.unsqueeze(torch.linspace(-3,3,10000),dim=1)
y = x.pow(3) + 0.3 * torch.rand(x.size())

class Net(nn.Module):
    def __init__(self,input_feature,num_hidden,outputs):
        super(Net,self).__init__()
        self.hidden = nn.Linear(input_feature,num_hidden)
        self.out = nn.Linear(num_hidden,outputs)

    def forward(self,x):
        x = F.relu(self.hidden(x)) #经过隐含层后需要再经过ReLU激活函数进行非线性处理
        x = self.out(x) #经过输出层
        return x

#设置输入为1维，隐含层节点数20，输出为1维
net = Net(input_feature=1,num_hidden=20,outputs=1)
inputs = x
target = y

#随机梯度下降
optimizer = optim.SGD(net.parameters(),lr=0.01)
#损失函数均方误差
criterion = nn.MSELoss()

#训练函数 参数：model：被训练的神经网络模型、criterion:损失函数、optimizer:优化器、epochs:训练轮数
def train(model,criterion,optimizer,epochs):
    for epoch in range(epochs):
        output = model(inputs)#前向传播
        loss = criterion(output,target)#计算损失值
        optimizer.zero_grad()#清空反向传播权重
        loss.backward()#反向传播计算梯度
        optimizer.step()#权值更新
        if epoch % 80 == 0:
            draw(output,loss)
    return model,loss

#绘图
def draw(output,loss):
    plt.cla()
    plt.scatter(x.numpy(),y.numpy(),s=0.1)#绘制散点图
    plt.plot(x.numpy(),output.data.numpy(),'r-',lw=5)
    plt.text(0.5,0,'loss=%s'%(loss.item()),fontdict={'size':20,'color':'red'})
    plt.pause(0.005)

start = perf_counter()
net,loss = train(net,criterion,optimizer,10000)
finish = perf_counter()
time = finish - start
print("计算时间：%s" % time)
print("final loss:",loss.item())
#print("weights:",list(net.parameters()))