import torch
import matplotlib.pyplot as plt
import numpy

#生成输入矩阵X，第二列（参数）为1
def Produce_X(x):
    x0 = torch.ones(x.numpy().size)
    X = torch.stack((x,x0),dim=1)
    return X

x=torch.Tensor([1.4,5,11,16,21])
y=torch.Tensor([14.4,29.6,62,85.5,113.4])
X = Produce_X(x)

inputs = X
target = y

#生成随机参数向量
w = torch.rand(2,requires_grad=True)

#寻找回归完成时的遍历次数
nCount = 0

#定义训练函数train（），不断调整w的值
#epochs代表遍历次数，learning_rate为学习率（梯度下降的速率）
def train(epochs=1,learing_rate=0.01):
    markloss = torch.Tensor([1000000])
    for epoch in range(epochs):
        output = inputs.mv(w)
        loss = (output - target).pow(2).sum() #前向传播计算损失函数

        if torch.gt(markloss,loss):
           markloss = loss
           nCount = epoch+1

        loss.backward()#损失函数计算关于w的梯度向量（微分）
        w.data -= learing_rate * w.grad #反向传播
        w.grad.zero_()
        if epoch % 800 == 0:
            draw(output,loss)
    return w,loss,nCount,markloss

#绘图
def draw(output,loss):
    plt.cla()
    plt.scatter(x.numpy(),y.numpy())#绘制散点图
    plt.plot(x.numpy(),output.data.numpy(),'r-',lw=5)
    plt.text(0.5,0,'loss=%s'%(loss.item()),fontdict={'size':20,'color':'red'})
    plt.pause(0.005)

#设置循环次数和学习率
w,loss,nCount,markloss = train(epochs=30000,learing_rate=1e-4)

print("final loss:",loss.item())
print("markLoss:",markloss.item())
print("有效回归次数：",nCount)
print("weights",w.data)

