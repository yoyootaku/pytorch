from torchvision import datasets, transforms
from torch import nn,optim
import torch.nn.functional as F
import torch.utils.data
#下载MNIST数据集
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307,),(0.3081,))])

trainset =  datasets.MNIST('data',train=True,download=True,transform=transform)
testset =  datasets.MNIST('data',train=False,download=True,transform=transform)



class LeNet(nn.Module):
    def __init__(self):
        super(LeNet,self).__init__()
        self.c1 = nn.Conv2d(1,6,5) #构建卷积层，输入1张灰度图，输出6张特征图，卷积核过滤器5*5
        self.c3 = nn.Conv2d(6,16,5) #构建卷积层，输入6张特征图，输出16张特征图，卷积核过滤器5*5
        self.fc1 = nn.Linear(16*4*4,120) #全连接层fc1
        self.fc2 = nn.Linear(120,84) #全连接层fc2
        self.fc3 = nn.Linear(84,10) #全连接层fc3
    def forward(self,x):
        x = F.max_pool2d(F.relu(self.c1(x)),2) #C1卷积后，relu()增加非线性拟合能力，再进行池化核为2的池化（下采样）
        x = F.max_pool2d(F.relu(self.c3(x)),2) #C3卷积后，relu()增加非线性拟合能力，再进行池化核为2的池化（下采样）
        x = x.view(-1,self.num_flat_features(x)) #计算特征点总数 16*4*4
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

lenet = LeNet()

criterion = nn.CrossEntropyLoss()#损失函数交叉熵
optimizer = optim.SGD(lenet.parameters(),lr=0.001,momentum=0.9)#优化器随机梯度下降

#batch_size一次性加载数据量，shuffle遍历数据打乱，num_workers两个子进程加载
trainloader = torch.utils.data.DataLoader(trainset,batch_size=4,shuffle=True,num_workers=2)

def train(model,criterion,optimizer,epochs=1):
    for epoch in range(epochs):
        running_loss = 0.0
        for i,data in enumerate(trainloader,0):#从第0项开始对trainloader数据进行枚举，data包含训练数据和标签
            inputs,labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()#权值更新

        running_loss += loss.item()
        if i%1000==999: #每训练1000次输出loss均值
            print('[Epoch:%d,Batch:%5d] Loss: %.3f' % (epoch+1,i+1,running_loss/1000))
            running_loss = 0.0

    print('Finished Training')

train(lenet,criterion,optimizer,epochs=2)