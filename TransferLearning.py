import torch
from torchvision import datasets,transforms
import matplotlib.pyplot as plt
import os
import torch.utils.data as Data
from torchvision import models
from torch import nn,optim



if __name__ == '__main__': #虽然是自己模块运行，但增加该语句后，可运行多线程

    data_transforms = {
        'train':transforms.Compose([
            transforms.Resize(230),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
        ]),
        'test':transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
        ]),
    }

    data_directory = 'data'

    trainset = datasets.ImageFolder(os.path.join(data_directory,'train'),data_transforms['train'])
    testset  = datasets.ImageFolder(os.path.join(data_directory,'test'),data_transforms['test'])



    trainloader = Data.DataLoader(dataset=trainset,batch_size=5,shuffle=True,num_workers=4)
    testloader  = Data.DataLoader(dataset=testset,batch_size=5,shuffle=True,num_workers=4)

    def imshow(inputs):
        inputs = inputs / 2 + 0.5
        inputs = inputs.numpy().transpose((1,2,0))
        plt.imshow(inputs)
        plt.show()

    inputs,classes = next(iter(trainloader))
    #imshow(torchvision.utils.make_grid(inputs))

    alexnet = models.alexnet(weights=models.AlexNet_Weights.DEFAULT) #0.15版本后适用weights来区分预训练模型的版本
    #print(alexnet)
    '''
    AlexNet(
        (features): Sequential( #卷积层 负责特征提取
        (0): Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
    (1): ReLU(inplace=True)
    (2): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
    (3): Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    (4): ReLU(inplace=True)
    (5): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
    (6): Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (7): ReLU(inplace=True)
    (8): Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (9): ReLU(inplace=True)
    (10): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (11): ReLU(inplace=True)
    (12): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
    )
    (avgpool): AdaptiveAvgPool2d(output_size=(6, 6))
    (classifier): Sequential(  #全连接层，负责分类
        (0): Dropout(p=0.5, inplace=False)
    (1): Linear(in_features=9216, out_features=4096, bias=True)
    (2): ReLU(inplace=True)
    (3): Dropout(p=0.5, inplace=False)
    (4): Linear(in_features=4096, out_features=4096, bias=True)
    (5): ReLU(inplace=True)
    (6): Linear(in_features=4096, out_features=1000, bias=True)
    )
    )
    '''
    for param in alexnet.parameters():
        param.requires_grad = False
    alexnet.classifier=nn.Sequential( #只更新分类（全连接层参数
        nn.Dropout(), #前向传播过程中随机丢弃一些神经网络层节点，有效避免模型过拟合，缺省0.5
        nn.Linear(256*6*6,4096),
        nn.ReLU(inplace=True),
        nn.Dropout(),#前向传播过程中随机丢弃一些神经网络层节点，有效避免模型过拟合，缺省0.5
        nn.Linear(4096,4096),
        nn.ReLU(inplace=True),
        nn.Linear(4096,2),)

    criterion =  nn.CrossEntropyLoss()
    optimizer = optim.SGD(alexnet.classifier.parameters(),lr=0.001,momentum=0.9) #lr 学习率：随机梯度下降的速率

    def train(model,criterion,optimizer,epochs=1):
        for epoch in range(epochs):
            running_loss = 0.0
            for i,data in enumerate(trainloader,0):
                inputs,labels = data
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs,labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if i%10==9:
                    print('[Epoch:%d,Batch:%5d] loss:%.3f' % (epoch+1,i+1,running_loss/100))
                    running_loss = 0.0

        print('Finished Training')

    def test(testloader,model):
        correct = 0
        total = 0
        for data in testloader:
            images,labels = data
            outputs = model(images)
            _,predicted = torch.max(outputs.data,1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
        print('Accuracy on the test set: %d %%' % (100 * correct / total))

    #存储与加载
    def load_param(model,path):
        if os.path.exists(path):
            model.load_state_dict(torch.load(path))
    def save_param(model,path):
        torch.save(model.state_dict(),path)

    #训练前加载，训练完后保存参数并测试
    load_param(alexnet,'tl_model.pkl')
    train(alexnet,criterion,optimizer,epochs=2)
    save_param(alexnet,'tl_model.pkl')
    test(testloader,alexnet)