#mnist_train.py  1.0

import torch
from torch import nn
from torch.nn import functional as F
from torch import optim

import torchvision

from utils import plot_curve, plot_image, one_hot


batch_size = 512
# step1.load_dataset
# 'mnist_data'：加载mnist数据集，路径
# train=True：选择训练集还是测试
# download=True：如果当前文件没有mnist文件就会自动从网上去下载
# torchvision.transforms.ToTensor():下载好的数据一般是numpy格式，转换成Tensor
# torchvision.transforms.Normalisze((0.1307,), (0.3081,))：正则化过程，为了让数据更好的在0的附近均匀的分布
# 上面一行可注释掉：但是性能会差到百分之70，加上是百分之80，更加方便神经网络去优化

train_loader = torch.utils.data.DataLoader(  # 加载训练集
    torchvision.datasets.MNIST('data', train=True, download=True,
        transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),#numpy转化为tensor
        torchvision.transforms.Normalize((0.1307,), (0.3081,))  # 均值是0.1307，标准差是0.3081，这些系数都是数据集提供方计算好的数据
                                                ])),
    batch_size=batch_size, shuffle=True)
# batch_size=batch_size：表示一次加载多少张图片
# shuffle = True 加载的时候做一个随机的打散

test_loader = torch.utils.data.DataLoader(  # 加载测试集
    torchvision.datasets.MNIST('data/', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size, shuffle=False)  # 测试集不用打散

x, y = next(iter(train_loader))
print(x.shape, y.shape)
# print(x.min(), x.max())
# plot_image(x, y, 'image_sample')
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 256)
        self.fc2 = nn.Linear(256,64)
        self.fc3 = nn.Linear(64,10)
    def forward(self, x):
        # x = [b,1,28,28]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()
optimizer = torch.optim.SGD(net.parameters(), lr=1e-2, momentum = 0.9)
train_loss = []
for epoch in range(3):
    for batch_dix, (x, y) in enumerate(train_loader):
            # x:[b,1,28,28] -> [b,784]
        x = x.view(x.size(0),28*28)
        out = net(x)
        y_onehot = one_hot(y)
        loss = F.mse_loss(out, y_onehot)

        optimizer.zero_grad()#清零梯度
        loss.backward()#得到梯度
        optimizer.step()#更新梯度

        train_loss.append(loss.item())

        if batch_dix % 10 == 0:
            print(epoch, batch_dix, loss.item())

plot_curve(train_loss)


def acc(test_loader,model):
    total_corret = 0
    for x, y in test_loader:
        x = x.view(x.size(0), 28*28)
        out = model(x)
        pred = out.argmax(dim=1)
        correct = pred.eq(y).sum().float().item()
        total_corret += correct

    total_num = len(test_loader.dataset)
    acc = total_corret /total_num
    return acc


print('test acc:',acc(test_loader, net))
x,y =next(iter(test_loader))
out = net(x.view(x.size(0),28*28))
pred = out.argmax(dim=1)
plot_image(x, pred,'test')
