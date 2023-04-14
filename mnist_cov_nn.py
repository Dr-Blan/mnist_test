# 使用卷积神经网络对 mnist 数据集进行训练和测试

import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import time

# Prepare dataset
train_dataset = datasets.MNIST(root='./dataset', train=True, transform=transforms.ToTensor(), download=True)  # 本地没有就加上download=True
test_dataset = datasets.MNIST(root='./dataset', train=False, transform=transforms.ToTensor(), download=True)  # train=True训练集，=False测试集
# 训练集乱序，测试集有序
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# Design model using class ------------------------------------------------------------------------------
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 10, kernel_size=5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(10, 20, kernel_size=5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(320, 50),
            torch.nn.Linear(50, 10),
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.conv1(x)  # 一层卷积层,一层池化层,一层激活层(图是先卷积后激活再池化，差别不大)
        x = self.conv2(x)  # 再来一次
        x = x.view(batch_size, -1)  # flatten 变成全连接网络需要的输入 (batch, 20,4,4) ==> (batch,320), -1 此处自动算出的是320
        x = self.fc(x)

        return x  # 最后输出的是维度为10的，也就是（对应数学符号的0~9）




# Super parameter
batch_size = 64
learning_rate = 0.01
momentum = 0.5
EPOCH = 5


# 实例化网络模型
model = Net()


# Construct loss and optimizer
criterion = torch.nn.CrossEntropyLoss()  # 交叉熵损失
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)  # lr学习率，momentum冲量



# Train and Test CLASS --------------------------------------------------------------------------------------
# 把单独的一轮一环封装在函数类里
def train():
    for inputs, labels in train_loader:
        # forward， 等价于 outputs = model.forward(images)
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # backward + update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# 测试函数
def test():
    correct = 0
    total = 0
    with torch.no_grad():  # 测试集不用算梯度
        for images, labels in test_loader:
            # 前向输出，等价于 outputs = model.forward(images)
            outputs = model(images)

            # dim = 1 列是第0个维度，行是第1个维度，沿着行(第1个维度)去找1.最大值和2.最大值的下标
            _, predicted = torch.max(outputs.data, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.shape[0]

    acc = correct / total

    return acc




# Start train and Test --------------------------------------------------------------------------------------

for epoch in range(EPOCH):
    start_time = time.time()
    train()
    print('The time cost of epoch %d is: %f' % (epoch+1, time.time()-start_time))
    acc_test = test()
    print('The accuracy of after %d epochs is: %f' % (epoch+1, acc_test))



