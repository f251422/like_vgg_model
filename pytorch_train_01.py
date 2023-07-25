import os
import torch
import torchvision as tv
import tensorflow as tf
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
show = ToPILImage() # 可以把Tensor转成Image，方便可视化
import time
import multiprocessing


# 调用训练的设备
#device = torch.device("cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu" )
# 定义对数据的预处理
transform = transforms.Compose([
        transforms.ToTensor(), # 转为Tensor
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # 归一化 先将输入归一化到(0,1)，再使用公式”(x-mean)/std”，将每个元素分布到(-1,1)
                             ])
# 获取当前脚本的运行路径
data_root = os.path.abspath(os.getcwd())

# 获取数据集的路径
image_path = os.path.join(data_root, 'image/')
#print(image_path)
batch_size=100
assert os.path.exists(image_path), '{} path does not exist.'.format(image_path)
# 加载训练数据集
# ImageFolder函数：假设所有文件按文件夹保存，每个文件夹下存储同一类别的图像，文件夹名为类别
# 参数含义：root：在指定路径下寻找图像，transform:对PILImage进行的转换操作，输入是使用loader读取的图片
trainset = datasets.ImageFolder(root=os.path.join(image_path, 'train'),transform=transform)
# 训练集（因为torchvision中已经封装好了一些常用的数据集，包括CIFAR10、MNIST等，所以此处可以这么写 tv.datasets.CIFAR10()）# 将下载的数据集压缩包解压到当前目录的DataSet目录下
#trainset = tv.datasets.CIFAR10(root='DataSet/',   train=True,download=False,transform=transform)
# 测试集
#testset = tv.datasets.CIFAR10('DataSet/',train=False, download=False, transform=transform)


# 加载测试训练集
testset = datasets.ImageFolder(root=os.path.join(image_path, 'val'),transform=transform)

#print("using {} images for training, {} images for validation.".format(train_num, val_num))

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
# 准备数据集(未下载的数据集)
#train_data = torchvision.datasets.CIFAR10("./dataset", train=True, transform=torchvision.transforms.ToTensor(),download=True)
#test_data = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor(),download=True)


# 数据集的 length 长度
train_data_size = len(trainset)
test_data_size = len(testset)
print("训练数据集的长度为：{}".format(train_data_size))
print("测试数据集的长度为：{}".format(test_data_size))

# 利用 DataLoader 来加载数据集
trainloader = torch.utils.data.DataLoader(
                    trainset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=0)
testloader = torch.utils.data.DataLoader(
                    testset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=0)
#dataloder_train = DataLoader(trainset, batch_size=64, drop_last=False)
#dataloder_test = DataLoader(testset, batch_size=64, drop_last=False)

if __name__ == '__main__':
    multiprocessing.freeze_support()
    # 在主模块中的其他代码
# 搭建神经网络
class Test(nn.Module):
    def __init__(self):
        super(Test, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(in_features=2048, out_features=1024),
            nn.Linear(in_features=1024, out_features=10)
        )

    def forward(self, x):
        x = self.model(x)
        return x


# 创建网络模型
test = Test()

test = test.to(device)  # 方法二
# test.to(device)

# 损失函数
loss_fc = nn.CrossEntropyLoss()

loss_fc = loss_fc.to(device)  # 方法二
# loss_fc.to(device)

# 优化器
learning_rate = 0.02  # 1e-2 = 1 * 10^(-2) = 1/100 = 0.01
optimizer = torch.optim.SGD(test.parameters(), lr=learning_rate)

# 设置训练网络的一些参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练的轮数
epoch =60

# 添加 tensorboard
writer = SummaryWriter("./newlogs")

# 获取开始的时间
start_time = time.time()

for i in range(epoch):
    print("----------第{}轮训练开始：-------------".format(i + 1))

    # 训练步骤开始
    test.train()  # 只对一些特殊的层起作用，代码中没有这些层也可以写上
    total_correct=0#添加变量
    for data in trainloader:
        imgs, targets = data
        imgs = imgs.to(device)  # 方法二
        targets = targets.to(device)
        output_train = test(imgs)
        loss = loss_fc(output_train, targets)  # 获得损失值 loss

        _,predicted=torch.max(output_train.data,1)#计算训练精度
        total_correct+=(predicted==targets).sum().item()

        # 使用优化器优化模型
        optimizer.zero_grad()  # 梯度清零
        loss.backward()  # 调用损失 loss，得到每一个参数的梯度
        optimizer.step()  # 调用优化器 optimizer 对我们的参数进行优化

        total_train_step = total_train_step + 1  # 记录训练次数
        if total_train_step % 100 == 0:
           end_time = time.time()  # 每运行100次结束的时间
           print("{}秒".format(end_time - start_time))  # 模型运行需要多长时间
           print("训练次数：{}， 损失值：{}".format(total_train_step, loss.item()))
           # loss.item()与loss时有区别的，loss.item()返回的是数字
           train_accuracy=total_correct/len(trainset)
           print("训练次数:{}, 训练精度:{:.2f}".format(total_train_step,train_accuracy*100))
           writer.add_scalar("train_loss", loss.item(), total_train_step)  # 逢100的整数记录
           writer.add_scalar("train_accuracy", train_accuracy, total_train_step)
           # 测试步骤开始
    test.eval()  # 只对一些特殊的层起作用，代码中没有这些层也可以写上
    total_test_loss = 0
    # 正确率
    total_accuracy = 0
    with torch.no_grad():  # 表示在 with 里的代码，它的梯度就没有了，保证不会进行调优
        for data in testloader:
            imgs, targets = data
            imgs = imgs.to(device)  # 方法二
            targets = targets.to(device)
            output_test = test(imgs)
            loss = loss_fc(output_test, targets)

            total_test_loss = total_test_loss + loss

            accuracy = (output_test.argmax(1) == targets).sum()  # 计算预测与实际 一致的个数
            total_accuracy = total_accuracy + accuracy  # 总的正确的个数

    print("整体测试集的损失值：{}".format(total_test_loss.item()))
    print("整体测试的正确率为：{}".format(total_accuracy / test_data_size))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy / test_data_size, total_test_step)
    total_test_step = total_test_step + 1
    # 保存每一轮训练的模型
    torch.save(test, "test_{}.pth".format(50))  # 方法以保存
    # 下面是方法二保存模型，将参数保存成字典型
    # torch.save(test.state_dict(), "test_{}".format(i))
    print("模型已保存")

writer.close()



