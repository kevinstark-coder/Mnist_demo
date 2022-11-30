'''
Project name: Hand digit writing
Authur: Qiankun Yang
Date: 2022.7.26
--------------------------------
In this project, we will do following things in order:
1.Load the mnist data
2.define an CNN model
3.define the loss function and train it
4.run test on this model
'''
import torch.optim as optim
import numpy as np
import torch
import os
import torch
import random
import pandas as pd
import torch.nn as nn
from PIL import Image
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


def seed_torch(seed=1):
    """
    set seed of random, making the program repeatable.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)                                    #为CPU设置种子用于生成随机数，以使得结果是确定的
    # if you use GPU, the following lines are needed.
    torch.cuda.manual_seed(seed)                               #为当前GPU设置随机种子；
    torch.cuda.manual_seed_all(seed) # if using multi-GPU.     #如果使用多个GPU，应该使用torch.cuda.manual_seed_all()为所有的GPU设置种子。
    torch.backends.cudnn.benchmark = False                      #初始寻找最优卷积
    torch.backends.cudnn.deterministic = True                  #继承上次卷积算法



class MNIST(Dataset):
    def __init__(self, root_dir, csv_file, transform = None):       # pd.read 用于 csv文件
        self.csv_item = pd.read_csv(csv_file)                           # PIL.Image.open(path)用于图片，返回数组，
        self.root_dir = root_dir
        self.transform = transform

    def __getitem__(self, idx):
        img_name = self.csv_item.iloc[idx, 0]
        img_name = os.path.join(self.root_dir, img_name)
        img = np.asarray(Image.open(img_name), np.float32)
        img = (img - 36.13) / 69.38                                # 数据处理
        img = np.expand_dims(img, axis=0)
        lab = self.csv_item.iloc[idx, 1]
        sample = {"image": img, "label": lab}                         # 生成字典

        if self.transform is not None:                              # 两种写法
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.csv_item)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 5)        # 1*16*16---16*12*12
        self.pool = nn.MaxPool2d(2, 2)          # 16*12*12--16*6*6
        self.conv2 = nn.Conv2d(16, 32, 3)       # 16*6*6 --32*4*4
        self.l1 = nn.Linear(32*4*4, 128)
        self.l2 = nn.Linear(128, 64)
        self.l3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 32 *4 *4)
        x= F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x

def train_main():
    seed_torch(1)
    batch_size = 8
    Epoch = 10                                                                   #设置batchsize epoch，epoch,损失函数
    criterion = nn.CrossEntropyLoss()

    traindata = MNIST(root_dir="./MNIST/toy_train", csv_file="./MNIST/torch_train.csv", transform=None)                 #实例化后使用使用torch，utils.data.Dataloader
    trainloader = torch.utils.data.DataLoader(traindata, batch_size= batch_size, shuffle=True, num_workers=1)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")                        #设置设备以及网络在家进去
    print("device", device)
    net = Net()
    net.to(device)
    optimizer = optim.SGD(net.parameters(),lr=0.001,momentum=0.9)
    loss_list = []
    acc_list = []
    for epoch in range(Epoch):
        for i, data in enumerate(trainloader, 0):
            images, labels = data["image"],data["label"]
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()  #在计算梯度之前讲上一次的置于0
            outputs = net(images)
            pred_labels = torch.argmax(outputs, dim=1)
            correct_n = np.asarray(pred_labels == labels).sum()
            acc= (correct_n+0.0)/(labels.size(dim=0))
            acc_list.append(acc)
            loss = criterion(outputs, labels)
            print(outputs.shape)
            print(labels.shape)
            loss_list.append(loss.detach())
            loss.backward()
            optimizer.step()
        torch.save(net.state_dict(),"model/epoch_{}.pt".format(epoch+1))
        loss_mean = np.asarray(loss_list).mean()
        acc_mean = np.asarray(acc_list).mean()
        print("The %d th epoch is finished\n" % (epoch+1))
        print("The average loss of this epoch is (%.3f),The average acc of this epoch is (%.3f)" % (loss_mean, acc_mean))

    print("Finished training")

if __name__=="__main__":
    train_main()
