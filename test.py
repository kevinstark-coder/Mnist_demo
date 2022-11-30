# -*- coding: utf-8 -*-

import torch
import torchvision
import torch.nn as nn
import numpy as np
from train import Net, MNISTDataset
from train import get_loss_and_accuracy

def mnist_test():
    batch_size = 8
    testset = MNISTDataset(csv_file = "./MNIST/torch_test.csv",
        root_dir = './MNIST/toy_test', transform = None)
    testloader = torch.utils.data.DataLoader(testset, 
        batch_size=batch_size, shuffle=True, num_workers=1)
    net = Net()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('device', device)
    net.to(device)
    ckpt_name  = "./model/epoch_20.pt"
    state_dict = torch.load(ckpt_name, map_location = device)
    net.load_state_dict(state_dict)
    net.eval()

    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = get_loss_and_accuracy(net, testloader, criterion, device)
    print("Accuracy on the test dataset", test_acc)

if __name__ == "__main__":
    mnist_test()