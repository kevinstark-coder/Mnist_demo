""""
The first main function is used for image resize
The second main function is for the MNIST test and prediction
"""
# # -*- coding: utf-8 -*-
# from PIL import Image
# import numpy as np
# import torch
# import matplotlib.pyplot as plt
# #from train import Net
# import cv2.cv2 as cv
# import cv2
#
# if __name__=="__main__":
#     # sample= np.zeros((3,4096, 3072))
#     for i in range(3):
#         image = cv2.imread("test/{}.jpg".format(i+1), cv2.IMREAD_GRAYSCALE)
#         print(type(image))
#         #sample = np.asarray(Image.open("test/{}.jpg".format(i+1)))
#         img_test = cv.resize(image, (16,16))
#         img_save = Image.fromarray(img_test)
#         img_save.save("test_{}.jpg". format(i))



#
# import torch
# import torchvision
# import torch.nn as nn
# import numpy as np
# from train import Net, MNISTDataset
# from train import get_loss_and_accuracy
# from PIL import Image
#
# def mnist_test():
#     img= np.asarray(Image.open("test_2.jpg").convert("L"))
#
#     print(img)
#     print(img.shape)
#     for i in range(16):
#         for j in range(16):
#             img[i][j] = 255-img[i][j]
#     print(img)
#     img_save = Image.fromarray(img)
#     img_save.save("2_phase.jpg")
#
#
#     net = Net()
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     print('device', device)
#     net.to(device)
#     ckpt_name  = "./model/epoch_20.pt"
#     state_dict = torch.load(ckpt_name, map_location = device)
#     net.load_state_dict(state_dict)
#     img=torch.FloatTensor(img)
#     img=torch.unsqueeze(img,0).to(device)
#     output=net(img)
#     net.eval()
#     print(output)
#     pred = torch.argmax(output, dim=1)
#     print(pred)
#
#
# if __name__ == "__main__":
#     mnist_test()