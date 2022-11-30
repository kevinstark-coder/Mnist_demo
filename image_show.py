from PIL import Image
import os
import numpy as np 
import matplotlib.pyplot as plt 

def show_image_example():
    img = Image.open("./MNIST/toy_train/0.jpg")
    plt.imshow(img)
    plt.show()

def get_image_mean_std():
    data_dir = "./MNIST/toy_train/"
    filenames = os.listdir(data_dir)
    data_list  = [] 
    for filename in filenames:
        img = Image.open(data_dir + '/' + filename)
        data_list.append(np.asarray(img))
    data_arr = np.asarray(data_list)
    print(data_arr.shape)
    print("mean:", data_arr.mean())
    print("std:", data_arr.std())

if __name__ == "__main__":
    show_image_example()
    get_image_mean_std()

