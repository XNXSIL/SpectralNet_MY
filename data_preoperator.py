#测试数据使用的是pytorchvision的数字图片
#对数字图片进行预处理。得到相似度矩阵存入数据data文件夹中

import torchvision.datasets as normal_datasets
import torchvision.transforms as transforms
import torch
from torch.autograd import Variable

#下载数据
def downloadData(isTrain):
    if(isTrain):
        # 从torchvision.datasets中加载一些常用数据集
        dataset = normal_datasets.MNIST(
            root='./data/',  # 数据集保存路径
            train=True,  # 是否作为训练集
            transform=transforms.ToTensor(),  # 数据如何处理, 可以自己自定义
            download=True)  # 路径下没有的话, 可以下载
    else:
        # 见数据加载器和batch
        dataset = normal_datasets.MNIST(root='./data/',
                                            train=False,
                                            transform=transforms.ToTensor())        
    return dataset

#加载数据
def loadData(dataset, batch_size, shuffle):
    loader = torch.utils.data.DataLoader(dataset=dataset,
                                            batch_size=batch_size,
                                            shuffle=shuffle)
    return loader

# 将数据处理成Variable, 如果有GPU, 可以转成cuda形式
def get_variable(x):
    x = Variable(x)
    return x.cuda() if torch.cuda.is_available() else x

#计算度矩阵
def get_DMatrix(data):
    pass

#计算邻接矩阵
def get_WMatrix(data):
    pass

#计算拉普拉斯矩阵
def get_LMatrix(data):
    pass

#通过相似度矩阵C计算M矩阵
def get_MMatirx(C, parm):
    D = torch.diag(C)
    return D - C + parm * D