#测试数据使用的是pytorchvision的数字图片
#对数字图片进行预处理。得到相似度矩阵存入数据data文件夹中

import torchvision.datasets as normal_datasets
import torchvision.transforms as transforms
import torchvision
import torch
from torch.autograd import Variable
from sklearn.cluster import KMeans
import numpy
import pandas
import matplotlib
from matplotlib import pyplot as plt
import scipy
from scipy import io
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
def get_MMatirx(C, parm, D=None):
    if D is None:
        D = torch.triu(C) - torch.triu(C, diagonal=1)
    return D - C + parm * D

def kmeans(k,x):
    clf = KMeans(n_clusters=k)
    s = clf.fit(x)
    return clf.cluster_centers_,clf.Labels_

#初始化Y0矩阵
def get_Y0Matrix(method, k, isDiscrete, data):
    if type(data) == torch.Tensor:
        input = data.numpy()
    if type(data) == torch.autograd.Variable:
        input = data.data.numpy()

    if method == "kmeans":
        cluster_center,labels = kmeans(k, input)
    n = input.shape[0]
    result = numpy.zeros(n,k)
    if isDiscrete:
        for i in range(n):
            for j in range(k):
                result[i][j] = numpy.linalg.norm(input[i,:] - cluster_center[j,:])
    else:
        for i in range(n):
            result[i][labels[i]-1] = 1

    #归一化
    return torch.from_numpy(result)

#读取csv文件
def readCSV(filename, header):
    csvData = pandas.read_csv(filename,header=header)
    
    return (csvData.shape[0],csvData.shape[1],torch.FloatTensor(csvData.values))

#保存数据为csv文件
def saveCSV(filename,data):
    df = pandas.DataFrame(data=data)
    df.to_csv(filename, encoding="utf-8-sig", mode="a", header=False, index=False)

#读取mat文件
def readMat(filename):
    features_struct = scipy.io.loadmat(filename)  # 用来读取mat文件
    name = features_struct['name']
    features = features_struct['X']
    A = features_struct['A']
    D = features_struct['D']
    C = features_struct['c']
    init_Y = features_struct['init_Y']
    labels = features_struct['y0']

    return A,D,C,init_Y,labels

#评判相似度矩阵C
def calCLoss(C):
    Ct = torch.transpose(C, 1, 0)
    loss = torch.norm(C - Ct)

    print(C)
    return loss

#调整相似度矩阵C
def adjustMatrixC(C):
    #diag(C) = 1
    out = torch.abs( (torch.transpose(C, 1, 0) + C) / 2 )
    for i in range(out.shape[0]):
        out[i][i] = 1
    return out

#输出图片
def show_img(deco_images,batch_size):
    plt.figure()
    for i in range(1, batch_size):
        plt.subplot(10,10, i)
        plt.imshow(deco_images.detach()[i,0,:,:].numpy(), cmap='gray')
    plt.show()

def calAccRate(real,predict):
    succNum = 0
    totalNum = predict.shape[0]
    print(predict.shape)
    for i in range(totalNum):
        if real[i][0] == predict[i][0]:
            succNum += 1
    print(succNum)
    succRate = float(succNum/totalNum)
    return succRate


def graph(x,y,x_name,y_name,title):
    plt.title(title)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.plot(x,y)
    plt.show()





