import torch
import torch.nn as nn
from data_preoperator import *
from layer import *
from loss import *
import matplotlib
from matplotlib import pyplot as plt 
import numpy 

matplotlib.use('tkagg')

# init parameters
num_epochs = 1500
batch_size = 165
learning_rate = 0.001
weight_decay = 1e-5
alpha = 0.1
beta = 0.01
layers = 7
labSize = 10

def TrainAutoEncoderLayer():
    train_dataset = downloadData(True)
    train_loader = loadData(train_dataset, batch_size, True)

    module = Autoencoder()
    
    # 选择损失函数和优化方法
    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam(module.parameters(), lr=learning_rate, weight_decay=weight_decay)

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = get_variable(images)
            labels = get_variable(labels)
    
            _,outputs = module(images)

            loss = loss_func(outputs, images)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            if (i + 1) % 100 == 0:
                print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f'
                    % (epoch + 1, num_epochs, i + 1, len(train_dataset) // batch_size, loss.item()))
    
    
    # Save the Trained Model
    torch.save(module.state_dict(), './module/AutoEncoder.pkl')
    return module

def TrainSelfExpressionLayer(auto_images):

    batch_size = 100
    module = FCLayer(1, batch_size, batch_size)

    loss_func = selfExpressionLoss
    optimizer = torch.optim.Adam(module.parameters(), lr=learning_rate, weight_decay=weight_decay)

    Z = auto_images.view(auto_images.size(0), -1)
    epochs = 6000
    # x = numpy.arange(1,epochs-1999)
    # y = numpy.ones(epochs-2000)
    for epoch in range(epochs):
        C, ZC = module(Z)
        e = 0.1
        print(calCLoss(C))
        loss = loss_func(e, Z, C, ZC)
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        # if epoch >= 2000:
        #     y[epoch-2000] = loss.item()
        print('selfExpression Epoch [%d/%d],Loss: %.4f'
            % (epoch + 1, epochs, loss.item()))

    
    # Save the Trained Model
    # torch.save(module.state_dict(), './module/SelfExpression.pkl')
    #print graph
    # plt.title("epoch_loss")
    # plt.xlabel("epoch")
    # plt.ylabel("loss")
    # plt.plot(x,y)
    # plt.show()
    return module

def TrainCSVData(Autoencoder, batch_size, labSize, Y0, M, labels):
    # (Yn,Yc,Y0) = readCSV("./data/Yale_32x32_PKN_Ncut_Y_0K.csv")
    # (Md,Mw,M) = readCSV("./data/init_PKN_M.csv")

    num_epochs = 500
    x = numpy.arange(1, layers+1)
    y = numpy.zeros(layers)
    L0 = torch.zeros(batch_size, labSize, dtype=torch.float32)

    module = list()
    optimizer = list()

    for layer in range(layers):
        module.append(SpectralNetLayer(batch_size,L0,alpha,beta,layer+1,labSize))
        module[layer].changeYZ0(Y0)
        optimizer.append(torch.optim.Adam(module[layer].parameters(), lr=learning_rate, weight_decay=weight_decay))
   
    for layer in range(layers):

        lastLoss = 0
        outputs = 0
        for i in range(num_epochs):
            Y,Z = module[layer](M)
            Yk = Y[layer]
            Zk = Z[layer]
            Yt = torch.transpose(Yk, 1, 0)
            Zt = torch.transpose(Zk, 1, 0)

            lossY = torch.trace(Yt.mm(M).mm(Yk))
            lossZ = torch.trace(Zt.mm(M).mm(Zk))
            
            lastLoss = 1/2 * lossY + 1/2 * lossZ
            _,outputs = torch.max(Yk, 1)
            outputs = outputs + 1
            optimizer[layer].zero_grad()
            lastLoss.backward()
            optimizer[layer].step()
            
            if i % 100 == 0:
                print("layer %d epoch [%d/%d] loss %.4f" % (layer, i, num_epochs,lastLoss.item()))
        y[layer] = lastLoss
        
        saveCSV("./tempData/Yk_"+str(layer+1)+".csv",numpy.array(outputs.numpy()))

    plt.title("layer_loss")
    plt.xlabel("layer")
    plt.ylabel("loss")
    plt.plot(x,y)
    plt.show()


def TrainNumberData(Autoencoder):
    batch_size = 100
    labSize = 10
    train_dataset = downloadData(True)
    train_loader = loadData(train_dataset, batch_size, True)

    L0 = torch.zeros(batch_size, labSize, dtype=torch.float32)

    module = SpectralNetLayer(batch_size,L0,alpha,beta,layers,labSize)

    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam(module.parameters(), lr=learning_rate, weight_decay=weight_decay)

    #****************************手写数字训练****************************************************************#
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = get_variable(images)
            labels = get_variable(labels)

            #通过CNN提取特征值
            auto_images,deco_images = Autoencoder(images)

            # show_img(deco_images, batch_size)
            
            #通过自我表达层获取相似度矩阵C，再计算M矩阵
            selfExpressionLayer = TrainSelfExpressionLayer(auto_images)
            C  = adjustMatrixC(selfExpressionLayer.fc[0].weight)
            saveCSV("./tempData/data.csv",C.detach().numpy())
            print(calCLoss(C))
            eta = 0.01
            M = get_MMatirx(C= C, eta=eta)

            #通过（kmeans/***）方法计算聚类中心，初始化Y0以及Z0
            isDiscrete = False   #是否初始化Y0为离散值
            Y0 = get_Y0Matrix("kmeans", labSize, isDiscrete, auto_images)

            module.changeYZ0(Y0)

            
            Y,Z = module(M)
            Yk = Y[layers-1]
            Zk = Z[layers-1]

            Yt = torch.transpose(Yk, 1, 0)
            Zt = torch.transpose(Zk, 1, 0)

            lossY = torch.trace(Yt.mm(M).mm(Yk))
            lossZ = torch.trace(Zt.mm(M).mm(Zk))
            
            loss = 1/2 * lossY + 1/2 * lossZ

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


if __name__ == "__main__":
    # TrainAutoEncoderLayer()
    AutoEncoderLayer = Autoencoder()
    AutoEncoderLayer.load_state_dict(torch.load("./module/AutoEncoder.pkl"))

    # TrainCSVData(AutoEncoderLayer)
    # TrainNumberData(AutoEncoderLayer)

    #***************文本数据训练***************************
    features,A,D,C,init_Y,labels = readMat("./data/data/Isolet_7797.mat")
    print(type(features))
    # print(C[0][0])

    eta = 0.01
    M = get_MMatirx(C=torch.FloatTensor(A), parm=eta, D=torch.FloatTensor(D))

    #离散初始化Y0
    # Y0 = torch.FloatTensor(init_Y)


    #连续初始化Y0
    Y0 = get_Y0Matrix("kmeans", C[0][0], True, features)


    lables = torch.FloatTensor(labels)
    TrainCSVData(AutoEncoderLayer, init_Y.shape[0], init_Y.shape[1], Y0, M, labels)



    #*****************************画图*****************

    # x = numpy.arange(1,6)
    # y = numpy.array([0.03091,0.02988,0.03001,0.05592,0.03912])
    # graph(x,y,"layer","ACC","ACC_layer")



    