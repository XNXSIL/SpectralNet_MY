import torch
import torch.nn as nn
from data_preoperator import *
from layer import *
from loss import *
import matplotlib
import numpy 
import time
import sys
import psutil
import random
import torch.nn.functional as F
matplotlib.use('tkagg')
from matplotlib import pyplot as plt 
from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import cosine_similarity
# init parameters
num_epochs = 1500
batch_size = 165
learning_rate = 0.0001
weight_decay = 1e-5
alpha = 0.1
beta = 0.01
layers = 100
labSize = 10


def TrainCNN():
    batch_size = 100
    num_epochs = 40
    train_dataset = downloadData(True)
    train_loader = loadData(train_dataset, batch_size, True)
    print("data  ok")
    module = CNN2()
    
    # 选择损失函数和优化方法
    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam(module.parameters(), lr=learning_rate, weight_decay=weight_decay)

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = get_variable(images)
            labels = get_variable(labels)
            images = F.normalize(images)
            
            encode,decode = module(images)

            # loss = 1/(2*batch_size) * torch.norm((images-decode),2)
            loss = loss_func(images,decode)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            # if (i + 1) % 100 == 0:
            print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f'
                % (epoch + 1, num_epochs, i + 1, len(train_dataset) // batch_size, loss.item()))
    
    
    # Save the Trained Model
    torch.save(module.state_dict(), './module/CNN2.pkl')
    return module

def TrainSelfExpression(images):

    batch_size = images.shape[0]
    module = DeepAutoEnCoderSelfExpress(batch_size)
    alpha1 = 1
    alpha2 = 1.0 * 10 ** (10 / 10.0 - 3.0)
    loss_func = selfExpressionLoss
    optimizer = torch.optim.Adam(module.parameters(), lr=learning_rate, weight_decay=weight_decay)

    epochs = 1000
    x = numpy.arange(0,epochs)
    y = numpy.ones(epochs)
    for epoch in range(epochs):
        C,Z,ZC,encode,decode = module(images)
        e = 0.1

        loss = loss_func(alpha1,alpha2, Z, C, ZC, images ,decode)
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        y[epoch] = loss.item()
        if epoch >= 200:
            print('selfExpression Epoch [%d/%d],Loss: %.4f'
                % (epoch + 1, epochs, loss.item()))

    
    # Save the Trained Model
    # torch.save(module.state_dict(), './module/SelfExpression.pkl')
    #print graph
    # plt.title("epoch_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.plot(x,y)
    plt.show()
    return module


def TrainCSVData3(init_Y, X, A, D, labels):
    layer = 1
    batch_size = 7797
    labSize = 26


    A = torch.FloatTensor(A)
    D = torch.FloatTensor(D)
    X = torch.FloatTensor(X)
    labels = torch.FloatTensor(labels[:,0])
    Y0 = torch.FloatTensor(init_Y)
    num_epochs = 1000

    M = D - A


    # selfExpress = TrainSelfExpressionLayer(X, batch_size)
    # A = adjustMatrixC(selfExpress.fc[0].weight)
    print(X.shape)
    A = knn_affinity(X, 100)
    D = torch.triu(A) - torch.triu(A, diagonal=1)
    M = D - A

    loss1 = torch.trace(torch.transpose(Y0,1,0).mm(M).mm(Y0))
    print("initLoss is %.4f" %(loss1.item()))

    _,outputs = torch.max(Y0, 1)
    outputs = outputs + 1
    print_accuracy(outputs.numpy(), labels.numpy(), labSize)



    lossX = numpy.arange(0,num_epochs)
    lossY = numpy.zeros(num_epochs)

    accX = numpy.arange(0,num_epochs)
    accY = numpy.zeros(num_epochs)

    # for layer in range(layers):
    module = SpectralNetNorm(batch_size, labSize, layer)
    optimizer = torch.optim.Adam(module.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    for epoch in range(num_epochs):
        Yk = orthonorm(module(A, D))

        _,outputs = torch.max(Yk, 1)
        outputs = outputs + 1

        loss = torch.trace(torch.transpose(Yk,1,0).mm(M).mm(Yk))
        # for i in range(layer):
        #     totalLoss += loss[i]

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        accY[epoch] = print_accuracy(outputs.numpy(), labels.numpy(), labSize)
        lossY[epoch] = loss
        print("layer %d epoch [%d/%d] ACC %.4f loss %.4f" %(layer,epoch,num_epochs,accY[epoch],lossY[epoch]))


    plt.subplot(2,1,1)
    plt.xlabel("layer")
    plt.ylabel("loss")
    plt.plot(lossX,lossY)

    plt.subplot(2,1,2)
    plt.xlabel("layer")
    plt.ylabel("accY")
    plt.plot(accX,accY)


    plt.show()

#单纯ADMM。没有lossBackward
def TrainCSVData2(init_Y, A, D, labels):
    layer = 200
    batch_size = 7797
    labSize = 26


    C = torch.FloatTensor(A)
    D = torch.FloatTensor(D)
    labels = torch.FloatTensor(labels[:,0])
    Y0 = torch.FloatTensor(init_Y)
    L0 = torch.zeros(labSize, batch_size, dtype=torch.float32)


    _,outputs = torch.max(Y0, 1)
    outputs = outputs + 1
    print_accuracy(outputs.numpy(), labels.numpy(), labSize)

    lossfunc = torch.nn.CrossEntropyLoss
    module = SpectralNetLayer(batch_size,L0,alpha,beta,layer+1,labSize)

    module.changeYZ0(Y0)
    Y,Z,loss1,loss2,loss3 = module(C, D)


    loss1X = numpy.arange(0,layer)
    loss1Y = numpy.zeros(layer)
    loss2X = numpy.arange(0,layer)
    loss2Y = numpy.zeros(layer)
    loss3X = numpy.arange(0,layer)
    loss3Y = numpy.zeros(layer)

    accYX = numpy.arange(0,layer)
    accYY = numpy.zeros(layer)

    accZX = numpy.arange(0,layer)
    accZY = numpy.zeros(layer)

    for layers in range(layer):
        Yk = Y[layers]
        Zk = Z[layers]

        _,outputs = torch.max(Yk, 1)
        outputs = outputs + 1

        _,outputsZ = torch.max(Zk,1)
        outputsZ = outputsZ + 1

        # y_true,_ = get_y_preds(outputs.numpy(), labels.numpy(), labSize)
        # entropy = torch.zeros(Yk.shape[0],Yk.shape[1])
        # for i in range(y_true):
        #     entropy[i][y_true[i]] = 1

        #使用公式loss
        # lastLoss = torch.trace(Yt.mm(M).mm(Zk))

        #使用交叉熵loss
        # lastLoss = lossfunc(entropy,labels)
        # print(lastLoss.item())


        # print(batch_size)
        print("layer ", layers)

        accYY[layers] = print_accuracy(outputs.numpy(), labels.numpy(), labSize)
        accZY[layers] = print_accuracy(outputsZ.numpy(), labels.numpy(), labSize)
        loss1Y[layers] = loss1[layers].item()
        loss2Y[layers] = loss2[layers].item()
        loss3Y[layers] = loss3[layers].item()

    plt.subplot(3,1,1)
    plt.title("layer_loss")
    plt.xlabel("layer")
    plt.ylabel("loss")
    plt.plot(loss1X,loss1Y)
    plt.plot(loss2X,loss2Y)
    plt.plot(loss3X,loss3Y)

    plt.subplot(3,1,2)
    plt.title("layer_accY")
    plt.xlabel("layer")
    plt.ylabel("accY")
    plt.plot(accYX,accYY)

    plt.subplot(3,1,3)
    plt.title("layer_accZ")
    plt.xlabel("layer")
    plt.ylabel("accZ")
    plt.plot(accZX,accZY)

    plt.show()

def TrainCSVData(Autoencoder, input, batch_size, labSize):
    # (Yn,Yc,Y0) = readCSV("./data/Yale_32x32_PKN_Ncut_Y_0K.csv")
    # (Md,Mw,M) = readCSV("./data/init_PKN_M.csv")

    num_epochs = 500
    layers = 5000
    eta = 0.01
    x = numpy.arange(1, layers+1)
    y = numpy.zeros(layers)
    L0 = torch.zeros(batch_size, labSize, dtype=torch.float32)

    batch = 10
    numpy.random.shuffle(input)

    Y0 = list()
    C = list()
    D = list()
    l_ture = list()

    info = psutil.virtual_memory()
    print("memory use :", info.percent)
    start = time.clock()
    for i in range(1):
        data = torch.FloatTensor([])
        label = torch.FloatTensor([])
        for k in range(i*batch_size,(i+1)*batch_size):
            data = torch.cat((data,input[k][0]),0)
            label = torch.cat((label,input[k][1]),0)

        data = get_variable(data)
        label = get_variable(label)

        Y0.append(get_Y0Matrix("kmeans", labSize, False, data))
        C.append(adjustMatrixC(TrainSelfExpressionLayer(data,batch_size).fc[0].weight, 0))
        D.append(get_DMatrix(C[i]))
        l_ture.append(label)
        # print(M[i])

        print("init %d param ok" %(i))
    end = time.clock()
    info = psutil.virtual_memory()
    print("load data ok use %ds" %(end-start))
    print("memory use :", info.percent)


    #************************带有lossBackward的**************************************************
    # for layer in range(layers):

    #     lastLoss = 0
    #     outputs = 0

    #     # data = get_variable(torch.FloatTensor(features))

    #     for i in range(num_epochs):

    #         for j in range(batch):
    #             module[layer].changeYZ0(Y0[j])
    #             start = time.clock()
    #             Y,Z = module[layer](C[j], D[j])

    #             Yk = Y[layer]
    #             Zk = Z[layer]
    #             Yt = torch.transpose(Yk, 1, 0)
    #             Zt = torch.transpose(Zk, 1, 0)

    #             lossY = torch.trace(Yt.mm(M[j]).mm(Yk))
    #             lossZ = torch.trace(Zt.mm(M[j]).mm(Zk))
                

    #             lastLoss = 1/2 * lossY + 1/2 * lossZ
    #             _,outputs = torch.max(Yk, 1)
    #             outputs = outputs + 1
    #             optimizer[layer].zero_grad()
    #             lastLoss.backward(retain_graph=True)


    #             optimizer[layer].step()
                
    #             end = time.clock()
    #             info = psutil.virtual_memory()
    #             print("%d step use %ds" %(j,end-start))
    #             print("memory use :", info.percent)

    #             print("run %d" %(j))
    #         # if i % 50 == 0:
    #             print("layer %d epoch [%d/%d] batch[%d/%d] loss %.4f" % (layer, i, num_epochs,j,batch,lastLoss.item()))
        
    #         y[layer] = lastLoss
            
    #         saveCSV("./tempData/Yk_"+str(layer+1)+".csv",numpy.array(outputs.numpy()))

    # plt.title("layer_loss")
    # plt.xlabel("layer")
    # plt.ylabel("loss")
    # plt.plot(x,y)
    # plt.show()


    #***************************不带backward************************************************************
    lossfunc = torch.nn.CrossEntropyLoss
    for layer in range(layers):
        module = SpectralNetLayer(batch_size,L0,alpha,beta,layer+1,labSize)

        module.changeYZ0(Y0[0])
        Y,Z,M = module(C[0], D[0])

        Yk = Y[layer]
        Zk = Z[layer]
        Yt = torch.transpose(Yk, 1, 0)
        Zt = torch.transpose(Zk, 1, 0)


        _,outputs = torch.max(Yk, 1)
        outputs = outputs + 1


        # print(batch_size)
        print("layer ", layer)
        print_accuracy(outputs.numpy(), l_ture[0].numpy(), labSize)
        # print("layer %d loss is %.4f" %(layer,lastLoss.item()))

def TrainSiameseNetwork():
    batch_size = 50
    labSize = 10
    positive_num = 4
    negetive_num = 10
    train_dataset = downloadData(True)
    train_loader = loadData(train_dataset, batch_size, True)
    num_epochs = 300

    module = SiameseNetwork()
    optimizer = torch.optim.Adam(module.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_func = ContrastiveLoss()

    lossX = numpy.arange(0,num_epochs)
    lossY = numpy.zeros(num_epochs)

    for i, (images, labels) in enumerate(train_loader):
        if i == num_epochs:
            break
        images = get_variable(images)
        labels = get_variable(labels)
        totLoss = 0

        #使用真实类标初始化正负点对
        neibor = getKneibor(lables=labels)
        isLabels = True
        
        # index = getKneibor(X=data,n_nbrs=batch_size)
        
        for batch in range(batch_size):
            for j in range(batch+1,batch_size):
                label = neibor[batch][j]
                output1,output2 = module(images[batch:batch+1],images[j:j+1])
                output1 = output1.view(output1.shape[0],-1)
                output2 = output2.view(output1.shape[0],-1)
                loss = loss_func(output1,output2, label)
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
                totLoss += loss.item()


        # print("epoch [%d/%d]  loss is %.4f" %(i,len(train_loader),totLoss))
        lossY[i] = totLoss
    torch.save(module.state_dict(), './module/Siamese2.pkl')
    plt.xlabel("layer")
    plt.ylabel("loss")
    plt.plot(lossX,lossY)
    plt.show()


def TrainNumberData():
    #初始化参数
    batch_size = 1024
    tot_size = 60000
    num_epochs = 1
    num_train = 500
    num_display = 10
    labSize = 10
    layers = 3
    fcWeight = 1000
    SpWeight = 10
    useTrainData =  True
    shuffle = True
    dataMethod = "siamese"
    getAMethod = "siamese"
    spectralLossName = "normal"
    n_nbrs = 100
    learning_rate = 1e-5
    weight_decay = 1e-5
    param1 = 1
    param2 = 0
    param3 = 0
    param4 = 0
    #初始化数据
    train_dataset = downloadData(useTrainData)
    train_loader = loadData(train_dataset, tot_size, shuffle)

    #初始化模型
    module = SSpectralNet(5)
    module2 = SSpectralNet(32)
    cnn = CNN()
    cnn.load_state_dict(torch.load("./module/CNN.pkl"))
    cnn2 = CNN2()
    cnn2.load_state_dict(torch.load("./module/CNN2.pkl"))
    siamese = SiameseNetwork()
    siamese.load_state_dict(torch.load("./module/Siamese2.pkl"))

    #初始化优化器
    optimizer = torch.optim.Adam(module.parameters(), lr=learning_rate)
    optimizer2 = torch.optim.Adam(module2.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3*num_train, gamma=0.1)
    scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3*num_train, gamma=0.1)

    #初始化描点
    lossX = numpy.arange(0,num_epochs*num_train)
    lossY = numpy.zeros(num_epochs*num_train)
    lossY2 = numpy.zeros(num_epochs*num_train)
    accX = numpy.arange(0,num_epochs*num_train)
    accY = numpy.zeros(num_epochs*num_train)
    accY2 = numpy.zeros(num_epochs*num_train)

    #初始化损失函数
    spectralLossFunc = getLoss(spectralLossName)


    images,labels =  next(iter(train_loader))
    images = get_variable(images)
    images = F.normalize(images)
    labels = get_variable(labels)

    #数据变形
    # images2 = transformData(images,"Conv",cnn2)
    images = transformData(images,dataMethod,siamese)
    for epoch in range(num_epochs):
        totalAcc = 0
        totalLoss = 0
        #打乱数据顺序
        indices = torch.randperm(tot_size)
        images = images[indices]
        # images2 = images2[indices]
        labels = labels[indices]
        x_train = images[:batch_size]
        # x_train2 = images2[:batch_size]
        y_train = labels[:batch_size]

        for i in range(num_train):
            #向前传播
            Y = module(x_train)
            # Y2 = module2(x_train2)

            #计算准确率
            _,outputs = kmeans(labSize, Y.detach().numpy())
            acc = print_accuracy(outputs, y_train.numpy(), labSize)

            # _,outputs2 = kmeans(labSize, Y2.detach().numpy())
            # acc2 = print_accuracy(outputs2, y_train.numpy(), labSize)

            #计算相似度矩阵
            A = get_AMatrix(x_train,getAMethod)

            #计算loss
            spectralLoss = spectralLossFunc(Y,A)
            # spectralLoss2 = spectralLossFunc(Y2,A)

            #反响传播
            optimizer.zero_grad()
            spectralLoss.backward(retain_graph=True)
            optimizer.step()
            scheduler.step()

            # optimizer2.zero_grad()
            # spectralLoss2.backward(retain_graph=True)
            # optimizer2.step()
            # scheduler2.step()

            #打印s
            if i % num_display == 0:
                print("epcho [%d/%d] train[%d/%d] S\loss is %lf,acc is %lf" %(epoch,num_epochs,i,num_train,spectralLoss.item(),acc))
                # print("epcho [%d/%d] train[%d/%d] loss is %lf,loss2 is %lf,acc is %lf,acc2 is %lf" %(epoch,num_epochs,i,num_train,spectralLoss.item(),spectralLoss2.item(),acc,acc2))
            lossY[epoch*num_train + i] = spectralLoss.item()*100
            # lossY2[epoch*num_train + i] = spectralLoss2.item()
            accY[epoch*num_train + i] = acc
            # accY2[epoch*num_train + i] = acc2


    #保存模型    
    torch.save(module.state_dict(), './module/SpectralNetNorm_Tanh.pkl')

    #画图
    plt.subplot(2,1,1)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.plot(lossX,lossY)
    # plt.plot(lossX,lossY2)

    plt.subplot(2,1,2)
    plt.xlabel("epoch")
    plt.ylabel("accY")
    plt.plot(accX,accY)
    # plt.plot(accX,accY2)
    plt.show()

def Test():
    batch_size = 1024
    train_dataset = downloadData(True)
    num_epochs = 500
    train_loader = loadData(train_dataset, batch_size, True)
    num_train = 1
    cnn = CNN()
    cnn.load_state_dict(torch.load("./module/CNN.pkl"))
    sspectral = SSpectralNet()

    lossfunc1 = getLoss("normal")
    lossfunc2 = getLoss("kmeansLoss")
    lossfunc3 = torch.nn.CrossEntropyLoss()
    lossfunc4 = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(sspectral.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2*num_epochs, gamma=0.1)


    lossX = numpy.arange(0,num_epochs*num_train)
    lossY1 = numpy.zeros(num_epochs*num_train)
    lossY2 = numpy.zeros(num_epochs*num_train)
    lossY3 = numpy.zeros(num_epochs*num_train)
    lossY4 = numpy.zeros(num_epochs*num_train)

    accX = numpy.arange(0,num_epochs*num_train)
    accY = numpy.zeros(num_epochs*num_train)
    accZ = numpy.zeros(num_epochs*num_train)
    for j,(images,labels) in  enumerate(train_loader):
        if j == num_train:
            break
        images = get_variable(images)
        labels = get_variable(labels)
        images = F.normalize(images)
        images = images.view(images.shape[0],-1)
        for i in range(num_epochs):
            encode,decode,Y,Z = sspectral(images,cnn)

            A = get_AMatrix(encode.detach(),"c_k",n_nbrs=200)
            _,outputsY = kmeans(10, Y.detach().numpy())
            # _,outputsY = torch.max(Y, 1)
            # _,outputsZ = kmeans(10, Z.detach().numpy())
            _,outputsZ = torch.max(Z, 1)

            loss1 = lossfunc1(Y,A)
            loss2 = lossfunc2(encode,Z)

            # outputsZ,_ = get_y_preds(outputs,outputsY, 10)
            targetY = torch.LongTensor(outputsY)
            loss3 = lossfunc3(Z,targetY)
            loss4 = lossfunc4(images,decode)
            P1 = 1
            P2 = 1
            P3 = 0
            P4 = 0
            loss = P1*loss1 + P2*loss2 + P3*loss3 + P4*loss4


            print(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            AccY = print_accuracy(outputsY, labels.numpy(), 10)
            AccZ = print_accuracy(outputsZ, labels.numpy(), 10)
            print(j,i,loss1.item(),loss2.item(),loss3.item(),loss4.item(),AccY,AccZ)
            lossY1[j*num_epochs + i] = loss1.item()
            lossY2[j*num_epochs + i] = loss2.item()
            lossY3[j*num_epochs + i] = loss3.item()
            lossY4[j*num_epochs + i] = loss4.item()
            accY[j*num_epochs + i] = AccY
            accZ[j*num_epochs + i] = AccZ

    # show_img(images.view(images.shape[0],1,28,28),batch_size)
    #画图
    plt.subplot(5,1,1)
    plt.xlabel("layer")
    plt.ylabel("Y")
    plt.plot(lossX,lossY1)
    # plt.plot(SlossX,SlossY)

    plt.subplot(5,1,2)
    plt.xlabel("layer")
    plt.ylabel("Z")
    plt.plot(lossX,lossY2)

    plt.subplot(5,1,3)
    plt.xlabel("layer")
    plt.ylabel("cross")
    plt.plot(lossX,lossY3)

    plt.subplot(5,1,4)
    plt.xlabel("layer")
    plt.ylabel("norm")
    plt.plot(lossX,lossY4)

    plt.subplot(5,1,5)
    plt.xlabel("layer")
    plt.ylabel("accY")
    plt.plot(accX,accY)
    plt.plot(accX,accZ)
    plt.show()



if __name__ == "__main__":
    #***************文本数据训练***************************
    # features,A,D,C,init_Y,labels = readMat("./data/data/Isolet_7797.mat")
    # TrainCSVData3(init_Y, features, A, D, labels)
    # TrainCNN()
    # TrainSiameseNetwork()
    # TrainAutoEncoderLayer(features)

    #加载训练好的编码器
    # cnn = CNN()
    # cnn.load_state_dict(torch.load("./module/CNN.pkl"))
    # encode,_ = AutoEncoderLayer(torch.FloatTensor(features))

    #deepAE
    # TrainSelfExpressionLayer()

    #加载训练好的siamese
    # siamese = SiameseNetwork()
    # siamese.load_state_dict(torch.load("./module/Siamese.pkl"))

    # TrainCSVData(AutoEncoderLayer)
    TrainNumberData() 
    # Test()

    
    # print(C[0][0])
    # print(D)
    



    # eta = 0.01
    # A = torch.FloatTensor(A)
    # M = get_MMatirx(C=A, parm=eta, D=torch.FloatTensor(D))
    #离散初始化Y0
    # Y0 = torch.FloatTensor(init_Y)


    #连续初始化Y0
    # Y0 = get_Y0Matrix("kmeans", C[0][0], False, encode)


    # input = list()


    # for i in range(encode.shape[0]):
    #     input.append((encode[i:i+1,:],torch.FloatTensor(labels[i])))

    # TrainCSVData(AutoEncoderLayer, input, 700, 26)



    #*****************************画图*****************

    # x = numpy.arange(1,6)
    # y = numpy.array([0.03091,0.02988,0.03001,0.05592,0.03912])
    # graph(x,y,"layer","ACC","ACC_layer")



    