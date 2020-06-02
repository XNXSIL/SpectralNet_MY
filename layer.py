#进行各个层的定义
import torch
import torch.nn as nn
import math
from data_preoperator import *
import psutil
from sklearn.cluster import SpectralClustering

class FCLear(nn.Module):
    def __init__(self):
        super(FCLear, self).__init__()

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(10,1024,bias=True),
            torch.nn.Dropout(p = 0.5),
            torch.nn.ReLU(),
            torch.nn.Linear(1024,1024,bias=True),
            torch.nn.Dropout(p = 0.5),
            torch.nn.ReLU(),
            torch.nn.Linear(1024,512,bias=True),
            torch.nn.Dropout(p = 0.5),
            torch.nn.ReLU(),
            torch.nn.Linear(512,10,bias=True),
            torch.nn.Softmax()
        )


    def forward(self, x):
        x = self.fc(x)
        return x

class SelfExpressionLayer(torch.nn.Module):
    def __init__(self):
        super(SelfExpressionLayer, self).__init__()
        self.fc = torch.nn.Linear(1024,1024,bias=False)
        # self.fc.weight = torch.nn.Parameter(torch.zeros(1024,1024,dtype=torch.float32))
    def forward(self,x):
        x = self.getCoef().mm(x)
        return x
    
    def getCoef(self):
        return abs(self.fc.weight + torch.transpose(self.fc.weight,1,0))/2

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784,500,bias=True),
            nn.ReLU(),
            nn.Linear(500,500,bias=True),
            nn.ReLU(),
            nn.Linear(500,2000,bias=True),
            nn.ReLU(),
            nn.Linear(2000,10,bias=True)
        )

        self.decoder = nn.Sequential(
            nn.Linear(10,2000,bias=True),
            nn.ReLU(),
            nn.Linear(2000,500,bias=True),
            nn.ReLU(),
            nn.Linear(500,500,bias=True),
            nn.ReLU(),
            nn.Linear(500,784,bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        encode = self.encoder(x)
        decode = self.decoder(encode)
        return encode, decode

class CNN2(nn.Module):
    def __init__(self):
        super(CNN2, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=3, padding=1),  # (b, 16, 10, 10)
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # (b, 16, 5, 5)
            nn.Conv2d(16, 8, 3, stride=2, padding=1),  # (b, 8, 3, 3)
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)  # (b, 8, 2, 2)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2),  # (b, 16, 5, 5)
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # (b, 8, 15, 15)
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),  # (b, 1, 28, 28)
            nn.Tanh()
        )

    def forward(self, x):
        encode = self.encoder(x)
        decode = self.decoder(encode)
        return encode, decode

#用于传统的全连层的图像分类
class DeepAutoEnCoderSelfExpress(nn.Module):
    def __init__(self, batch_size):
        super(DeepAutoEnCoderSelfExpress, self).__init__()

        self.fc = nn.Linear(batch_size, batch_size,bias=False)
        self.cnn = CNN()
        self.cnn.load_state_dict(torch.load("./module/CNN.pkl"))

    def active(self,x):
        return torch.nn.functional.softmax(x)

    def forward(self, x):
        encode = self.cnn.encoder(x)

        Z = encode.view(encode.shape[0],-1)

        # C = cosC(Z,C,30)
        # adjustMatrixC(C)

        ZC = torch.transpose(self.fc(torch.transpose(Z,1,0)),1,0)

        decode = self.cnn.decoder(ZC.view(ZC.shape[0],16,5,5))

        C = self.fc.weight
        return C,Z,ZC,encode,decode

class SiameseNetwork(nn.Module):
     def __init__(self):
         super(SiameseNetwork, self).__init__()
         self.cnn1 = nn.Sequential(
             nn.ReflectionPad2d(1),
             nn.Conv2d(1, 4, kernel_size=3),
             nn.ReLU(inplace=True),
             nn.BatchNorm2d(4),
             nn.Dropout2d(p=.2),
             
             nn.ReflectionPad2d(1),
             nn.Conv2d(4, 8, kernel_size=3),
             nn.ReLU(inplace=True),
             nn.BatchNorm2d(8),
             nn.Dropout2d(p=.2),
 
             nn.ReflectionPad2d(1),
             nn.Conv2d(8, 8, kernel_size=3),
             nn.ReLU(inplace=True),
             nn.BatchNorm2d(8),
             nn.Dropout2d(p=.2),
         )
 
         self.fc1 = nn.Sequential(
             nn.Linear(6272, 500),
             nn.ReLU(inplace=True),
 
             nn.Linear(500, 500),
             nn.ReLU(inplace=True),
 
             nn.Linear(500, 5)
         )
 
     def forward_once(self, x):
         output = self.cnn1(x)
         output = output.view(output.size()[0], -1)
         output = self.fc1(output)
         return output
 
     def forward(self, input1, input2):
         output1 = self.forward_once(input1)
         output2 = self.forward_once(input2)
         return output1, output2


def orthonorm_op(x, epsilon=1e-7):
    '''
    Computes a matrix that orthogonalizes the input matrix x

    x:      an n x d input matrix
    eps:    epsilon to prevent nonzero values in the diagonal entries of x

    returns:    a d x d matrix, ortho_weights, which orthogonalizes x by
                right multiplication
    '''
    x_2 = torch.mm(torch.transpose(x, 1, 0), x)
    x_2 += torch.eye(x.size()[1])*epsilon
    
    L = torch.cholesky(x_2)

    ortho_weights = torch.transpose(torch.inverse(L), 1, 0) * math.sqrt(x.shape[0])
    return ortho_weights

def orthonorm(x,ortho_weights=None):
    x = x.double()
    if ortho_weights == None:
        ortho_weights = orthonorm_op(x)
    return x.mm(ortho_weights).float()

def orthonorm_my(Y):
    _,outputs = torch.max(Y, 1)

    out = torch.zeros(Y.shape[0], Y.shape[1])
    labNum = list()
    for i in range(Y.shape[1]):
        k = len(outputs[outputs == i])
        k = k if k > 0 else 1
        labNum.append(1 / math.sqrt(k))
    for i in range(Y.shape[0]):
        out[i][outputs[i]] = labNum[outputs[i]]
    return Variable(out, requires_grad=True)

class SSpectralNet(torch.nn.Module):
    def __init__(self,inputSize):
        super(SSpectralNet, self).__init__()

        
        self.spectral = SpectralNetNorm(inputSize)


    def forward(self,x):
        Y = orthonorm(self.spectral(x))
        return Y


class SpectralNetNorm(torch.nn.Module):
    def __init__(self,inputSize):
        super(SpectralNetNorm, self).__init__()

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(inputSize,1024,bias=True),
            torch.nn.Dropout(p = 0.3),
            torch.nn.ReLU(),
            torch.nn.Linear(1024,1024,bias=True),
            torch.nn.Dropout(p = 0.3),
            torch.nn.ReLU(),
            torch.nn.Linear(1024,512,bias=True),
            torch.nn.Dropout(p = 0.3),
            torch.nn.ReLU(),
            torch.nn.Linear(512,10,bias=True),
            torch.nn.Tanh()
        )

    def active(self, Y):
        return torch.nn.functional.softmax(Y,dim=1)



    def forward(self, X):
        X = self.fc(X)

        return X

#深度谱聚类层
class SpectralNetLayer(torch.nn.Module):
    def __init__(self,batch_size,L0,alpha,beta,layers,labSize):
        super(SpectralNetLayer, self).__init__()

        #相似度矩阵size
        self.batch_size = batch_size

        #初始化参数
        self.Y0 = 0
        self.L0 = L0
        self.Z0 = 0

        #初始化权重
        # self.W = W

        self.labSize = labSize

        #网络层数
        self.layers = layers

        self.alpha = alpha
        self.beta = beta
        self.eta = 0.01
        self.t = 0.01

        #全连接层
        self.fc = torch.nn.ModuleList()
        self.fc2 = torch.nn.ModuleList()
        for i in range(layers):
            self.fc.append(torch.nn.Linear(self.labSize, self.labSize, bias = False))
            self.fc2.append(torch.nn.Linear(self.labSize, self.labSize, bias = False))

        # #权重初始化
        # for m in self.modules():
        #     if isinstance(m, torch.nn.Linear):
        #         #nn.init.kaiming_normal_(m.weight, mode='fan_out')
        #         #m.weight.data.normal_(0, 1/20)
        #         m.weight = torch.nn.Parameter(self.W.t()) 
        #         #m.weight = torch.nn.Parameter(self.A.t())


    def self_active(self, x):
        return torch.nn.functional.softmax(x,dim=1)
    
    def dicrete(self,x):
        _,index = torch.max(x,1)
        out = torch.zeros(x.shape[0],x.shape[1])
        for i in range(x.shape[0]):
            out[i][index[i]] = 1

        return out

    def calYZL(self, A, D, Yk, Zk, Lk, t, beta):
        #Yk shape is n*c
        #A 是邻接矩阵
        #D 是度矩阵

        U = get_UMatrix(Yk, A, D)
        Yk_1 = Yk
        Zk_1 = Zk
        Lk = torch.transpose(Lk,1,0)
        Lk_1 = Lk
        loss1 = 0
        loss2 = 0
        loss3 = 0
        for i in range(Yk_1.shape[1]):
            M = U[i][i] ** 2 * D - U[i][i] * A
            Yk_1[:,i:i+1] = Yk[:,i:i+1] - t * (M.mm(Zk[:,i:i+1]) + Lk[:,i:i+1] + beta * (Yk[:,i:i+1] - Zk[:,i:i+1]))
            Zk_1[:,i:i+1] = Zk[:,i:i+1] - t * (torch.transpose(M,1,0).mm(Yk_1[:,i:i+1]) - Lk[:,i:i+1] - beta * (Yk_1[:,i:i+1] - Zk[:,i:i+1]))
            Lk_1[:,i:i+1] = Lk[:,i:i+1] + beta * (Yk_1[:,i:i+1] - Zk_1[:,i:i+1])
            
            loss1 += torch.transpose(Yk_1[:,i:i+1],1,0).mm(M).mm(Zk_1[:,i:i+1])
            loss2 += torch.transpose(Lk_1[:,i:i+1],1,0).mm(Yk_1[:,i:i+1] - Zk_1[:,i:i+1])
            loss3 += 1/2 * beta * torch.norm((Yk_1[:,i:i+1]- Zk_1[:,i:i+1]))

        return Yk_1,Zk_1,torch.transpose(Lk_1,1,0),loss1,loss2,loss3


    def changeYZ0(self, Y0):
        self.Y0 = Y0
        self.Z0 = Y0

    def forward(self, C, D):

        Y = list()
        Var = list()
        Z = list()
        L = list()
        Loss1 = list()
        Loss2 = list()
        Loss3 = list()

        for k in range(self.layers):
            info = psutil.virtual_memory()
            print("memory use :", k, info.percent)
            if k == 0 : 
                Yk,Zk,Lk,loss1,loss2,loss3 = self.calYZL(C, D, self.Y0, self.self_active(self.Z0), self.L0, self.t, self.beta)

                Y.append(self.self_active(Yk))
                Z.append(self.self_active(Zk))
                L.append(Lk)
                Loss1.append(loss1)
                Loss2.append(loss2)
                Loss3.append(loss3)
            else :
                Yk,Zk,Lk,loss1,loss2,loss3 = self.calYZL(C, D, Y[k-1], Z[k-1], L[k-1], self.t, self.beta)

                Y.append(self.self_active(Yk))
                Z.append(self.self_active(orthonorm(Zk)))
                L.append(Lk)

                Loss1.append(loss1)
                Loss2.append(loss2)
                Loss3.append(loss3)
            print(Y[k])
            print("layer %d loss1 is %.4f, loss2 is %.4f, loss3 is %.4f" %(k,loss1,loss2,loss3))
        return Y,Z,Loss1,Loss2,Loss3

