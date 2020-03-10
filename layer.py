#进行各个层的定义
import torch
import torch.nn as nn
import math

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
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
class FCLayer(nn.Module):
    def __init__(self, layerSize, inSize, outSize):
        super(FCLayer, self).__init__()

        self.layerSize = layerSize
        self.fc = nn.ModuleList()
        for i in range(layerSize-1):
            self.fc.append(nn.Linear(inSize, inSize))
        self.fc.append(nn.Linear(inSize, outSize))

    def forward(self, x):

        out = x
        for k in range(self.layerSize):
            out = self.fc[k](out)
        return out 

# class TrainLayer(nn.Module):
#     def __init__(self):
#         super(TrainLayer, self).__init__()
#         # 使用序列工具快速构建
#         self.encoder = EncoderLayer()
#         self.decoder = DecoderLayer()


#         if torch.cuda.is_available():
#             self.encoder = self.encoder.cuda()
#             self.decoder = self.decoder.cuda()
 
#     def forward(self, x):
#         out = self.encoder(x)
#         out = self.decoder(out)
#         return out


def orthonorm_op(x, epsilon=1e-7):
    '''
    Computes a matrix that orthogonalizes the input matrix x

    x:      an n x d input matrix
    eps:    epsilon to prevent nonzero values in the diagonal entries of x

    returns:    a d x d matrix, ortho_weights, which orthogonalizes x by
                right multiplication
    '''
    x_2 = torch.mm(torch.transpose(x, 1, 0), x)
    x_2 = torch.eye(x.size()[1])*epsilon
    
    L = torch.cholesky(x_2)

    ortho_weights = torch.transpose(torch.inverse(L), 1, 0) * math.sqrt(x.size()[1])
    return ortho_weights

def orthonorm(x):
    ortho_weights = orthonorm_op(x)
    return x.mm(ortho_weights)

#深度谱聚类层
class SpectralNetLayer(torch.nn.Module):
    def __init__(self,batch_size,Y0,L0,Z0,alpha,beta,layers,labSize):
        super(SpectralNetLayer, self).__init__()

        #相似度矩阵size
        self.batch_size = batch_size

        #初始化参数
        self.Y0 = Y0
        self.L0 = L0
        self.Z0 = Z0

        #初始化权重
        # self.W = W

        self.labSize = labSize

        #网络层数
        self.layers = layers

        self.alpha = alpha
        self.beta = beta

        #全连接层
        self.fc = torch.nn.ModuleList()
        for i in range(layers):
            self.fc.append(torch.nn.Linear(self.batch_size, self.batch_size, bias = False))

        # #权重初始化
        # for m in self.modules():
        #     if isinstance(m, torch.nn.Linear):
        #         #nn.init.kaiming_normal_(m.weight, mode='fan_out')
        #         #m.weight.data.normal_(0, 1/20)
        #         m.weight = torch.nn.Parameter(self.W.t()) 
        #         #m.weight = torch.nn.Parameter(self.A.t())


    def self_active(self, x):
        return torch.nn.functional.softmax(x)
    
    def forward(self, M):

        Y = list()
        Var = list()
        Z = list()
        L = list()

        for k in range(self.layers):
            if k == 0 : 
                Y.append(self.self_active(self.Y0 - self.fc[k](M.mm(self.Y0) + self.L0 + self.beta*(self.Y0 - self.Z0))))
                Var.append(self.L0 + self.alpha*self.beta*(Y[k] - self.Z0))
                # Z.append(orthonorm(self.Z0 - self.fc[k](M.mm(self.Z0) -  Var[k] - self.beta*(Y[k] - self.Z0))))
                Z.append(self.Z0 - self.fc[k](M.mm(self.Z0) -  Var[k] - self.beta*(Y[k] - self.Z0)))
                L.append(Var[k] + self.alpha*self.beta*(Y[k] - Z[k]))

                
                
            else :
                Y.append(self.self_active(self.Y[k-1] - self.fc[k](M.mm(self.Y[k-1]) + self.L[k-1] + self.beta*(self.Y[k-1] - self.Z[k-1]))))
                Var.append(self.L[k-1] + self.alpha*self.beta*(Y[k] - self.Z[k-1]))
                # Z.append(orthonorm(self.Z[k-1] - self.fc[k](M.mm(self.Z[k-1]) -  Var[k] - self.beta*(Y[k] - self.Z[k-1]))))
                Z.append(self.Z[k-1] - self.fc[k](M.mm(self.Z[k-1]) -  Var[k] - self.beta*(Y[k] - self.Z[k-1])))
                L.append(Var[k] + self.alpha*self.beta*(Y[k] - Z[k]))

        return Y

