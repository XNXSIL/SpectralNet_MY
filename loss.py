#定义各种loss函数
import torch
import torch.nn.functional as F
from data_preoperator import *
#自我表达层loss函数
def selfExpressionLoss(alpha1,alpha2, Z, C, ZC, input,output):
    IX = input.view(input.shape[0],-1)
    OX = output.view(output.shape[0],-1)
    loss1 = 0.5 * alpha1 * torch.sum( (IX-OX) ** 2)
    loss2 = alpha2 * torch.sum(C ** 2)
    loss3 = 0.5 * torch.sum( (Z - ZC) ** 2)
    loss = loss1 + loss2 + loss3
    # print("loss1 %.4f loss2 %.4f loss3 %.4f totLoss %.4f" %(loss1, loss2,loss3,loss))
    return loss

def kmeansLoss(X,Y):
    Xt = torch.transpose(X,1,0)
    Yt = torch.transpose(Y,1,0)
    loss = torch.norm((Xt - Xt.mm(Y).mm(torch.inverse(Yt.mm(Y) + torch.eye(Y.shape[1]) * 0.0001)).mm(Yt)))
    return loss

def normalSpectralLoss(Y,A):
    loss = torch.norm(squared_distance(Y,W=A).mul(A),1)
    # D = squared_distance(Y)
    # loss = torch.sum(A.mm(D)) / (2 * A.shape[0])
    return loss

def getLoss(lossName):
    if lossName == "normal":
        return normalSpectralLoss
    elif lossName == "kmeansLoss":
        return kmeansLoss


        
class ContrastiveLoss(torch.nn.Module):

    def __init__(self, margin=16.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = squared_distance(output1,Y=output2)
        loss_contrastive = torch.mean((1-label) * euclidean_distance +     # calmp夹断用法
                                      (label) * torch.clamp(self.margin - euclidean_distance, min=0.0))
        return loss_contrastive




