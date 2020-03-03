#定义各种loss函数
import torch

#卷积层提取特征值loss函数
#input&output 是二维矩阵
def cnnLoss(output, input):
    loss = torch.norm((input - output))
    return loss

#自我表达层loss函数
def selfExpressionLoss(e, C, Z):
    loss1 = e * torch.norm(C).numpy()
    loss2 = 1/2 * torch.norm(Z - Z.mm(C)).numpy() ** 2
    return loss1 + loss2





