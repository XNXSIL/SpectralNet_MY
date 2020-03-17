#定义各种loss函数
import torch


#自我表达层loss函数
def selfExpressionLoss(e, Z, C, ZC):
    loss1 = e * torch.norm(C)
    loss2 = 1/2 * torch.norm(Z - ZC) ** 2
    print("loss1 %.4f loss2 %.4f" %(loss1, loss2))
    return loss1 + loss2





