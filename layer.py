#进行各个层的定义
import torch
import torch.nn as nn

#编码层，用于提取特征矩阵Z
class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return out



#解码层，用于将特征矩阵反推到原图片
class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=5, padding=2),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.MaxPool2d(2))
 
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return out 


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

class TrainLayer(nn.Module):
    def __init__(self):
        super(TrainLayer, self).__init__()
        # 使用序列工具快速构建
        self.encoder = EncoderLayer()
        self.decoder = DecoderLayer()


        if torch.cuda.is_available():
            self.encoder = encoder.cuda()
            self.decoder = decoder.cuda()
 
    def forward(self, x):
        out = self.encoder(x)
        out = self.decoder(out)
        return out



