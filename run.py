import torch
import torch.nn as nn
from data_preoperator import *
from layer import *
from loss import *

# init parameters
num_epochs = 10
batch_size = 100
learning_rate = 0.001
weight_decay = 1e-5

Y0 = 1.0 /batch_size * torch.rand(batch_size, batch_size, dtype=torch.float32)
Z0 = torch.zeros(batch_size, batch_size, dtype=torch.float32)
L0 = torch.zeros(batch_size, batch_size, dtype=torch.float32)
alpha = 0.1
beta = 0.01
layers = 10
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


    module = FCLayer(1, batch_size, batch_size)

    loss_func = selfExpressionLoss
    optimizer = torch.optim.Adam(module.parameters(), lr=learning_rate, weight_decay=weight_decay)

    Z = auto_images.view(auto_images.size(0), -1)
    epochs = 10000
    for epoch in range(epochs):
        ZC = module(Z)
        C  = module.state_dict()['fc.0.weight']
        e = 0.01

        loss = loss_func(e, Z, C, ZC)
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        # print('selfExpression Epoch [%d/%d],Loss: %.4f'
        #     % (epoch + 1, epochs, loss.item()))

    
    # Save the Trained Model
    # torch.save(module.state_dict(), './module/SelfExpression.pkl')
    return module

def TrainSpectralNetLayer(Autoencoder):
    train_dataset = downloadData(True)
    train_loader = loadData(train_dataset, batch_size, True)

    module = SpectralNetLayer(batch_size,Y0,L0,Z0,alpha,beta,layers,labSize)    

    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam(module.parameters(), lr=learning_rate, weight_decay=weight_decay)

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = get_variable(images)
            labels = get_variable(labels)

            auto_images,_ = Autoencoder(images)
            print(auto_images.size())
            selfExpressionLayer = TrainSelfExpressionLayer(auto_images)
            C  = selfExpressionLayer.state_dict()['fc.0.weight']
            eta = 0.01
            M = get_MMatirx(C, eta)

            print(M.size())
            outputs = module(M)
            print(outputs)
            
    
    # Save the Trained Model
    # torch.save(module.state_dict(), './module/SelfExpression.pkl')
    return module

if __name__ == "__main__":
    # TrainAutoEncoderLayer()
    AutoEncoderLayer = Autoencoder()
    AutoEncoderLayer.load_state_dict(torch.load("./module/AutoEncoder.pkl"))

    TrainSpectralNetLayer(AutoEncoderLayer)

    