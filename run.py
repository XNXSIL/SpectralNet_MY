import torch
import torch.nn as nn
from data_preoperator import *
from layer import *
from loss import *

num_epochs = 1
batch_size = 100
learning_rate = 0.001


def TrainModel():
    train_dataset = downloadData(True)
    train_loader = loadData(train_dataset, batch_size, True)

    module = TrainLayer()
    
    # 选择损失函数和优化方法
    loss_func = cnnLoss
    optimizer = torch.optim.Adam(module.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = get_variable(images)
            labels = get_variable(labels)
    
            outputs = module(images)

            loss = loss_func(outputs, images)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            if (i + 1) % 100 == 0:
                print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f'
                    % (epoch + 1, num_epochs, i + 1, len(train_dataset) // batch_size, loss.item()))
    
    
    # Save the Trained Model
    torch.save(module.encoder.state_dict(), './module/cnn.pkl')
    return module

if __name__ == "__main__":
    TrainModel()
    