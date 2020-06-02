import torch.utils.data as data
import torch
import h5py

class DataFromH5File(data.Dataset):
    def __init__(self, filepath):
        h5File = h5py.File(filepath, 'r')
        self.hr = h5File['hr']
        self.lr = h5File['lr']
        
    def __getitem__(self, idx):
        label = torch.from_numpy(self.hr[idx]).float()
        data = torch.from_numpy(self.lr[idx]).float()
        return data, label
    
    def __len__(self):
        assert self.hr.shape[0] == self.lr.shape[0], "Wrong data length"
        return self.hr.shape[0]

trainset = DataFromH5File("./pretrain_weight/ae_mnist_weights.h5")
train_loader = data.DataLoader(dataset=trainset, batch_size=1024, shuffle=True,  num_workers=8, pin_memory=True)

for step, (lr, hr) in enumerate(train_loader):
    print(lr.shape)
    print(hr.shape)
    break