# Here I'm just importing the files
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def generate_data(ndata,nfeat, split=0.8):
    # a is the number of examples and b is the number of features

    # create a random matrix of size ndata_points X nfeatures
    ndata = 1000
    nfeat = 5

    # this function samples a matrix of random values from uniform distn over [0,1]
    a = np.random.rand(ndata, nfeat)

    # divide into train and test data
    tsize = int(ndata*split)
    train_data = a[:tsize, :]
    test_data = a[tsize:, :]
    # print("train and test data generated")
    # print("generating ground truth")
    # ground truth must be boolean
    gt = np.random.rand(ndata, 1)
    # >0.5 means all the values above 0.5 will become true and <0.5 will become false
    # convert 
    gt = gt>0.5
    gt = gt.astype(float)
    train_gt = gt[:tsize]
    test_gt = gt[tsize:]
    # print("generated everything")      
    return train_data, train_gt, test_data, test_gt



tdata, tgt, tedata, tegt = generate_data(1000, 5, 0.7)
# convert all data to tensors
tdata = torch.Tensor(tdata)
tgt = torch.Tensor(tgt)
tedata = torch.Tensor(tedata)
tegt = torch.Tensor(tegt)
# print("creation of datasets done")
# print("Now lets try building the model")

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(5, 3)
        self.fc2 = nn.Linear(3, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        # generally x's size is ndata x feat
        l1out = self.relu(self.fc1(x))
        l2out = self.fc2(l1out)
        out = self.sigmoid(l2out)
        return out

model = Net()

loss_fn = nn.BCELoss()

nepochs = 1000
optimiser = optim.Adam(model.parameters(), lr=1e-3)

for i in range(0,nepochs):
    # forward propagate and return the output
    model.train()
    y = model(tdata)
    # calculate the loss
    loss = loss_fn(y, tgt)

    # now backpropagation
    # .backward() calculated the gradients and stores them with the tensor
    loss.backward()
    # now we need to do the Gradient descent
    optimiser.step()
    # lets do the testing
    with torch.no_grad():
        model.eval()
        y_test = model(tedata)
        test_loss = loss_fn(y_test, tegt)

    print("train loss: ", loss.item(), "test loss: ", test_loss.item())

