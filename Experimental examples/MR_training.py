# %% This code is to...
# author: Hong-Wei Li, Email: hongweili@seu.edu.cn

import random
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.io import savemat, loadmat
from torch import nn, optim

# chose to use gpu or cpu
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# To guarantee same results for every running, which might slow down the training speed
torch.set_default_dtype(torch.float64)
seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
np.random.seed(seed)  # Train_numpy module.
random.seed(seed)  # Python random module.
torch.manual_seed(seed)

# Define the configuration
h_s = 20
in_s = 1
out_s = 1
layer_s = 1


# Define LSTM
class Lstm(nn.Module):
    def __init__(self):
        super(Lstm, self).__init__()
        self.lstm = nn.LSTM(in_s, h_s, layer_s, bias = False)
        self.linear = nn.Linear(h_s, out_s, bias = False)

    def forward(self, u):
        h0 = torch.zeros(layer_s, np.size(u, 1), h_s)
        c0 = torch.zeros(layer_s, np.size(u, 1), h_s)
        y, (hn, cn) = self.lstm(u, (h0, c0))
        y = self.linear(y)
        return y


# Create model
LSTM_model = Lstm()
criterion = nn.MSELoss()

# Prepare data
f = loadmat('MR_training_data.mat')
dt = 0.005
u = f['u']
Nt = np.size(u, 0)
tend = (Nt - 1) * dt
t = np.linspace(0, tend, Nt).reshape(-1, 1)
u_torch = torch.tensor(u).reshape([Nt, 1, in_s])
y_ref = f['y_ref']
y_ref_torch = torch.tensor(y_ref).reshape([Nt, 1, in_s]).to(device)

# %% Training
Train_num = 10000
optimizer = optim.Adam(LSTM_model.parameters(), 0.01)
loss_all = np.zeros((Train_num + 1, 1))
y_pre_torch = LSTM_model(u_torch).to(device)
loss = criterion(y_pre_torch, y_ref_torch)
loss_all[0:1, :] = loss.item()
loss_m = loss.item()
LSTM_model_m = LSTM_model
start = time.time()
for i in range(Train_num):
    LSTM_model.zero_grad()
    loss.backward()
    optimizer.step()
    y_pre_torch = LSTM_model(u_torch).to(device)
    loss = criterion(y_pre_torch, y_ref_torch)
    i1 = i + 1
    loss_all[i1:i1+1, :] = loss.item()

    if loss.item() < loss_m:
        loss_m = loss.item()
        LSTM_model_m = LSTM_model

    if i1 % 10 == 0 or i == 0:
        print(f'iteration: {i1}, LSTM loss: {loss.item()}')
        end = time.time()
        per_time = (end - start) / i1
        print('Average training time: %.6f s per one training' % per_time)
        print('Cumulative training time: %.6f s' % (end - start))
        left_time = (Train_num - i1) * per_time
        print(f"Executed at {time.strftime('%H:%M:%S', time.localtime())},", 'left time: %.6f s\n' % left_time)

end = time.time()
cost_time = end - start
print('Total training time: %.3f s' % cost_time)
print(f'LSTM loss: {loss.item()}')

y_pre = y_pre_torch.reshape([Nt, 1]).detach().numpy()
torch.save(LSTM_model_m.state_dict(), 'MR_trained_model.pt')
# %% Load trained model if the users do not want to train the model again
LSTM_model.load_state_dict(torch.load('MR_trained_model.pt'))
# %% Check training performance
y_pre_torch = LSTM_model(u_torch).to(device)
y_pre = y_pre_torch.reshape([Nt, 1]).detach().numpy()

plt.plot(u, y_ref)
plt.plot(u, y_pre)
plt.show()

# %%
in_l = 1 # input length
in_s = 1 # input size
batch_s = 2 # batch size (random integer)
IN = torch.zeros((in_l, batch_s, in_s))
LSTM_model.eval()
torch.onnx.export(LSTM_model, IN, 'MR.onnx',
                  input_names = ['input'], output_names = ['output'],
                  dynamic_axes = {'input': {1: 'batch'},
                                  'output': {1: 'batch'}},
                  opset_version=14) # version supported by MATLAB
