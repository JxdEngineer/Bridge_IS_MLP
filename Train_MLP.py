# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 12:13:13 2022

@author: jxudong
"""
# use the two lines to solve "OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
# OMP: Hint This means that multiple copies of the OpenMP runtime have been linked into the program. That is dangerous, since it can degrade performance or cause incorrect results. The best thing to do is to ensure that only a single OpenMP runtime is linked into the process, e.g. by avoiding static linking of the OpenMP runtime in any library. As an unsafe, unsupported, undocumented workaround you can set the environment variable KMP_DUPLICATE_LIB_OK=TRUE to allow the program to continue to execute, but that may cause crashes or silently produce incorrect results. For more information, please see http://www.intel.com/software/products/support/."
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
# %% training
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d, Axes3D 
import time

class Nets(nn.Module):

    def __init__(self,  in_dim, hidden_dim = 128):
        
        super(Nets, self).__init__()
                
        self.mlp = nn.Sequential(
                                nn.Linear(in_dim, hidden_dim),   
                                nn.ELU(),
                                nn.Linear(hidden_dim, hidden_dim),  
                                nn.ELU(),
                                nn.Linear(hidden_dim, hidden_dim),  
                                nn.ELU(),
                                nn.Linear(hidden_dim, hidden_dim),  
                                nn.ELU(),
                                nn.Linear(hidden_dim, 3),  
                                # nn.Dropout(0.1),                                       
                                        )

    def forward(self, XY):
                   
        return self.mlp(XY)
    
def load_data(XY_dir, S_dir):
    XY1 = pd.read_csv(XY_dir[0])
    XY2 = pd.read_csv(XY_dir[1])
    XY3 = pd.read_csv(XY_dir[2])
    XY4 = pd.read_csv(XY_dir[3])
    S = pd.read_csv(S_dir)
    
    XY1 = torch.from_numpy(XY1.to_numpy()).float()
    XY2 = torch.from_numpy(XY2.to_numpy()).float()
    XY3 = torch.from_numpy(XY3.to_numpy()).float()
    XY4 = torch.from_numpy(XY4.to_numpy()).float()
    S = torch.from_numpy(S.to_numpy()).float()
  
    return XY1, XY2, XY3, XY4, S

# use GPU for training
device = torch.device("cuda")

if __name__ == '__main__':
    
    start_time = time.time()
    
    n_epoch = 200
    lr = 1e-3
    batch_size = 128
    wd = 0.2 # regularization factor
    
    XY_dir = ["./wheel1_loc.csv",
            "./wheel2_loc.csv",
            "./wheel3_loc.csv",
            "./wheel4_loc.csv"]
    S_dir = "./u_raw.csv"
       
    XY1, XY2, XY3, XY4, S = load_data(XY_dir, S_dir)
    S = S[:, [0,1,2]]

    # Seed
    # seed = 123
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)

    IF_net = Nets(in_dim = 2)
    params = (
                list(IF_net.parameters())         
              )    
    optimizer = optim.Adam(params, weight_decay=wd)
    
    loss_mse = nn.MSELoss()
    loss_meter = []
    
    IF_net = IF_net.to(device)
    loss_mse = loss_mse.to(device)
    
    train_set = torch.utils.data.TensorDataset(XY1, XY2, XY3, XY4, S)
    train_loader = torch.utils.data.DataLoader(
                                            train_set,
                                            batch_size= batch_size,
                                            shuffle=True)   
    IF_net.train()
    
    for epoch in range(n_epoch):
        for which_batch, (XY1_batch, XY2_batch, XY3_batch, XY4_batch, S_batch) in enumerate(tqdm(train_loader)):
            batch_loss = 0
            XY1_batch = XY1_batch.to(device)
            XY2_batch = XY2_batch.to(device)
            XY3_batch = XY3_batch.to(device)
            XY4_batch = XY4_batch.to(device)
            S_batch = S_batch.to(device) 
            optimizer.zero_grad()         
            S_test = 3.0696*IF_net(XY1_batch) + 3.0696*IF_net(XY2_batch) + \
                3.2952*IF_net(XY3_batch) + 3.2952*IF_net(XY4_batch)   # axle weight * IS
            loss = loss_mse(S_test, S_batch)
            loss.backward()
            optimizer.step()  
            batch_loss += loss         
        batch_loss = loss / batch_size  
        print('Epoch: {}, loss: {:.4f}'.format(epoch, batch_loss)) 
        loss_meter.append(batch_loss.cpu().detach().numpy())   
        
    print("--- %s seconds ---" % (time.time() - start_time))
    
    plt.close('all')
    plt.figure()
    plt.plot(np.asarray(loss_meter))
# %% save identified influence surface (evaluate the identified IS with MATLAB)
mesh_N = 101
X = np.linspace(-6, 6, num = mesh_N)
Y = np.linspace(0, 40, num = mesh_N)
X, Y = np.meshgrid(X, Y)
X = np.expand_dims(X, axis = -1)
Y = np.expand_dims(Y, axis = -1)
XY_test = np.concatenate((X,Y),-1) 
XY_test = torch.from_numpy(XY_test).float()
IF_net.eval()
Z_test = IF_net(XY_test.to(device)).cpu()
Z_test = Z_test.detach().numpy()
XY_test = XY_test.detach().numpy()

X_save = np.reshape(X,(mesh_N,mesh_N))
Y_save = np.reshape(Y,(mesh_N,mesh_N))      
Z_save = Z_test.transpose(2,0,1).reshape(-1,Z_test.shape[1])
np.savetxt('X.csv', X_save, delimiter=',')
np.savetxt('Y.csv', Y_save, delimiter=',')
np.savetxt('Z.csv', Z_save, delimiter=',')
np.savetxt('loss.csv', np.asarray(loss_meter), delimiter=',')

# visualize the influence surface
plt.close('all')
sensor_n = 1-1
xs = X.squeeze(-1)
ys = Y.squeeze(-1)
zs = Z_test[:,:,sensor_n]
fig, ax = plt.subplots(subplot_kw={"projection": "3d"},
                       )
surf = ax.plot_surface(xs, ys, zs, cmap=cm.coolwarm,
                    linewidth=0, antialiased=False)