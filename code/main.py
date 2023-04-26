# main function
import torch
from dataset import PDEDataSet
import optimal
import model
import os
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from ignite.utils import convert_tensor
import deepxde as dde

# setting parameters                                                      

path_save_loss_0 = "./outputs/loss3_0.npy"
path_save_loss_1 = "./outputs/loss3_1.npy"
path_save_loss_2 = "./outputs/loss3_2.npy"
path_save_para_0 = "./outputs/para3_0.npy"
path_save_para_1 = "./outputs/para3_1.npy"
path_save_para_2 = "./outputs/para3_2.npy"
os.makedirs('./outputs',exist_ok=True)
os.makedirs("./exp",exist_ok=True)
os.makedirs("./exp_ad",exist_ok=True)

device = torch.device("cuda:0")# if torch.cuda.is_available() else "cpu")
torch.autograd.set_detect_anomaly(True)
# fixed seed
np.random.seed(20230415)
# generate data
sample_size = 24*24
beta = [1,1,1,1,1]
scale = 0.1
train_data = PDEDataSet(sample_size,beta,scale)
train_data.generate_data()
batch_size = 96
data_loader = DataLoader(train_data, batch_size=batch_size)
# models
solver = model.PINN_solver(2,1,device)
equation_func = model.equation_part()
true_pde_solver = model.PDE_solver(data_loader,2,1,beta=0,niter=5, 
        nupdate=100, nlog=5, nepoch=200, device=device, tor = 1e-3)
fullmodel = model.full_model(solver,equation_func)
# start 

# zero step
# r = 4, s = 5, d = 2
lr = 1e-3
optimizer = torch.optim.Adam(fullmodel.parameters(), lr=lr, betas=(0.9, 0.999))
lamda = 0.1
para_optim = optimal.OptimalProcess(data_loader, fullmodel, optimizer, lamda,niter=5, 
        nupdate=100, nlog=5, nepoch=200, path='./exp', device=device, tor = 1e-3)
train_loss_0, para_mat_0 = para_optim.run()

# first step
# estimated beta
beta_hat = {}
for i in equation_func.beta.keys():
    beta_hat[i] = equation_func.beta[i].data
# solve equation
true_pde_solver.update_beta(beta_hat)
true_pde_solver.run()
# reset optimizer
para_optim.reset_optim()
lamda = 0.15
para_optim.update_lamda(lamda)
# estimate high order derivatives
position = convert_tensor(torch.tensor(train_data.pos_vec).float(), device)
old_u = true_pde_solver.get_u(position,order = 3)
# update dataset
train_data.update_data(old_u)
data_loader_update = DataLoader(train_data, batch_size=batch_size)
para_optim.update_data(data_loader_update)
train_loss_1, para_mat_1 = para_optim.run(iter_time=1)

# second step
# estimated beta
beta_hat = {}
for i in equation_func.beta.keys():
    beta_hat[i] = equation_func.beta[i].data
# solve equation
true_pde_solver.update_beta(beta_hat)
true_pde_solver.reset_optim()
true_pde_solver.run()
# reset optimizer
para_optim.reset_optim()
lamda = 0.20
para_optim.update_lamda(lamda)
# estimate high order derivatives
position = convert_tensor(torch.tensor(train_data.pos_vec).float(), device)
old_u = true_pde_solver.get_u(position,order = 2)
# update dataset
train_data.update_data(old_u)
data_loader_update = DataLoader(train_data, batch_size=batch_size)
para_optim.update_data(data_loader_update)
train_loss_2, para_mat_2 = para_optim.run(iter_time=2)


save_mat_loss_0 = np.array([train_loss_0[keys] for keys in train_loss_0.keys()])
save_mat_loss_1 = np.array([train_loss_1[keys] for keys in train_loss_1.keys()])
save_mat_loss_2 = np.array([train_loss_2[keys] for keys in train_loss_2.keys()])

save_mat_para_0 = np.array([para_mat_0[keys] for keys in para_mat_0.keys()])
save_mat_para_1 = np.array([para_mat_1[keys] for keys in para_mat_1.keys()])
save_mat_para_2 = np.array([para_mat_2[keys] for keys in para_mat_2.keys()])

np.save(path_save_loss_0,save_mat_loss_0)
np.save(path_save_loss_1,save_mat_loss_1)
np.save(path_save_loss_2,save_mat_loss_2)

np.save(path_save_para_0,save_mat_para_0)
np.save(path_save_para_1,save_mat_para_1)
np.save(path_save_para_2,save_mat_para_2)






# optimizer = torch.optim.Adam(full_model.parameters(), lr=tau_1, betas=(0.9, 0.999))
# train_process = GFN_optimal.OptimalProcess(
#     train=train,test=test,net=full_model,optimizer=optimizer,lambda_0=lambda_0,
#     tau_2=tau_2,niter=niter,nupdate=nupdate,nepoch=nepoch,nlog=nlog,path = path,device=device,tor=tor_1)
# ret = train_process.run()
# train_loss = ret["train_loss_mat"]
# test_loss = ret["test_loss_mat"]
# parameter_value = ret["parameter_mat"]
# parameter_value2 = ret["parameter_mat2"]
# #print(train_loss)
# #print(test_loss)
# #print(parameter_value)
# #print(parameter_value2)
# save_mat = np.array([train_loss,test_loss,parameter_value,parameter_value2])
# np.save(path_save,save_mat)
