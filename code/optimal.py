# optimal process
import numpy as np
import torch
import torch.nn as nn
import os, sys
from ignite.utils import convert_tensor
import deepxde as dde
import collections


from nuisance_function import Logger
import model

# define inner optimal process 
class OptimalProcess(object):
    def __init__(
        self, train, model, optimizer,
        lamda,niter=1, 
        nupdate=100, nlog=5, nepoch=1, path='./exp', device="cpu", tor = 1e-3):

        self.device = device
        self.path = path
        self.logger = Logger(filename=os.path.join(self.path, 'log.txt'))
        print(' '.join(sys.argv))
        self.train = train
        self.nepoch = nepoch
        
        self.traj_loss = nn.MSELoss()
        self.model = model
        self.optimizer = optimizer

        self.niter = niter

        self.nlog = nlog
        self.nupdate = nupdate
        self.tor = tor    
        
        self.loss = 0
        # resample_size: list [indomain,boundary,initial]
        self.lamda = lamda
          
    def log(self, epoch, iteration, metrics):
        message = '[{step}][{epoch}/{max_epoch}][{i}/{max_i}]'.format(
            step=epoch *len(self.train)+ iteration+1,
            epoch=epoch+1,
            max_epoch=self.nepoch,
            i=iteration+1,
            max_i=len(self.train)
        )
        for name, value in metrics.items():
            message += ' | {name}: {value:.2e}'.format(name=name, value=value)
            
        print(message)
    
    def reset_optim(self):
        self.optimizer.state = collections.defaultdict(dict)
        print('The optimizer has been reset')
    
    def update_lamda(self,lamda):
        self.lamda = lamda
        
    def update_data(self,data):
        self.train = data
        
    def train_step(self, batch, iter_time, old_u=False, backward = True):
        batch = convert_tensor(batch, self.device)
        # states: batch*value_dim
        # position: batch*domain_dim
        position = batch["position"].float()
        states = batch["value"].float()
        f = batch["f"].float()
        bound_position = batch["bound_position"].float()
        bound_condition = batch["bound_condition"]
        if iter_time > 0:
            old_u = batch["old_u"]
        # data part
        # pred = self.solver(position)
        # loss_data = self.traj_loss(pred,states)
        
        # equation part 
        position.requires_grad_(True)
        bound_position.requires_grad_(True)
        u = self.model(position)
        bound_u = self.model(bound_position)
        equation_left,loss_aug,loss_bound = self.model.equation_value(position,u,bound_position,bound_u,bound_condition,iter_time,old_u)
        loss_eq = self.traj_loss(equation_left,f)
        # data part
        loss_data = self.traj_loss(u,states)
        
        loss = loss_data + self.lamda * (loss_eq + loss_aug + loss_bound)
        if backward:
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        loss_dict = {
            'loss_bound': loss_bound,
            'loss_eq': loss_eq,
            'loss_data': loss_data,
            'loss_aug': loss_aug,
            'loss': loss
        }
        
        return loss_dict
    
    def run(self,iter_time = 0):
        train_loss_mat = {'loss_bound':[],'loss_eq':[],'loss_data':[]}
        parameter_mat = {'beta_0':[],'beta_1':[],'beta_2':[],'beta_3':[],'beta_4':[]}
        loss_old = -100
        loss_now = -100
        print('\n--------------------------------update started--------------------------------')
        print('--------------------------------the',iter_time,'iteration--------------------------------')
        for epoch in range(self.nepoch): 
            for iteration, data in enumerate(self.train, 0):
                for _ in range(self.niter):
                    self.loss = self.train_step(data,iter_time)

                total_iteration = epoch * (len(self.train)) + (iteration + 1)

                if total_iteration % self.nlog == 0:
                    self.log(epoch, iteration,self.loss)
                    print('beta_0:',self.model.equation_func.beta['beta_0'].data,
                          'beta_1:',self.model.equation_func.beta['beta_1'].data,
                          'beta_2:',self.model.equation_func.beta['beta_2'].data,
                          'beta_3:',self.model.equation_func.beta['beta_3'].data,
                          'beta_4:',self.model.equation_func.beta['beta_4'].data,)
            for keys in train_loss_mat.keys():
                train_loss_mat[keys].append(self.loss[keys].item())
            for keys in parameter_mat.keys():
                parameter_mat[keys].append(torch.Tensor.detach(torch.Tensor.cpu(self.model.equation_func.beta[keys])).numpy().copy())
                       
            loss_now = self.loss['loss'].detach()
            if abs(loss_now-loss_old)/abs(loss_now) < 1e-4 and epoch >= 100:
                print('\nIteration break due to loss decreasing stopped. current loss is:', loss_now,'Current epoch is:', epoch+1)
                break
            loss_old = loss_now
        print('--------------------------------update finished--------------------------------\n')
        return train_loss_mat, parameter_mat

 
    
                
                
