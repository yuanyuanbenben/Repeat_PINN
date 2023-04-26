import numpy as np
import torch 
import torch.nn as nn 
from ignite.utils import convert_tensor
import deepxde as dde
import collections
from siren_pytorch import Sine
# import copy

class PINN_solver(nn.Module):
    def __init__(self,input_dim,output_dim,device):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, 16),
            Sine(),
            nn.Linear(16,64),
            Sine(),
            nn.Linear(64,256),
            Sine(),
            nn.Linear(256,64),
            Sine(),
            nn.Linear(64,16),
            Sine(),
            nn.Linear(16,self.output_dim),
        ).to(self.device)

        
    def forward(self,x):
        return self.net(x)

class equation_part(nn.Module):
    def __init__(self):
        super().__init__()
        self.beta = nn.ParameterDict({
            'beta_0': nn.Parameter(torch.tensor(0.5)), 
            'beta_1': nn.Parameter(torch.tensor(0.5)), 
            'beta_2': nn.Parameter(torch.tensor(0.5)), 
            'beta_3': nn.Parameter(torch.tensor(0.5)), 
            'beta_4': nn.Parameter(torch.tensor(0.5)), 
            # 'beta_5': nn.Parameter(torch.tensor(0.5)), 
            # 'beta_6': nn.Parameter(torch.tensor(0.5)), 
            # 'beta_': nn.Parameter(torch.tensor(0.5)), 
        })
        self.traj_loss = nn.MSELoss()
        
    def forward(self,position,u,bound_position,bound_u,bound_condition,iter_time,old_u=False):
        u_x = dde.grad.jacobian(u, position, i=0, j=0)
        u_xx = dde.grad.jacobian(u_x, position, i=0, j=0)
        b_u_x = dde.grad.jacobian(bound_u, bound_position, i=0, j=0)
        if iter_time < 2:
            u_xxx = dde.grad.jacobian(u_xx, position, i=0, j=0)
            b_u_xx = dde.grad.jacobian(b_u_x, bound_position, i=0, j=0)
            if iter_time < 1:
                u_xxxx = dde.grad.jacobian(u_xxx, position, i=0, j=0)
                b_u_xxx = dde.grad.jacobian(b_u_xx, bound_position, i=0, j=0)
        # u_xxxxx = dde.grad.jacobian(u_xxxx, position, i=0, j=0)
        # u_xxxxxx = dde.grad.jacobian(u_xxxxx, position, i=0, j=0)
        
        u_y = dde.grad.jacobian(u, position, i=0, j=1)
        u_yy = dde.grad.jacobian(u_y, position, i=0, j=1)
        b_u_y = dde.grad.jacobian(bound_u, bound_position, i=0, j=1)
        if iter_time < 2:
            u_yyy = dde.grad.jacobian(u_yy, position, i=0, j=1)
            b_u_yy = dde.grad.jacobian(b_u_y, bound_position, i=0, j=1)
            if iter_time < 1:
                u_yyyy = dde.grad.jacobian(u_yyy, position, i=0, j=1)
                b_u_yyy = dde.grad.jacobian(b_u_yy, bound_position, i=0, j=1)
        # u_yyyyy = dde.grad.jacobian(u_yyyy, position, i=0, j=1)
        # u_yyyyyy = dde.grad.jacobian(u_yyyyy, position, i=0, j=1)
        loss_agum = 0
        if iter_time == 2:
            u_xx_old = old_u['u_xx']
            u_yy_old = old_u['u_yy']
            u_xxx = old_u['u_xxx']
            u_yyy = old_u['u_yyy']
            u_xxxx = old_u['u_xxxx']
            u_yyyy = old_u['u_yyyy']
            loss_agum = self.traj_loss(u_xx_old,u_xx) + self.traj_loss(u_yy_old,u_yy)
            true_b_u = bound_condition['u'].float()
            true_b_u_x = bound_condition['u_x'].float()
            true_b_u_y = bound_condition['u_y'].float()
            loss_bound = self.traj_loss(bound_u,true_b_u) + self.traj_loss(b_u_x,true_b_u_x) + self.traj_loss(b_u_y,true_b_u_y)
            
        if iter_time == 1:
            u_xxx_old = old_u['u_xxx']
            u_yyy_old = old_u['u_yyy']
            u_xxxx = old_u['u_xxxx']
            u_yyyy = old_u['u_yyyy']
            loss_agum = self.traj_loss(u_xxx_old,u_xxx) + self.traj_loss(u_yyy_old,u_yyy)
            true_b_u = bound_condition['u'].float()
            true_b_u_x = bound_condition['u_x'].float()
            true_b_u_y = bound_condition['u_y'].float()
            true_b_u_xx = bound_condition['u_xx'].float()
            true_b_u_yy = bound_condition['u_yy'].float()
            loss_bound = self.traj_loss(bound_u,true_b_u) + self.traj_loss(b_u_x,true_b_u_x) + self.traj_loss(b_u_y,true_b_u_y) + self.traj_loss(b_u_xx,true_b_u_xx) + self.traj_loss(b_u_yy,true_b_u_yy)
        
        if iter_time == 0:
            true_b_u = bound_condition['u'].float()
            true_b_u_x = bound_condition['u_x'].float()
            true_b_u_y = bound_condition['u_y'].float()
            true_b_u_xx = bound_condition['u_xx'].float()
            true_b_u_yy = bound_condition['u_yy'].float()
            true_b_u_xxx = bound_condition['u_xxx'].float()
            true_b_u_yyy = bound_condition['u_yyy'].float()
            loss_bound = self.traj_loss(bound_u,true_b_u) + self.traj_loss(b_u_x,true_b_u_x) + self.traj_loss(b_u_y,true_b_u_y
                        ) + self.traj_loss(b_u_xx,true_b_u_xx) + self.traj_loss(b_u_yy,true_b_u_yy) + self.traj_loss(b_u_xxx,true_b_u_xxx) + self.traj_loss(b_u_yyy,true_b_u_yyy)
        # equation_left = (u_xx**2 + u_yy**2) * self.beta['beta_'] + u * self.beta['beta_0'] + (u_x+u_y) * self.beta['beta_1'] + (u_xx + u_yy) * self.beta['beta_2'] + (
        #     u_xxx + u_yyy)* self.beta['beta_3'] +  (u_xxxx + u_yyyy) * self.beta['beta_4'] #+ (
        #     # u_xxxxx + u_yyyyy) * self.beta['beta_5'] + (u_xxxxxx + u_yyyyyy) * self.beta['beta_6'] 
        
        equation_left = self.beta['beta_0'] * (u**2) + self.beta['beta_1'] * (u_x*u_y) +self.beta['beta_2']*(
            u_xx + u_yy)**2 + self.beta['beta_3'] * (u_xxx + u_yyy) + self.beta['beta_4'] * (u_xxxx + u_yyyy)
        
        return equation_left, loss_agum, loss_bound


    
class full_model(nn.Module):
    def __init__(self,solver,equation_func):
        super().__init__()
        self.solver = solver
        self.equation_func = equation_func
        
    def forward(self,positions):
        return self.solver(positions)
    
    def equation_value(self,positions,u,bound_position,bound_u,bound_condition,iter_time,old_u):
        return self.equation_func(positions,u,bound_position,bound_u,bound_condition,iter_time,old_u)
    

class PDE_solver(nn.Module):
    def __init__(self,train,input_dim,output_dim,beta,niter=1, 
        nupdate=100, nlog=5, nepoch=1, device="cpu", tor = 1e-3):
        super().__init__()
        self.train = train
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, 16),
            Sine(),
            nn.Linear(16,64),
            Sine(),
            nn.Linear(64,256),
            Sine(),
            nn.Linear(256,256),
            Sine(),
            nn.Linear(256,64),
            Sine(),
            nn.Linear(64,16),
            Sine(),
            nn.Linear(16,self.output_dim),
        ).to(self.device)
        self.nepoch = nepoch
        self.beta = beta
        self.traj_loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.net.parameters(),lr=1e-3,betas=(0.9, 0.999))
        self.niter = niter

        self.nlog = nlog
        self.nupdate = nupdate
        self.tor = tor    
        
        self.loss = 0
        
    def forward(self,x):
        return self.net(x)
    
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
        
    def update_beta(self,beta):
        self.beta = beta
        
    def reset_optim(self):
        self.optimizer.state = collections.defaultdict(dict)
        
    def equation_value(self,position,u,bound_position,bound_u,bound_condition):
        u_x = dde.grad.jacobian(u, position, i=0, j=0)
        u_xx = dde.grad.jacobian(u_x, position, i=0, j=0)
        u_xxx = dde.grad.jacobian(u_xx, position, i=0, j=0)
        u_xxxx = dde.grad.jacobian(u_xxx, position, i=0, j=0)
        # u_xxxxx = dde.grad.jacobian(u_xxxx, position, i=0, j=0)
        # u_xxxxxx = dde.grad.jacobian(u_xxxxx, position, i=0, j=0)
        
        u_y = dde.grad.jacobian(u, position, i=0, j=1)
        u_yy = dde.grad.jacobian(u_y, position, i=0, j=1)
        u_yyy = dde.grad.jacobian(u_yy, position, i=0, j=1)
        u_yyyy = dde.grad.jacobian(u_yyy, position, i=0, j=1)
        # u_yyyyy = dde.grad.jacobian(u_yyyy, position, i=0, j=1)
        # u_yyyyyy = dde.grad.jacobian(u_yyyyy, position, i=0, j=1)
        
        b_u_x = dde.grad.jacobian(bound_u, bound_position, i=0, j=0)
        b_u_xx = dde.grad.jacobian(b_u_x, bound_position, i=0, j=0)
        b_u_xxx = dde.grad.jacobian(b_u_xx, bound_position, i=0, j=0)
    
        b_u_y = dde.grad.jacobian(bound_u, bound_position, i=0, j=1)
        b_u_yy = dde.grad.jacobian(b_u_y, bound_position, i=0, j=1)
        b_u_yyy = dde.grad.jacobian(b_u_yy, bound_position, i=0, j=1)
        
        equation_left =  self.beta['beta_0']*(u**2) +  self.beta['beta_1']*(u_x*u_y) +self.beta['beta_2']*(
            u_xx + u_yy)**2 + self.beta['beta_3'] * (u_xxx + u_yyy) + self.beta['beta_4'] * (u_xxxx + u_yyyy)
        
        true_b_u = bound_condition['u'].float()
        true_b_u_x = bound_condition['u_x'].float()
        true_b_u_y = bound_condition['u_y'].float()
        true_b_u_xx = bound_condition['u_xx'].float()
        true_b_u_yy = bound_condition['u_yy'].float()
        true_b_u_xxx = bound_condition['u_xxx'].float()
        true_b_u_yyy = bound_condition['u_yyy'].float()
        
        loss_bound = self.traj_loss(bound_u,true_b_u) + self.traj_loss(b_u_x,true_b_u_x) + self.traj_loss(b_u_y,true_b_u_y
                        ) + self.traj_loss(b_u_xx,true_b_u_xx) + self.traj_loss(b_u_yy,true_b_u_yy) + self.traj_loss(b_u_xxx,true_b_u_xxx) + self.traj_loss(b_u_yyy,true_b_u_yyy)
        return equation_left, loss_bound
    
    def train_step(self, batch, backward = True):
        batch = convert_tensor(batch, self.device)
        # states: batch*value_dim
        # position: batch*domain_dim
        position = batch["position"].float()
        states = batch["value"].float()
        f = batch["f"].float()
        bound_position = batch["bound_position"].float()
        bound_condition = batch["bound_condition"]
        # data part
        # pred = self.solver(position)
        # loss_data = self.traj_loss(pred,states)
        
        # equation part 
        position.requires_grad_(True)
        bound_position.requires_grad_(True)
        u = self.net(position)
        bound_u = self.net(bound_position)
        equation_left, loss_bound = self.equation_value(position,u,bound_position,bound_u,bound_condition)
        loss_eq = self.traj_loss(equation_left,f)
        
        loss_data = self.traj_loss(u,states)
        
        loss = loss_eq + 1e-6 * loss_data + loss_bound * 10
        if backward:
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        loss = {
            'loss_bound':loss_bound,
            'loss_eq': loss_eq,
            'loss_data': loss_data,
            'loss': loss
        }
        
        return loss
    
    def run(self):
        loss_old = -100
        loss_now = -100
        print('\n--------------------------------solve PDE started--------------------------------')
        print(self.beta)
        for epoch in range(self.nepoch): 
            for iteration, data in enumerate(self.train, 0):
                for _ in range(self.niter):
                    self.loss = self.train_step(data)

                total_iteration = epoch * (len(self.train)) + (iteration + 1)

                if total_iteration % self.nlog == 0:
                    self.log(epoch, iteration,self.loss)
            loss_now = self.loss['loss'].detach()
            if abs(loss_now-loss_old)/abs(loss_now) < 1e-4 and epoch >= 100:
                print('\nIteration break due to loss decreasing stopped. current loss is:', loss_now,'Current epoch is:', epoch+1)
                break
            loss_old = loss_now
        print('--------------------------------solve PDE finished--------------------------------\n')
        
    def get_u(self,position,order):
        position.requires_grad_(True)
        u = self.net(position)
        u_x = dde.grad.jacobian(u, position, i=0, j=0)
        u_y = dde.grad.jacobian(u, position, i=0, j=1)
        u_xx = dde.grad.jacobian(u_x, position, i=0, j=0)
        u_xxx = dde.grad.jacobian(u_xx, position, i=0, j=0)
        u_xxxx = dde.grad.jacobian(u_xxx, position, i=0, j=0)
        u_yy = dde.grad.jacobian(u_y, position, i=0, j=1)
        u_yyy = dde.grad.jacobian(u_yy, position, i=0, j=1)
        u_yyyy = dde.grad.jacobian(u_yyy, position, i=0, j=1)
        if order == 2:
            return {'u_xx':u_xx.data, 'u_yy':u_yy.data, 'u_xxx':u_xxx.data,'u_yyy':u_yyy.data,'u_xxxx':u_xxxx.data,'u_yyyy':u_yyyy.data}
        if order == 3:
            return {'u_xxx':u_xxx.data,'u_yyy':u_yyy.data,'u_xxxx':u_xxxx.data,'u_yyyy':u_yyyy.data}
        