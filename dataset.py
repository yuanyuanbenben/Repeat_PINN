import numpy as np
import torch
from torch.utils.data import Dataset

class PDEDataSet(Dataset):
    def __init__(self,sample_size,beta,scale=0.1):
        super().__init__()
        self.u_vec = 0
        self.pos_vec = 0
        self.f_vec = 0
        self.len = 0
        
        self.sample_size = sample_size
        self.beta = beta
        self.scale = scale
        self.r = len(beta)
        self.grid_size = int(np.sqrt(self.sample_size))
        
        self.N = 200
        self.s = 5
        self.idx = 0
        self.old_u = False
        
    def fft_cos_grid(self,grid,N=200):
        # grid: grid
        # ret: grid * N
        ret = np.array([[np.cos(k*np.pi*x/5) for k in range(1,N+1)] for x in grid])
        return 1/np.sqrt(5) * ret
    
    def fft_sin_grid(self,grid,N=200):
        # grid: grid
        # ret: grid * N
        ret = np.array([[np.sin(k*np.pi*x/5) for k in range(1,N+1)] for x in grid])
        return 1/np.sqrt(5) * ret
    
    def fft_value(self,alpha,value_grid):
        # alpha: N * 1
        # ret : grid * 1
        return np.matmul(value_grid,alpha)
    
    def generate_data(self):
        # Define the domain and grid size
        print('\n--------------------------------generate data--------------------------------')
        x_grid_size = self.grid_size
        y_grid_size = self.grid_size
        x_range = np.linspace(0, 10, x_grid_size, endpoint=True)
        y_range = np.linspace(0, 10, y_grid_size, endpoint=True)
        
        # grid value for basis
        grid_cos_x = self.fft_cos_grid(x_range,self.N)
        grid_sin_x = self.fft_sin_grid(x_range,self.N)
        grid_cos_y = self.fft_cos_grid(y_range,self.N)
        grid_sin_y = self.fft_sin_grid(y_range,self.N)
        
        # Generate coefficients
        u_x_coef = np.array([((-1)**(k+1)) * (k**(-(self.s+0.5))) for k in range(1,self.N+1)]).reshape(-1,1)
        u_y_coef = np.array([(k**(-(self.s+0.5))) for k in range(1,self.N+1)]).reshape(-1,1) * 2
        
        # alpha_c = np.array([np.sum(self.beta[i] * ((-1)**(k+1+(i)//2)) * ((np.pi/5)**(i)) * (k**(-(self.s+0.5-i))) for i in range(self.r)) for k in range(1,self.N+1)]).reshape(-1,1)
        # alpha_s = np.array([np.sum(self.beta[i] * ((-1)**(k+1+(i+1)//2)) * ((np.pi/5)**(i)) * (k**(-(self.s+0.5-i))) for i in range(self.r)) for k in range(1,self.N+1)]).reshape(-1,1)
        # eta_c = np.array([np.sum(self.beta[i] * ((-1)**((i)//2)) * ((np.pi/5)**(i)) * (k**(-(self.s+0.5-i))) for i in range(self.r)) for k in range(1,self.N+1)]).reshape(-1,1) * 2
        # eta_s = np.array([np.sum(self.beta[i] * ((-1)**((i+1)//2)) * ((np.pi/5)**(i)) * (k**(-(self.s+0.5-i))) for i in range(self.r)) for k in range(1,self.N+1)]).reshape(-1,1) * 2
        
        u_x_coef_c = np.array([((-1)**(k+1)) * ((np.pi/5)) * (k**(-(self.s+0.5-1))) for k in range(1,self.N+1)]).reshape(-1,1)
        u_x_coef_s = np.array([((-1)**(k+2)) * ((np.pi/5)) * (k**(-(self.s+0.5-1))) for k in range(1,self.N+1)]).reshape(-1,1)
        u_y_coef_c = np.array([((np.pi/5)) * (k**(-(self.s+0.5-1))) for k in range(1,self.N+1)]).reshape(-1,1) * 2
        u_y_coef_s = np.array([(-1)* ((np.pi/5)) * (k**(-(self.s+0.5-1))) for k in range(1,self.N+1)]).reshape(-1,1) * 2
        
        u_xx_coef_c = np.array([((-1)**(k+2)) * ((np.pi/5)**2) * (k**(-(self.s+0.5-2))) for k in range(1,self.N+1)]).reshape(-1,1)
        u_xx_coef_s = np.array([((-1)**(k+2)) * ((np.pi/5)**2) * (k**(-(self.s+0.5-2))) for k in range(1,self.N+1)]).reshape(-1,1)
        u_yy_coef_c = np.array([(-1)* ((np.pi/5)**2) * (k**(-(self.s+0.5-2))) for k in range(1,self.N+1)]).reshape(-1,1) * 2
        u_yy_coef_s = np.array([(-1)* ((np.pi/5)**2) * (k**(-(self.s+0.5-2))) for k in range(1,self.N+1)]).reshape(-1,1) * 2
        
        u_xxx_coef_c = np.array([((-1)**(k+2)) * ((np.pi/5)**3) * (k**(-(self.s+0.5-3))) for k in range(1,self.N+1)]).reshape(-1,1)
        u_xxx_coef_s = np.array([((-1)**(k+3)) * ((np.pi/5)**3) * (k**(-(self.s+0.5-3))) for k in range(1,self.N+1)]).reshape(-1,1)
        u_yyy_coef_c = np.array([(-1)* ((np.pi/5)**3) * (k**(-(self.s+0.5-3))) for k in range(1,self.N+1)]).reshape(-1,1) * 2
        u_yyy_coef_s = np.array([((-1)**2)* ((np.pi/5)**3) * (k**(-(self.s+0.5-3))) for k in range(1,self.N+1)]).reshape(-1,1) * 2
        
        u_xxxx_coef_c = np.array([((-1)**(k+3)) * ((np.pi/5)**4) * (k**(-(self.s+0.5-4))) for k in range(1,self.N+1)]).reshape(-1,1)
        u_xxxx_coef_s = np.array([((-1)**(k+3)) * ((np.pi/5)**4) * (k**(-(self.s+0.5-4))) for k in range(1,self.N+1)]).reshape(-1,1)
        u_yyyy_coef_c = np.array([((-1)**2)* ((np.pi/5)**4) * (k**(-(self.s+0.5-4))) for k in range(1,self.N+1)]).reshape(-1,1) * 2
        u_yyyy_coef_s = np.array([((-1)**2)* ((np.pi/5)**4) * (k**(-(self.s+0.5-4))) for k in range(1,self.N+1)]).reshape(-1,1) * 2
        
        # get value
        self.ux = self.fft_value(u_x_coef,grid_cos_x) + self.fft_value(u_x_coef,grid_sin_x)
        self.uy = self.fft_value(u_y_coef,grid_cos_y) + self.fft_value(u_y_coef,grid_sin_y)
        
        self.u_x = self.fft_value(u_x_coef_c,grid_cos_x) + self.fft_value(u_x_coef_s,grid_sin_x)
        self.u_y = self.fft_value(u_y_coef_c,grid_cos_y) + self.fft_value(u_y_coef_s,grid_sin_y)
        
        self.u_xx = self.fft_value(u_xx_coef_c,grid_cos_x) + self.fft_value(u_xx_coef_s,grid_sin_x)
        self.u_yy = self.fft_value(u_yy_coef_c,grid_cos_y) + self.fft_value(u_yy_coef_s,grid_sin_y)
        
        self.u_xxx = self.fft_value(u_xxx_coef_c,grid_cos_x) + self.fft_value(u_xxx_coef_s,grid_sin_x)
        self.u_yyy = self.fft_value(u_yyy_coef_c,grid_cos_y) + self.fft_value(u_yyy_coef_s,grid_sin_y)
        
        self.u_xxxx = self.fft_value(u_xxxx_coef_c,grid_cos_x) + self.fft_value(u_xxxx_coef_s,grid_sin_x)
        self.u_yyyy = self.fft_value(u_yyyy_coef_c,grid_cos_y) + self.fft_value(u_yyyy_coef_s,grid_sin_y)
        
        # f_x = self.fft_value(alpha_c,grid_cos_x) + self.fft_value(alpha_s,grid_sin_x)
        # f_y = self.fft_value(eta_c,grid_cos_y) + self.fft_value(eta_s,grid_sin_y)
 
        u = np.array([self.ux[i,0] + self.uy[j,0] for i in range(x_grid_size) for j in range(y_grid_size)])
        
        # s = np.array([u_xx[i,0]**2 + u_yy[j,0]**2 for i in range(x_grid_size) for j in range(y_grid_size)])
        # f_ = np.array([f_x[i,0] + f_y[j,0] for i in range(x_grid_size) for j in range(y_grid_size)])
        # f = f_ + self.beta_ * 
        f = np.array([self.combine(i,j) for i in range(x_grid_size) for j in range(y_grid_size)])
        # Reshape the solution and position arrays into vectors
        u_vec = u + np.random.normal(scale=self.scale,size = u.shape)
        u_vec = u_vec
        pos_vec = np.array([(x, y) for x in x_range for y in y_range])
        f_vec = f
        # boundary condition
        self.b = {}
        self.bound_pos = np.array([(x, y) for x in x_range for y in [0,10]] + [(x,y) for x in [0,10] for y in y_range])
        self.b['u'] = np.array([self.ux[i,0] + self.uy[j,0] for i in range(x_grid_size) for j in [0,y_grid_size-1]] + [self.ux[i,0] + self.uy[j,0] for i in [0,x_grid_size-1] for j in range(y_grid_size)])
        self.b['u_x'] = np.array([self.u_x[i,0] for i in range(x_grid_size) for j in [0,y_grid_size-1]] + [self.u_x[i,0] for i in [0,x_grid_size-1] for j in range(y_grid_size)])
        self.b['u_y'] = np.array([self.u_y[j,0] for i in range(x_grid_size) for j in [0,y_grid_size-1]] + [self.u_y[j,0] for i in [0,x_grid_size-1] for j in range(y_grid_size)])
        self.b['u_xx'] = np.array([self.u_xx[i,0] for i in range(x_grid_size) for j in [0,y_grid_size-1]] + [self.u_xx[i,0] for i in [0,x_grid_size-1] for j in range(y_grid_size)])
        self.b['u_yy'] = np.array([self.u_yy[j,0] for i in range(x_grid_size) for j in [0,y_grid_size-1]] + [self.u_yy[j,0] for i in [0,x_grid_size-1] for j in range(y_grid_size)])
        self.b['u_xxx'] = np.array([self.u_xxx[i,0] for i in range(x_grid_size) for j in [0,y_grid_size-1]] + [self.u_xxx[i,0] for i in [0,x_grid_size-1] for j in range(y_grid_size)])
        self.b['u_yyy'] = np.array([self.u_yyy[j,0] for i in range(x_grid_size) for j in [0,y_grid_size-1]] + [self.u_yyy[j,0] for i in [0,x_grid_size-1] for j in range(y_grid_size)])
        
        # Shuffle the position and solution arrays randomly
        self.idx = np.random.permutation(len(u_vec))
        self.u_vec = u_vec[self.idx]
        lens = len(self.u_vec)
        self.u_vec = np.array(self.u_vec).reshape(lens,1)
        self.pos_vec = pos_vec[self.idx]
        self.f_vec = f_vec[self.idx]
        self.f_vec = np.array(self.f_vec).reshape(lens,1)
        
        self.bound_idx = np.random.permutation(len(u_vec))
        # augement data
        self._bound_pos = np.repeat(self.bound_pos,x_grid_size//4,axis=0)[self.bound_idx]
        self._b = {}
        for i in self.b.keys():
            self._b[i] = np.repeat(self.b[i],x_grid_size//4,axis=0)[self.bound_idx].reshape(lens,1)
       
        
        
        
        self.len = len(self.u_vec)
        print('--------------------------------generate data finished--------------------------------\n')
    
    def combine(self,i,j):
        return self.beta[0] * (self.ux[i,0] + self.uy[j,0])**2 + self.beta[1] * (self.u_x[i,0]*self.u_y[j,0]) + self.beta[2]*(
            self.u_xx[i,0] + self.u_yy[j,0])**2 + self.beta[3]*(self.u_xxx[i,0] + self.u_yyy[j,0]) + self.beta[4]*(
                self.u_xxxx[i,0] + self.u_yyyy[j,0])
    
    def update_data(self,old_u):
        self.old_u = old_u
        
    def item(self, idx):
        bound_dict = {}
        for keys in self._b.keys():
            bound_dict[keys] = torch.tensor(self._b[keys][idx,:])
            
        origin_dict = ({"position":torch.tensor(self.pos_vec[idx,:]), 
                "value":torch.tensor(self.u_vec[idx,:]),
                "f":torch.tensor(self.f_vec[idx,:]),
                "bound_position":torch.tensor(self._bound_pos[idx,:]),
                "bound_condition":bound_dict,
                })
        
        if self.old_u == False:
            return origin_dict
        
        new_dict = {}
        for keys in self.old_u.keys():
            new_dict[keys] = self.old_u[keys][idx,:]
            
        origin_dict['old_u'] = new_dict
        return origin_dict

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        # Split the position and solution arrays into batches
        return self.item(idx)

# import matplotlib.pyplot as plt

# # Define the batch size and number of batches
# batch_size = 1000

# # Create the data loader
# data_loader = DataLoader(PDEDataSet(batch_size), batch_size=batch_size)

# # Select a batch
# batch_pos, batch_u = next(iter(data_loader))

# # Reshape the position and solution arrays into grids
# t_grid = np.reshape(batch_pos[:,:,0], (batch_size, -1))
# x_grid = np.reshape(batch_pos[:,:,1], (batch_size, -1))
# y_grid = np.reshape(batch_pos[:,:,2], (batch_size, -1))
# u_grid = np.reshape(batch_u, (batch_size, -1))

# # Plot the solution as a heat map
# plt.imshow(u_grid, cmap='jet', extent=[0, 1, 0, 1], origin='lower')
# plt.colorbar()
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Batch of size ' + str(batch_size))
# plt.show()


