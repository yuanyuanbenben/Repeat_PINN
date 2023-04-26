import matplotlib.pyplot as plt
import numpy as np
import os
os.makedirs('./pic',exist_ok=True)

save_mat_loss_0 = np.load("./outputs/loss1_0.npy")
save_mat_loss_1 = np.load("./outputs/loss1_1.npy")
save_mat_loss_2 = np.load("./outputs/loss1_2.npy")

save_mat_para_0 = np.load("./outputs/para1_0.npy")
save_mat_para_1 = np.load("./outputs/para1_1.npy")
save_mat_para_2 = np.load("./outputs/para1_2.npy")

# leng_0 = np.shape(save_mat_loss_0)[1]
# leng_1 = np.shape(save_mat_loss_1)[1]
# leng_2 = np.shape(save_mat_loss_2)[1]

# plt.plot(range(leng_0),save_mat_loss_0[0,:],'r',range(leng_0,leng_1+leng_0),save_mat_loss_1[0,:],'b',range(leng_0+leng_1,leng_0+leng_1+leng_2),save_mat_loss_2[0,:],'g')
# plt.legend(labels = ['iter 0','iter 1','iter 2'],loc=1)
# plt.ylabel('mse')
# plt.xlabel('epoch')
# plt.title('boundary condition loss')
# plt.savefig("./pic/loss_bound.png")
# plt.show()

# plt.plot(range(leng_0),save_mat_loss_0[1,:],'r',range(leng_0,leng_1+leng_0),save_mat_loss_1[1,:],'b',range(leng_0+leng_1,leng_0+leng_1+leng_2),save_mat_loss_2[1,:],'g')
# plt.legend(labels = ['iter 0','iter 1','iter 2'],loc=1)
# plt.ylabel('mse')
# plt.xlabel('epoch')
# plt.title('equation loss')
# plt.savefig("./pic/loss_eq.png")
# plt.show()

# plt.plot(range(leng_0),save_mat_loss_0[2,:],'r',range(leng_0,leng_1+leng_0),save_mat_loss_1[2,:],'b',range(leng_0+leng_1,leng_0+leng_1+leng_2),save_mat_loss_2[2,:],'g')
# plt.legend(labels = ['iter 0','iter 1','iter 2'],loc=1)
# plt.ylabel('mse')
# plt.xlabel('epoch')
# plt.title('data loss')
# plt.savefig("./pic/loss_data.png")
# plt.show()

leng_0 = np.shape(save_mat_para_0)[1]
leng_1 = np.shape(save_mat_para_1)[1]
leng_2 = np.shape(save_mat_para_2)[1]

plt.plot(range(leng_0),save_mat_para_0[0,:],'r-',range(leng_0,leng_1+leng_0),save_mat_para_1[0,:],'b-',range(leng_0+leng_1,leng_0+leng_1+leng_2),save_mat_para_2[0,:],'g-',
         range(leng_0),save_mat_para_0[1,:],'r--',range(leng_0,leng_1+leng_0),save_mat_para_1[1,:],'b--',range(leng_0+leng_1,leng_0+leng_1+leng_2),save_mat_para_2[1,:],'g--',
         range(leng_0),save_mat_para_0[2,:],'r*',range(leng_0,leng_1+leng_0),save_mat_para_1[2,:],'b*',range(leng_0+leng_1,leng_0+leng_1+leng_2),save_mat_para_2[2,:],'g*',
         range(leng_0),save_mat_para_0[3,:],'r:',range(leng_0,leng_1+leng_0),save_mat_para_1[3,:],'b:',range(leng_0+leng_1,leng_0+leng_1+leng_2),save_mat_para_2[3,:],'g:',
         range(leng_0),save_mat_para_0[4,:],'r-.',range(leng_0,leng_1+leng_0),save_mat_para_1[4,:],'b-.',range(leng_0+leng_1,leng_0+leng_1+leng_2),save_mat_para_2[4,:],'g-.',
         range(leng_0+leng_1+leng_2),np.ones(leng_0+leng_1+leng_2),'k--')
plt.legend(labels = ['iter 0','iter 1','iter 2'],loc=1)
plt.ylabel('para value')
plt.xlabel('epoch')
plt.savefig("./pic/para.png")
plt.show()

# relative error

error_pinn = np.abs(1 - save_mat_para_0[:,-1])
error_mid = np.abs(1 - save_mat_para_1[:,-1])
error_modified = np.abs(1 - save_mat_para_2[:,-1])

relative_error = error_modified / error_pinn

relative_error_mid = error_mid / error_pinn

print(save_mat_para_0[:,-1])
print(save_mat_para_1[:,-1])
print(save_mat_para_2[:,-1])
print(error_pinn)
print(error_mid)
print(error_modified)
print(error_mid/error_pinn)
print(error_modified/error_pinn)