import numpy as np
import matplotlib.pyplot as plt
import torch
import os

def load_data(L,temp,MCS):
    return np.fromfile(f"../ising_wolff/dataIsing2D_L{L}/config_L{L}_T{temp:.3f}.bin",dtype = np.int32).reshape(MCS,L**2)

def gaussScore(x0,temp,nSteps,dt,device = "cpu"):
    x0 = torch.from_numpy(x0).to(torch.float32).to(device)
    N = x0.shape[1]
    W = torch.Tensor(nSteps+1,N,N)
    #b = torch.Tensor(nSteps+1,N)
    cov = torch.cov(x0.T)
    m0 = torch.mean(x0,dim=1)
    for i in range(nSteps+1):
        exp_t = torch.exp(torch.Tensor([-2*i*dt]))
        exp_tsq = exp_t*exp_t
        W[i] = torch.linalg.inv(temp*(1-exp_tsq)*torch.eye(N) + exp_tsq*cov)
        #b[i] = -exp_t* torch.mm(W[i],m0)
    return W

def backward(xT,W,temp,nSteps,dt,full_traj = False,device = "cpu"):
    P = xT.shape[0]
    N = xT.shape[1]
    nbatches = 50
    batch_size = int(P/nbatches)
    if(full_traj):
        x_recon = torch.Tensor(P,nSteps+1,N)
        x_recon[:,-1,:] = xT
    else:
        x_recon = xT
    for tt in range(nSteps+1)[::-1]:
        std = np.sqrt(2*temp*dt)*torch.randn_like(xT)

        for n in range(nbatches):
            if(full_traj):
                score = -torch.matmul(x_recon[n*batch_size:(n+1)*batch_size,tt,:],W[tt])
                x_recon[n*batch_size:(n+1)*batch_size,tt-1,:] = x_recon[n*batch_size:(n+1)*batch_size,tt,:]*(1+dt) + 2*temp*score*dt + std[n*batch_size:(n+1)*batch_size]
            else:
                score = -torch.matmul(x_recon[n*batch_size:(n+1)*batch_size],W[tt])
                x_recon[n*batch_size:(n+1)*batch_size] = x_recon[n*batch_size:(n+1)*batch_size]*(1+dt)+ 2*temp*score*dt + std[n*batch_size:(n+1)*batch_size]
            del score
    
    return x_recon

def Dkl(x0,x_recon):
    eps = 1e-05
    c1,tmp = np.histogram(np.mean(x0,axis = 1),bins = 21,range=(-1.25,1.25),density = True)
    c2,tmp = np.histogram(np.mean(x_recon,axis = 1),bins = 21,range=(-1.25,1.25),density = True)
    return np.sum(c1*np.log((c1+eps)/(c2+eps)))

L = 12
N = L**2
P = 50000
MCS = 200000
nSteps = 300
dt = 0.02
temp = 2
Ts = np.linspace(2.27,3.22,20)
Dkls = np.array([])
os.system(f"mkdir data_N{N}_T{temp:.3f}_P{P}")
for i,T in enumerate(Ts):
    x0 = load_data(L,T,MCS)
    W = gaussScore(x0[:P],temp,nSteps,dt)
    xT = np.sqrt(temp)*torch.randn((P,N))
    x_recon = backward(xT,W,temp,nSteps,dt,True).numpy()
    np.save(f"data_N{N}_T{temp:.3f}_P{P}/recon_temp{T:.3f}_nSteps{nSteps}_dt{dt:.2f}",x_recon[:,0,:])
    Dkls = np.append(Dkls,Dkl(x0,x_recon[:,0,:]))
    del x_recon
    #print(f"-----Done {i}-----")

plt.figure(figsize = (8,6))
plt.title(r"$D_{KL}(T)$")
plt.plot(Ts,Dkls)
plt.xlabel(r"$T$")
plt.ylabel(r"$D_{KL}$")
plt.savefig(f"data_N{N}_T{temp:.3f}_P{P}/dkls.pdf")
np.save(f"data_N{N}_T{temp:.3f}_P{P}/dkls",Dkls)
