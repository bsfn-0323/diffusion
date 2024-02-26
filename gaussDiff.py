import numpy as np
import matplotlib.pyplot as plt
import torch
import os

def load_data(L,temp,MCS):
    return np.fromfile(f"../ising_wolff/dataIsing2D_L{L}/config_L{L}_T{temp:.3f}.bin",dtype = np.int32).reshape(MCS,L**2)

def gaussScore(x0,temp,nSteps,dt,device = "cpu"):
    x0 = torch.from_numpy(x0).to(torch.float32).to(device)
    N = x0.shape[1]
    W = torch.Tensor(nSteps+1,N,N).to(device)
    #b = torch.Tensor(nSteps+1,N)
    cov = torch.cov(x0.T).to(device)
    m0 = torch.mean(x0,dim=1).to(device)
    for i in range(nSteps+1):
        exp_t = torch.exp(torch.Tensor([-2*i*dt])).to(device)
        exp_tsq = exp_t*exp_t
        W[i] = torch.linalg.inv(temp*(1-exp_tsq)*torch.eye(N).to(device) + exp_tsq*cov)
        #b[i] = -exp_t* torch.mm(W[i],m0)
    return W

def backward(xT,W,temp,nSteps,dt,full_traj = False,device = "cpu"):
    P = xT.shape[0]
    N = xT.shape[1]
    nbatches = 50
    batch_size = int(P/nbatches)
    if(full_traj):
        x_recon = torch.Tensor(P,nSteps,N).to(device)
        x_recon[:,-1,:] = xT.to(device)
    else:
        x_recon = xT.to(device)
    for tt in range(nSteps)[::-1]:
        std = np.sqrt(2*temp*dt)*torch.randn_like(xT).to(device)

        for n in range(nbatches):
            if(full_traj):
                score = -torch.matmul(x_recon[n*batch_size:(n+1)*batch_size,tt,:],W[tt])
                x_recon[n*batch_size:(n+1)*batch_size,tt-1,:] = x_recon[n*batch_size:(n+1)*batch_size,tt,:]*(1+dt) + 2*temp*score*dt + std[n*batch_size:(n+1)*batch_size]
            else:
                score = -torch.matmul(x_recon[n*batch_size:(n+1)*batch_size],W[tt])
                x_recon[n*batch_size:(n+1)*batch_size] = x_recon[n*batch_size:(n+1)*batch_size]*(1+dt)+ 2*temp*score*dt + std[n*batch_size:(n+1)*batch_size]
            del score
    
    return x_recon.to("cpu")

def Dkl(x0,x_recon):
    eps = 1e-05
    c1,tmp = np.histogram(np.mean(x0,axis = 1),bins = 21,range=(-1.25,1.25),density = True)
    c2,tmp = np.histogram(np.mean(x_recon,axis = 1),bins = 21,range=(-1.25,1.25),density = True)
    return np.sum(c1*np.log((c1+eps)/(c2+eps)))

L = 8

N = L**2
P = 100000
MCS = 200000
nSteps = 300
dt = 0.02
temp = 2
slices = np.linspace(1,nSteps,9,dtype = np.int32) -1
Ts = np.linspace(4.5,4.9,5)
Dkls = np.array([])
idx = np.random.choice(range(MCS),P, replace = False)
os.system(f"mkdir data_N{N}_T{temp:.3f}_P{P}")
for i,T in enumerate(Ts):
    x0 = load_data(L,T,MCS)
    W = gaussScore(x0[idx],temp,nSteps,dt,"cuda")
    xT = np.sqrt(temp)*torch.randn((P,N))
    x_recon = backward(xT,W,temp,nSteps,dt,False,"cuda").numpy()
    np.save(f"data_N{N}_T{temp:.3f}_P{P}/recon_temp{T:.3f}_nSteps{nSteps}_dt{dt:.2f}",x_recon)
    Dkls = np.append(Dkls,Dkl(x0[idx],x_recon))
    del x_recon
    #print(f"-----Done {i}-----")

plt.figure(figsize = (8,6))
plt.title(r"$D_{KL}(T)$")
plt.plot(Ts,Dkls)
plt.xlabel(r"$T$")
plt.ylabel(r"$D_{KL}$")
plt.savefig(f"data_N{N}_T{temp:.3f}_P{P}/dkls.pdf")
np.save(f"data_N{N}_T{temp:.3f}_P{P}/dkls",Dkls)
