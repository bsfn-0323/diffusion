import numpy as np
import torch

def load_data(L,temp,MCS):
    return np.fromfile(f"../ising_wolff/dataIsing2D_L{L}/config_L{L}_T{temp:.3f}.bin",dtype = np.int32).reshape(MCS,L**2)

def Deltat(n,dt,diffTemp):
    return diffTemp*(1-np.exp(-2*(n+1)*dt))

def getScoreParams(x0,diffTemp,dt,nSteps):
    N = x0.shape[1]
    C0 = np.cov(x0.T)
    W = np.empty((nSteps,N,N))
    invW = np.empty((nSteps,N,N))
    M = np.empty((nSteps,N,N))
    K = np.empty((nSteps,N,N))
    ts= np.array([*range(nSteps)])
    Dts = Deltat(ts,dt,diffTemp)
    exp2ts = np.exp(-2*(ts+1)*dt)

    for t in range(nSteps):
        invW[t] = Dts[t]*np.eye(N) + C0*exp2ts[t]
        W[t] = np.linalg.inv(invW[t])
        M[t] = (3*Dts[t]+exp2ts[t])*C0*exp2ts[t]+ 3*Dts[t]*(Dts[t] + exp2ts[t])*np.eye(N)
    
    K = np.matmul(W,M) - np.eye(W.shape[1])
    K_g = -2*M + invW + np.matmul(M,np.matmul(W,M))

    C6 = Dts*(15*Dts**2 - 6*Dts + 1) + exp2ts *(45*Dts**2 - 12*Dts + 1) + exp2ts**2 * (15*Dts-2) + exp2ts**3
    C4 = 3*(Dts+exp2ts)-1
    num = np.trace(K,axis1=1,axis2=2) - N*C4
    den = np.trace(K_g,axis1=1,axis2=2) - N*C6

    g = -num/den
    A =  W + g.reshape(nSteps,1,1)*(np.matmul(M,W)-np.eye(N).reshape(1,N,N))

    return A,g

def backward(xT,A,g,temp,nSteps,dt,full_traj = False,every = 5,device = "cuda"):
    P = xT.shape[0]
    N = xT.shape[1]
    A =torch.from_numpy(A).to(device).to(torch.float32)
    g =torch.from_numpy(g).to(device).to(torch.float32)
    nbatches = 100
    batch_size = int(P/nbatches)
    tts = np.linspace(1,nSteps,nSteps-1,dtype = np.int32)
    if(full_traj):
        redSteps = int(nSteps/every)
        tslice = np.linspace(1,nSteps+1,redSteps,dtype = np.int32)
        x_recon = torch.Tensor(P,redSteps,N).to(device)
        x_now =xT.to(device)
    else:
        x_recon = xT.to(device)
    with torch.no_grad():
        for tt in tts[::-1]:
            std = np.sqrt(2*temp*dt)*torch.randn_like(xT).to(device)
            #if tt==1:
            #    std = torch.zeros(xT.shape)

            for n in range(nbatches):
                if(full_traj):
                    score = myscore(x_now[n*batch_size:(n+1)*batch_size,:], A[tt-1],g[tt-1],device)
                    x_now[n*batch_size:(n+1)*batch_size,:] = x_now[n*batch_size:(n+1)*batch_size,:]*(1+dt) + 2*temp*score*dt + std[n*batch_size:(n+1)*batch_size]
                    if(tt in tslice):
                        x_recon[n*batch_size:(n+1)*batch_size,np.where(tslice==tt)[0][0],:]=x_now[n*batch_size:(n+1)*batch_size,:]
                else:
                    score = myscore(x_recon[n*batch_size:(n+1)*batch_size], A[tt-1],g[tt-1],device)
                    x_recon[n*batch_size:(n+1)*batch_size] = x_recon[n*batch_size:(n+1)*batch_size]*(1+dt)+ 2*temp*score*dt + std[n*batch_size:(n+1)*batch_size]
                del score
    
    return x_recon.to("cpu").numpy()

def myscore(x,A,g,device = "cuda"):
    return  -torch.matmul(x,A.T) + g*x*(x**2-1)

Tmin = 2.27
Tmax = 3.22
meas = 20
Ts = np.linspace(Tmin,Tmax,meas)

L = 8
N = L**2
P = 100000
diffTemp = 2
nSteps = 300
dt = 0.02
x_recon = np.empty((meas,P,N))
for i,temp in enumerate(Ts):
    idx = np.random.choice(range(200000),P,replace = False)
    data = load_data(L,temp,200000)

    A,g = getScoreParams(data,diffTemp,dt,nSteps)

    xT = np.sqrt(diffTemp)*torch.randn((P,N))
    x_recon[i] = backward(xT,A,g,diffTemp,nSteps,dt,full_traj=False,device= "cpu")
np.save(f"x_recon_L{L}/x_recon_L{L}_Tmin{Tmin}_Tmax{Tmax}_difftemp{diffTemp}",x_recon)