
import numpy as np
from pylab import *
import matplotlib.pyplot as plt
from scipy.special import gamma, factorial
import pandas as pd
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import inv
import pickle
import argparse
#from MW_DIST import *
from NNC.GetTAUS import *

import random




def GEN_GS(nu,Mw,sig0,sig1,WinGp,WinGpp,SYMZ):
    Me = 12.96
    M=[125/Me,750/Me,2540/Me]
    M=np.array(M)
    nn=np.array([17.7,32,32.6244])
    gam = (nn[2]-nn[0])/np.log(M[2]/M[0])
    M0 = M[0]*np.exp(-nn[0]/gam)

    ZIM_LIST=[1000,1.0,0.4]
    ZIM_LIST[2]=nu
    SIG=[sig0,sig1]

    n_real=gam*np.log(Mw/M0)
    n_input1 = int(gam*np.log(Mw/M0))
    n_input2 = int(gam*np.log(Mw/M0))+1
    ratio1 = abs(n_real-n_input2)
    ratio2 = abs(n_real-n_input1)
    TAU_ALL1, zeta_out1 = Cal_TAUS(M,Me,nn,ZIM_LIST,SIG,n_input1,SYMZ)
    TAU_ALL2, zeta_out2 = Cal_TAUS(M,Me,nn,ZIM_LIST,SIG,n_input2,SYMZ)
    detxy=[0.0,0.0]

    #ni = ni*MWALL/sum(ni*MWALL)
    logwAGp=np.reshape(linspace(WinGp[0],WinGp[1],50),(1,50))
    logwAGpp=np.reshape(linspace(WinGpp[0],WinGpp[1],50),(1,50))
    Gp = np.zeros_like(logwAGp)
    Gpp = np.zeros_like(logwAGpp)
    WGp=10**logwAGp
    WsGp=np.reshape(WGp,(-1))
    WGpp=10**logwAGpp
    WsGpp=np.reshape(WGpp,(-1))
    i=0

    for TAU_ALL,zeta_out,ratio in [[TAU_ALL1,zeta_out1,ratio1],[TAU_ALL2,zeta_out2,ratio2]]:
        for n,MW,N,TAU3 in TAU_ALL:
            tau_Cnk = TAU3[0]
            tau_min=min(min(tau_Cnk),TAU3[1])
            tau_Cnk=tau_Cnk/tau_min
            Mw_RN=N
            tau=tau_Cnk
            Gp+=  ratio*np.sum(WGp**2*tau**2/(1+WGp**2*tau**2),axis=0)/Mw_RN
            Gpp+= ratio*np.sum(WGpp*tau/(1+WGpp**2*tau**2),axis=0)/Mw_RN



    return np.log10(WsGp),np.log10(Gp),np.log10(WsGpp),np.log10(Gpp)


def GEN_GS_X(nu,Mw,sig0,sig1,DX,DY,X1,X2,SYMZ):
    Me = 12.96
    M=[125/Me,750/Me,2540/Me]
    M=np.array(M)
    nn=np.array([17.7,32,32.6244])
    gam = (nn[2]-nn[0])/np.log(M[2]/M[0])
    M0 = M[0]*np.exp(-nn[0]/gam)

    ZIM_LIST=[1000,1.0,0.4]
    ZIM_LIST[2]=nu
    SIG=[sig0,sig1]

    n_real=gam*np.log(Mw/M0)
    n_input1 = int(gam*np.log(Mw/M0))
    n_input2 = int(gam*np.log(Mw/M0))+1
    ratio1 = abs(n_real-n_input2)
    ratio2 = abs(n_real-n_input1)
    TAU_ALL1, zeta_out1 = Cal_TAUS(M,Me,nn,ZIM_LIST,SIG,n_input1,SYMZ)
    TAU_ALL2, zeta_out2 = Cal_TAUS(M,Me,nn,ZIM_LIST,SIG,n_input2,SYMZ)


    detxy=[0.0,0.0]




    #ni = ni*MWALL/sum(ni*MWALL)
    logwAGp=np.reshape(X1-DX,(1,-1))
    logwAGpp=np.reshape(X2-DX,(1,-1))

    Gp = np.zeros_like(logwAGp)
    Gpp = np.zeros_like(logwAGpp)
    WGp=10**logwAGp
    WsGp=np.reshape(WGp,(-1))
    WGpp=10**logwAGpp
    WsGpp=np.reshape(WGpp,(-1))
    m_Rouse = 0 #2
    i=0

    for TAU_ALL,zeta_out,ratio in [[TAU_ALL1,zeta_out1,ratio1],[TAU_ALL2,zeta_out2,ratio2]]:
        for n,MW,N,TAU3 in TAU_ALL:
            tau_Cnk = TAU3[0]
            tau_min=min(min(tau_Cnk),TAU3[1])
            tau_Cnk=tau_Cnk/tau_min
            N_chain=N #np.sum(zeta_out)
            Mw_RN=N_chain
            tau=tau_Cnk
            Gp+=  ratio*np.sum(WGp**2*tau**2/(1+WGp**2*tau**2),axis=0)/Mw_RN
            Gpp+= ratio*np.sum(WGpp*tau/(1+WGpp**2*tau**2),axis=0)/Mw_RN



    return np.reshape(np.log10(Gp)+DY,(-1)),np.reshape(np.log10(Gpp)+DY,(-1))




def func_SIG(X):
    a, b, c,d,e=[1.02486875,0.35011296,0.20291189,-0.43373979,-0.33320135]
    if np.size(X)==1:
        x=X
        return a*(x-b+e*x**2)**c + d if x>0.42 else 0
    else:
        return [ a*(x-b+e*x**2)**c + d if x>0.42 else 0 for x in X]



def generate_dataset(n,m):
    X_data = []
    Y_data = []
    for _ in range(m):
        sig0=random.uniform(0.42,1.6)
        sig1_low=func_SIG(sig0)
        sig1=random.uniform(sig1_low,sig1_low+0.3)
        dsig1 = sig1-sig1_low
        nu = random.uniform(0.3,0.5)
        dnu=nu-0.3
        lnMw = random.uniform(np.log(5),np.log(130))
        dlnMw = lnMw - np.log(5)
        Mw = np.exp(lnMw)
        if Mw<10:
            WinGp=[random.uniform(-7.5,-5.8),random.uniform(-1,-0.0)]
        if Mw>=10 and Mw<20:
            WinGp=[random.uniform(-8,-6.5),random.uniform(-1,-0.0)]
        if Mw>=20 and Mw<50:
            WinGp=[random.uniform(-9,-7.5),random.uniform(-1,-0.0)]
        if Mw>=50 and Mw<100:
            WinGp=[random.uniform(-10,-8.8),random.uniform(-1,-0.0)]
        if Mw>=100:
            WinGp=[random.uniform(-10.5,-9.0),random.uniform(-1,-0.0)]

        WinGpp=[WinGp[0]+random.uniform(-0.2,0.2),WinGp[1]+random.uniform(-0.2,0.2)]

        W1,Gp,W2,Gpp = GEN_GS(nu,Mw,sig0,sig1,WinGp,WinGpp)
        W1,Gp,W2,Gpp =[np.reshape(W1,(-1)),np.reshape(Gp,(-1)),np.reshape(W2,(-1)),np.reshape(Gpp,(-1))]
        X = np.column_stack((W1, Gp, W2, Gpp))
        Y = [sig0, dsig1, dnu, dlnMw]
        X_data.append(X)
        Y_data.append(Y)
        if _ %100==0:
            print(_,  '  datas generated',n)

    return np.array(X_data), np.array(Y_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and save datasets.")
    parser.add_argument("-n", "--num_datasets", type=int, default=1, help="Number of datasets to generate.")

    args = parser.parse_args()
    n=args.num_datasets
    X, Y = generate_dataset(n,10000)
    filename = f"training_dataN{n}.pkl"
    with open(filename, 'wb') as f:
        pickle.dump({'X': X, 'Y': Y}, f)




    print(str(n)+"Training data has been saved!")
    X, Y = generate_dataset(2000)
    with open('test_dataN.pkl', 'wb') as f:
            pickle.dump({'X': X, 'Y': Y}, f)
            print("test data has been saved!")

#plt.loglog(GpW[0],np.reshape(GpW[1],(-1)),linewidth=3)
#plt.loglog(GppW[0],np.reshape(GppW[1],(-1)), linestyle='--',linewidth=3)
#plt.show()
