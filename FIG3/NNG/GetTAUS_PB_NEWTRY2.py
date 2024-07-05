
import numpy as np
from pylab import *
import matplotlib.pyplot as plt
from scipy.special import gamma, factorial
import pandas as pd
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import inv
import pickle




def get_seq(n):
    # Generate the initial sequence
    seq = list(range(n))
    # Rearrange according to specified rules
    left_seq = seq[::2]  # Take every other element
    right_seq = seq[1::2][::-1]  # Take the remaining elements and reverse them
    # Concatenate the two parts
    seq = left_seq + right_seq
    return seq
def Cnk(n,k):
    return gamma(n+1)/gamma(k+1)/gamma(n-k+1)

def Cnk_al(n,k,al=1):
    if al==1: return Cnk(n,k)
    if al!=1:
        if k==0: C = 1
        if k<=n/2 and k>0:
            C=0
            for i in range(k,n+1):
                C+= al**i*Cnk(i-1,k-1)

        if k>n/2:
            return Cnk_al(n,n-k,al)
        return C

def Cnk_ratio(n,k,ratio=1):
    return Cnk(n,k)**ratio

"""
For big n, cnk will be very large.
Stirling formula will be applied to cal cnk
"""

def Cnk_ratio_bign(n,k,ratio=1):
    logCnk = (stir(n)-stir(k)-stir(n-k))*ratio
    return np.exp(logCnk)

def stir(n): # use stirling formula to approximate gamma function
    if n>20:
        det = np.log(1 + 1/(12*n) + 1/(288*n**2) - 139/(51840*n**3) - 571/(2488320*n**4))
        return n*np.log(n)-n+0.5*np.log(2*np.pi*n)+det
    else:
        return np.log(gamma(n+1))

def Get_Zeta_Sig(zetaALL,zeta0,n,ne,sig,dist_type=1,detm=1):
    N=int(n*ne)+1
    zeta_sig =np.zeros(N,dtype=np.float64)
    for i in range(n*ne+1):
        for j in range(n+1):
            kj = j*ne
            if dist_type==1:
                zeta_sig[i]+= zetaALL[j]*np.exp(-(i-kj)**2/2/sig**2)
            if dist_type==2:
                zeta_sig[i]+= np.log(zetaALL[j])*np.exp(-(i-kj)**2/2/sig**2)
        if dist_type ==2:
            zeta_sig[i]=np.exp(zeta_sig[i])-1.0
    if dist_type==3:
        for i in range(n+1):
            for j in range(detm):
                i1 = i*ne-detm//2 +j
                if i1>=0 and i1<=n*ne:
                    zeta_sig[i1]=zetaALL[i]

    return zeta_sig

def Zeta_Dist(Max_Zeta,n,k,sig):
    max_ln = np.log(Max_Zeta)
    zeta_val = max_ln*np.exp(-(k-n/2)**2/2/sig**2)
    return np.exp(zeta_val)

def sumC_n_j(nn0,sig0,sig1,Max_Zeta):
    sumC=0
    for i in range(nn0):
        cnk=Cnk_ratio_bign(nn0,i,1)
        sumC+=Zeta_Dist(Max_Zeta,nn0,i,sig0)**(1-sig1)*cnk**sig1

    return sumC

def get_alI_II(nn0,sig0,sig1):
    Max_Zeta = Cnk_ratio_bign(nn0,nn0//2,1)
    tau_max_rouse1 = (12-2)**2/np.pi**2
    tau_max_rouse2 = nn0**2*(12-2)**2/np.pi**2
    sigr0=sig0*nn0
    al_II = Zeta_Dist(Max_Zeta,nn0,0,sigr0)**(1-sig1)/tau_max_rouse2
    al_III =Max_Zeta/tau_max_rouse2
    sumC=sumC_n_j(nn0,sigr0,sig1,Max_Zeta)
    al_I = sumC/2**nn0

    return al_I,al_II,al_III

def solve_sig_beta(al_I,al_II,n):
    er0 = 100.0
    Max_Zeta = Cnk_ratio_bign(n,n//2,1)
    tau_max_rouse1 = (24-2)**2/np.pi**2
    tau_max_rouse2 = n**2*(24-2)**2/np.pi**2
    SIG_ALL = [i / 100.0 for i in range(10, 101, 5)] + [i / 10.0 for i in range(10, 20)] + [i / 100.0 for i in range(200, 2610, 25)]
    for i in range(40):
        bet= i*0.025
        for sig in SIG_ALL:
            sigr=sig*n
            al_II_t = Zeta_Dist(Max_Zeta,n,0,sigr)**(1-bet)/tau_max_rouse2
            sumC=sumC_n_j(n,sigr,bet,Max_Zeta)
            al_I_t=sumC/2**n
            er = (np.log(al_I)-np.log(al_I_t))**2+(np.log(al_II)-np.log(al_II_t))**2
            if er<er0:
                sig1_min=bet
                sig_min=sig
                er0=er
                al_I_min=al_I_t
                al_II_min=al_II_t
    return sig_min,sig1_min

def solve_sigNEW(al_I,al_II,n,bet):
    er0 = 100.0
    Max_Zeta = Cnk_ratio_bign(n,n//2,1)
    tau_max_rouse1 = (12-2)**2/np.pi**2
    tau_max_rouse2 = n**2*(12-2)**2/np.pi**2
    SIG_ALL = [i / 100.0 for i in range(1, 101, 1)] + [i / 100.0 for i in range(100, 200,2)] + [i / 100.0 for i in range(200, 2610, 5)]
    #SIG_ALL=SIG_ALL+[i for i in range(27, 100)]
    #SIG_ALL =  [i / 100.0 for i in range(200, 2610, 5)]
    for sig in SIG_ALL:
        #al_II_t = Zeta_Dist(Max_Zeta,n,0,sig)**(1-bet)/tau_max_rouse2
        sigr=sig*n
        sumC=sumC_n_j(n,sigr,bet,Max_Zeta)
        al_I_t=sumC/2**n
        er = (np.log(al_I)-np.log(al_I_t))**2
        if er<er0:
            sig1_min=bet
            sig_min=sig
            er0=er
            al_I_min=al_I_t
    return sig_min,er0


def MSEG(Gp,Gpp,YG1,YG2):
    return np.mean((Gp - YG1) ** 2)+np.mean((Gpp - YG2) ** 2)


def Cal_TAUS(gam,M0,ZIM_LIST,SIG,n_heavy):
    n0=np.log(1/M0)*gam



    Approx_ALL=['ROUSE_NONE','ROUSE_ON','ZIMM_v']
    Approx_type=Approx_ALL[0]
    NU_ZIMM = 3/5
    ratio = 1
    zeta0 = 1.00

    MN_MAX,NU,ALPH =ZIM_LIST


    ker_ALL=[[1],[1,-1],[-1,2,-1],[-1,-2,6,-2,-1],[-1,2,-3,4,-3,2,-1]]
    ker_LR =[[],
            [],
            [ [ [1/2,-1/2] ], [[-1/2,1/2]]  ],
            [ [ [3/6,-2/6,-1/6],[-1/6,4/6,-2/6,-1/6] ],[[-1/6,-2/6,3/6],[-1/6,-2/6,4/6,-1/6] ]  ],
            [[ [2/4,-3/4,2/4,-1/4],[-1/4,3/4,-3/4,2/4,-1/4],[1/4,-2/4,3/4,-3/4,2/4,-1/4] ],
            [ [-1/4,2/4,-3/4,2/4],[-1/4,2/4,-3/4,3/4,-1/4],[-1/4,2/4,-3/4,3/4,-2/4,1/4]   ]]  ]


    K=2
    DIST_TYPE=3
    detm_values = [1, 3, 5, 7, 9]  # List of detm values
    sig_values = [0.2,0.5,0.8]
    K_Values=[0,1,2,3,4]
    ratio_values=[0.7,1,0.8,1,0.9,1.0]
    m_values=[0,0.5,1.0,1.5,2.0,3.0]

    n_values=[30]
    ne_values=[4]
    colors = ['b', 'g', 'r', 'c', 'm','k']  # Corresponding colors for each detm value

    detxy=[5.35+0.105,4.05-0.1]



    detm=1

    m_rouse =2

    eta_ALL=[]
    MW_ALL=[]
    #gam=1/5.15


    """
    当n0=5.813时,zeta0=1
    Me0=n0*4
    其它n时，先求出MW
    zeta0n= zeta0*MW*Me0/n/ne
    """


    TAU_ALL=[]

    for  n in range(n_heavy,n_heavy+1):
        #n = int((MW/M0)**gam)
        #n = int(gam*np.log(MW/M0))
        MW=np.exp(n/gam)*M0
        zeta0n= zeta0*MW*n0/n
        lam = MW*n0/n
        zeta0n= zeta0
        k0 = 1.0
        #zeta0n=zeta0
        ker = ker_ALL[K]
        ker = ker/np.max(ker)
        kerED =ker_LR[K]
        ne=4*3
        N=int(n*ne)+1
        """
         SYMZ
         """
        nn = n
        nne = ne
        J = N - nn*nne - 1
        zeta = np.ones(N,dtype=np.float64)/zeta0

        """
           TAU_Cnk
        """
        zetaALL =[]
        Max_Zeta = Cnk_ratio_bign(n,n//2,ratio)
        sig = SIG[0]*n
        al = SIG[1] #0.25
        zeta_end =1 # 1/4/np.sin(np.pi/2/nne)**2
        for k in range(n+1):
            c1=Cnk_ratio_bign(n,k,ratio)*zeta0
            c2=Zeta_Dist(Max_Zeta,n,k,sig)*zeta0
            #c2=Max_Zeta*zeta0
            c = np.exp(np.log(c1)*al+np.log(c2)*(1-al))
            #c = Cnk_ratio_bign(n,k,ratio)*zeta_end
            zetaALL.append(c)
            zeta[k*ne]= 1/c #gamma(k+1)*gamma(n-k+1)/gamma(n+1)
        """
        SYMZ
        """
        zetaALL_SYMZ = zetaALL
        #seq=get_seq(nn)
        #for k in seq:
        #    zetaALL_SYMZ.append(zetaALL[k])


        zeta_sig = Get_Zeta_Sig(zetaALL,zeta0,n,ne,sig,dist_type=DIST_TYPE,detm=detm)

        Z_leftM = np.eye(N, dtype=np.float64)*zeta0n
        # Modify Z_leftM
        for i in range(N):
            if len(ker)==1:
                Z_leftM[i,i]+=zeta_sig[i]
            else:
                if i>=len(kerED[0]) and N-1-i>=len(kerED[0]):
                    for j in range(len(ker)):
                        Z_leftM[i,i-len(ker)//2+j] += ker[j]*zeta_sig[i]
                if i<len(kerED[0]):
                    for j in range(i+1+len(ker)//2):
                        Z_leftM[i,j]+=kerED[0][i][j]*zeta_sig[i]
                if N-1-i< len(kerED[0]):
                    for j in range(N-1-i+1+len(ker)//2):
                        Z_leftM[i,i-len(ker)//2+j]+=kerED[1][N-1-i][j]*zeta_sig[i]

        Z_leftM_SYMZ = np.eye(N, dtype=np.float64)*zeta0n
        for i in range(nn):
            for j in range(len(ker)):
                I = i*nne+1 - len(ker)//2+j
                Z_leftM_SYMZ[i*nne+1,I] = ker[j]*zetaALL_SYMZ[i]
                Z_leftM_SYMZ[I,i*nne+1] = Z_leftM_SYMZ[i*nne+1,I]
                if I!=i*nne+1:
                    Z_leftM_SYMZ[I,I] = - Z_leftM_SYMZ[i*nne+1,I]+zeta0
                if I==i*nne+1:
                    Z_leftM_SYMZ[i*nne+1,I] = ker[j]*zetaALL_SYMZ[i] +zeta0

        """
        ******** IMPORTANT *****************
        Replacing Z_leftM_SYMZ with Z_leftM will make the program back to the non-symz mode.
        """

        Z_leftM_sparse = csc_matrix(Z_leftM_SYMZ)
        Z_leftM_inverse = inv(Z_leftM_sparse).todense()




        Z=np.zeros((N,N),dtype=np.float64)

        Z[0,0],Z[0,1],Z[N-1,N-1],Z[N-1,N-2] = [-k0,k0,-k0,k0]

        for i in range(1,N-1):
            Z[i,i-1],Z[i,i],Z[i,i+1]=[1*k0,-2*k0,1*k0]


        Hmn = np.eye(N, dtype=np.float64)*zeta0n
        for i in range(N):
            for j in range(N):
                if i==j:
                    Hmn[i,j]=1
                if i!=j and abs(i-j)<MN_MAX:
                    Hmn[i,j]=ALPH/(abs(i-j))**NU
        Z=Hmn @ Z
        Z = Z_leftM_inverse @ Z

        # Get the relaxation times
        w, v = np.linalg.eig(Z)
        w=-w
        w=np.sort(np.real(w))


        tau_Cnk=np.reshape(1/w[1:N],(N-1,1))*lam**2
        tau_Cnk0=tau_Cnk[0]

        mintau = lam**2/4.0

        tau0=tau_Cnk0
        p=np.reshape(np.array(range(N-1))+1,(N-1,1))


        logwA=np.reshape(linspace(-6-detxy[0],5-detxy[0],50),(1,50))
        W=10**logwA
        Ws=np.reshape(W,(-1))



        MW_ALL.append(np.exp(n/gam)*M0)
        TAU_ALL.append([n,np.exp(n/gam)*M0,N,[tau_Cnk,mintau]])

    return TAU_ALL,np.array(1/zeta)
