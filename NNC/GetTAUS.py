
import numpy as np
from pylab import *
import matplotlib.pyplot as plt
from scipy.special import gamma, factorial
import pandas as pd
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import inv
import pickle


def plot_zeta(zeta,n,ne):
    x = np.arange(0,n*ne+1)
    y=np.log(zeta)

    fig, ax = plt.subplots(figsize=(8, 4))
    plt.bar(x, y+0.2,color='tab:blue',width=0.8,bottom=-0.2)

    #plt.title('$ln\zeta_j/\zeta_b\propto \exp[-(j-n/2)^2/2\sigma^2]$',fontsize=20)
    #plt.title('$\zeta_j/\zeta_b $',fontsize=15)

    plt.xticks(fontsize=20)
    plt.xticks(fontsize=10)
    plt.ylim(-0.20,max(y)+1)

    plt.xlabel("Beads' Position (j)",fontsize=13)
    plt.ylabel('$ln(\zeta_j/\zeta_b)$',fontsize=25)
    #plt.savefig('Dist_Zeta_Assoc.png', dpi=600)
    plt.savefig('Dist_Zeta_Cnk.png', dpi=600)
    plt.close()



def Cal_TAUS(M,M0,nn0,ZIM_LIST,SIG,n_heavy,SYMZ):
    gam = (nn0[2]-nn0[0])/np.log(M[2]/M[0])
    M0 = M[0]*np.exp(-nn0[0]/gam)

    n0=np.log(1/M0)*gam
    def get_seq(n):
        # 生成初始序列
        seq = list(range(n))
        # 按照指定规则重排
        left_seq = seq[::2]  # 隔一个取一个
        right_seq = seq[1::2][::-1]  # 取剩下的，并反转
        # 拼接两部分
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

    def stir(n):
        if n>20:
            det = np.log(1 + 1/(12*n) + 1/(288*n**2) - 139/(51840*n**3) - 571/(2488320*n**4))
            return n*np.log(n)-n+0.5*np.log(2*np.pi*n)+det
        else:
            return np.log(gamma(n+1))


    def Get_Zeta_Sig(zetaALL,zeta0,n,ne,sig,dist_type=1,detm=1):
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
        #SIG=sig/n
        #zeta_val = max_ln*np.cos(np.pi*(k-n/2)*SIG/n)
        return np.exp(zeta_val)

    ROUSE_NUMERICAL= True
    #Approx_type='ROUSE_NONE'
    #Approx_type='ROUSE_ON'
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


    ne=12
    detm=1
    sig=0.1

    eta_ALL=[]
    MW_ALL=[]
    """
    当n0=5.813时,zeta0=1
    其它n时，先求出MW
    zeta0n= zeta0*MW*n0/n
    """
    TAU_ALL=[]

    for  n in range(n_heavy,n_heavy+1):
        MW=np.exp(n/gam)*M0
        zeta0n= zeta0*MW*n0/n
        lam = MW*n0/n
        zeta0n= zeta0
        k0 = 1.0
        #zeta0n=zeta0
        ker = ker_ALL[K]
        ker = ker/np.max(ker)
        kerED =ker_LR[K]
        N=int(n*ne)+1
        """
         SYMZ
         """
        nn = n//2
        nne = ne*2
        J = N - nn*nne - 1
        zeta = np.ones(N,dtype=np.float64)/zeta0

        """
           TAU_Cnk
        """
        zetaALL =[]
        Max_Zeta = Cnk_ratio_bign(n,n//2,ratio)
        sig = SIG[0]*n
        al = SIG[1] #0.25
        for k in range(n+1):
            c1=Cnk_ratio_bign(n,k,ratio)*zeta0
            c2=Zeta_Dist(Max_Zeta,n,k,sig)*zeta0
            c = np.exp(np.log(c1)*al+np.log(c2)*(1-al))
            zetaALL.append(c)
            zeta[k*ne]= 1/c #gamma(k+1)*gamma(n-k+1)/gamma(n+1)
        """
        SYMZ
        """
        zetaALL_SYMZ = []
        seq=get_seq(nn)
        #print('seq,nn:',seq,nn)
        #a=input('check seq')
        for k in seq:
            zetaALL_SYMZ.append(zetaALL[k])


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

        if SYMZ==True:Z_leftM_sparse = csc_matrix(Z_leftM_SYMZ)
        if SYMZ==False:Z_leftM_sparse = csc_matrix(Z_leftM)
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


        tau_Cnk=np.reshape(1/w[1:N],(N-1,1)) *lam**2
        tau_Cnk0=tau_Cnk[0]

        mintau=lam**2.0/4
        tau0=tau_Cnk0
        p=np.reshape(np.array(range(N-1))+1,(N-1,1))
        logwA=np.reshape(linspace(-6-detxy[0],5-detxy[0],50),(1,50))
        W=10**logwA
        Ws=np.reshape(W,(-1))


        MW_ALL.append(np.exp(n/gam)*M0)
        TAU_ALL.append([n,np.exp(n/gam)*M0,N,[tau_Cnk,mintau]])

    return TAU_ALL,np.array(1/zeta)
