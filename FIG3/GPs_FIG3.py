
import numpy as np
from pylab import *
import matplotlib.pyplot as plt
from scipy.special import gamma, factorial
import pandas as pd
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import inv
import pickle
#from MW_DIST import *
#from NNG.GetTAUS_PB_NEWTRY1 import *
from NNG.GetTAUS_PB_NEWTRY2 import * # FIX the problems of sig
import pickle
import argparse
from pylab import *
import ast
from cycler import cycler
from scipy.special import erf
from scipy.optimize import fsolve
from matplotlib.ticker import AutoMinorLocator



# 使用matplotlib的tab10颜色系列
tab_colors = plt.cm.tab10.colors
# 为每种颜色提供两个连续条目
doubled_colors = [color for color in tab_colors for _ in range(2)]
# 设置自定义颜色循环
plt.rcParams['axes.prop_cycle'] = cycler(color=doubled_colors)

Marker_ALL = ['o','v','^','<','>','s','p','*','H','+','x','D','P','X']

def get_m0_gam():
    Me = 12.96
    M=[290/Me,750/Me,2540/Me]
    nn=np.array([29,32,41])
    M=[125/Me,750/Me,2540/Me]
    nn=np.array([17.7,32,32.6244])
    gam = (nn[2]-nn[0])/np.log(M[2]/M[0])
    M0 = M[0]*np.exp(-nn[0]/gam)
    return gam,M0


def convert_molecular_weight(value):
    # 如果值中包含'K'或者'k', 则乘以1000转换
    if 'k' in value or 'K' in value:

        return float(value[:-1])
    return float(value)/1000


parser = argparse.ArgumentParser()

# for PB92
parser.add_argument('-i', '--input', type=str, default="PB92_3SYM", help="Input file prefix")
parser.add_argument('-f', type=str, nargs='+', default=['PBGP92.csv'], help="Input experimental data")
parser.add_argument('-M', type=str, nargs='+', default=['20.7k','44.1k','97k','201k'], help="Molecular weights")
parser.add_argument('-c', '--calibr', type=str, default='97k', help="the curve for calibration")
dGK=[0,0,0,0]
"""
# for PS6
parser.add_argument('-i', '--input', type=str, default="PS6_4SYM", help="Input file prefix")
parser.add_argument('-f', type=str, nargs='+', default=['PSGP6.csv'], help="Input experimental data")
parser.add_argument('-M', type=str, nargs='+', default=['125k','290k','750k','2740k' ], help="Molecular weights")
parser.add_argument('-c', '--calibr', type=str, default='2740k', help="the curve for calibration")
dGK=[0.07,0.10,-0.05,0]
"""

parser.add_argument('-B', '--bd', type=str, default='0',nargs='+', help="A list of omega's min max: [2,7] [3,9] ...")
parser.add_argument('-dw', '--dw', type=float, default=0, help="translation of omega")
parser.add_argument('-dG', '--dG', type=float, default=0, help="translation of modulus")

# dw=8,dG=6.8 for PB92
# dW = 5.5, dG=6 for PS6

args = parser.parse_args()
dw=args.dw;dG=args.dG

bds = [ast.literal_eval(l) for l in args.bd]

#parser.add_argument('-s', '--optstep', type=float, default=1.0, help="size of optimized step")
#parser.add_argument('-th', '--thres', type=float, default=0.001, help="size of optimized step")

Polytype = args.input
filenames = [f for f in args.f]
Mws = [convert_molecular_weight(mw) for mw in args.M]
cali=args.calibr


def Get_Sig0(al,bet,x0,Mw,Me0):
    M =Mw/Me0
    sig0= al*(np.log(M)-np.log(x0))**2+bet
    if sig0>2.0:
        sig0=2.0
    #if sig0>1.3:
    #    sig0=1.3
    return sig0

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
def sig0_pb(x):
    a,b,c,k=[3.5726,1.6167,0.544,0.819]
    return  k * gelu(a * (x - b)) + c
def equations(kk,x1,y1):
    x0 =1.4
    return y1-sig0_pb(kk*(x1-x0)+x0)
def solve_sig0(x1,y1):
    p0=1.0
    p_opt, infodict, ier, msg = fsolve(equations, p0,args=(np.log10(x1),y1), full_output=True)
    kk=p_opt
    return kk


def func_SIG1(X):
    coef=[ 5.69754625e+05, -1.03412085e+06,  7.55272600e+05, -2.82833320e+05,
        5.72400639e+04, -6.02050980e+03,  2.86368746e+02, -3.83948161e+00,
        4.23837411e-01]
    coefr=coef
    a=coef[-1]
    coefr[-1]=a-X
    roots = np.roots(coefr)
    real_roots_in_range = [root.real for root in roots if root.imag == 0 and 0 <= root.real <= 0.6]
    if real_roots_in_range:
        y=min(real_roots_in_range)+0.03
    else:
        y=0
    return y

def func_SIG(X):
    a, b, c,d,e=[1.02486875,0.35011296,0.20291189,-0.43373979,-0.33320135]
    if np.size(X)==1:
        x=X
        return a*(x-b+e*x**2)**c + d if x>0.42 else 0
    else:
        return [ a*(x-b+e*x**2)**c + d if x>0.42 else 0 for x in X]



def Get_Nu(Nu,Mw_true,Mw,Me):
    det_nu=0.15
    lnM1=np.log10(5);lnM2=np.log10(100)
    lnMM =(lnM1+lnM2)*0.5
    c0=Nu-np.tanh( (np.log10(Mw_true)-lnMM)/(lnM2-lnM1) )*det_nu
    return np.tanh( (np.log10(Mw/Me)-lnMM)/(lnM2-lnM1) )*det_nu +c0



with open('data/'+Polytype + 'PARA.pkl', 'rb') as f:
    para_dict = pickle.load(f)

Nu = para_dict['Nu']  #alpha coeff for |i-j|^\alpha
Me_rate = para_dict['Me_0/Mw_0'] #'Me_0/Mw_0': np.exp(PARA[1])/12.96/Mw_Real,
sig0=para_dict['Sig0'] # sig for Gauss
sig1=para_dict['Sig1'] # beta for ...
dx = para_dict['DX']
dy = para_dict['DY']
Mw_true =para_dict['Mw_true']
detxy=[dx,dy]

def get_defaultBD_Mw(Mws,detxy):
    Mws_true =np.array( [mw*Me_rate for mw in Mws])
    Mw0=17.0 # -7
    detW = np.log(Mws_true/Mw0)
    W_low = -detW*1.4-6.75+detxy[0]-np.log10(4)
    bds =[ [W_low[i],detxy[0]-0.7] for i in range(detW.shape[0])]
    return bds



if bds[0]==0:
    bds = get_defaultBD_Mw(Mws,detxy)




Me = 12.96

def read_gp(filename):
    WGP = pd.read_csv('data/'+filename, header=None)
    for start_row in range(WGP.shape[0]):
        try:
            float(WGP.iloc[start_row, 0])
            break
        except ValueError:
            pass
    # Convert everything that can't be turned into a float to NaN
    def to_float_or_nan(val):
        try:
            return float(val)
        except:
            return np.nan

    data = WGP.iloc[start_row:, :].applymap(to_float_or_nan).values
    # Trim data for each column individually based on the first NaN occurrence
    cols_trimmed = []
    for col in data.T:  # Transpose to iterate over columns
        valid_idx = np.where(np.isnan(col))[0]
        if valid_idx.size > 0:
            first_invalid = valid_idx[0]
            cols_trimmed.append(col[:first_invalid])
        else:
            cols_trimmed.append(col)
    return cols_trimmed,WGP




GP_EXP=[]
GPP_EXP=[]
for filename in filenames:
    cols_trimmed,WGP =read_gp(filename)
    N_EXP = WGP.shape[1]//4
    for i in range(N_EXP):
        GP_EXP.append([cols_trimmed[i*4],cols_trimmed[i*4+1]])
        GPP_EXP.append([cols_trimmed[i*4+2],cols_trimmed[i*4+3]])




K=0
#Mps =[34.5,61.2,115,260,670,3200]
#Mws =[34,57.16,125,290,750,2740]
#Me = 2.2cols_trimmed[i*4+1]

Mws_rescale=[m*Me_rate for m in Mws]
Me = Mws[0]/Mws_rescale[0]



ZIM_LIST=[1000,1,0.4]
ZIM_LIST[2] = Nu

M25,sig0_25=[20,0.40]
alph = (sig0-sig0_25)/(np.log(Mw_true)-np.log(M25))**2
bet =sig0_25
det_sig1 = sig1 - func_SIG1(sig0)-(Mw_true-30)/120*0.08
#Det_Sig = Det_Sig*(Det_Sig>0)
sig0_TYPE = 'GELU'
slope_sig0= (sig0-0.42)/((np.tanh((np.log(Mw_true)-4.5))+1)/2.0)
if sig0>0.42 and np.log10(Mw_true)>1.5:
    kk=solve_sig0(Mw_true,sig0)
else:
    sig0_TYPE = 'slope'

gam,M0=get_m0_gam()
K=0;KK=0
GPS_TH=[]
plt.close()
nn0=int(gam*np.log(Mw_true/M0))+1

print(Mw_true,M0,Me)

al_I,al_II,al_III = get_alI_II(nn0,sig0,sig1)
print(al_I,al_II)

for Mstr,Mw,bds_M in zip(args.M, Mws,bds):
    #SIG0=Get_Sig0(alph,bet,M25,Mw,Me)

    n_real=gam*np.log(Mw/M0/Me)

    n_input1 = int(gam*np.log(Mw/M0/Me))
    n_input2 = int(gam*np.log(Mw/M0/Me))+1
    #al_I,al_II,al_III = get_alI_II(n_input2,SIG0,SIG1)
    SIG0,SIG1=solve_sig_beta(al_I,al_II,n_input2)
    SIG=[SIG0,SIG1]
    print(Mw_true,Me,Mw)
    print(SIG0,SIG1,sig0_TYPE)
    # draw the plot
    ratio1 = abs(n_real-n_input2)
    ratio2 = abs(n_real-n_input1)

    """
    adjust SIG0 and SIG1
    """
    Xp = np.reshape(np.array(GP_EXP[K][0]),(1,-1))
    Yp = np.reshape(np.array(GP_EXP[K][1]),(-1))-dGK[K]

    Xpp = np.reshape(np.array(GPP_EXP[K][0]),(1,-1))
    Ypp = np.reshape(np.array(GPP_EXP[K][1]),(-1))-dGK[K]
    def GET_GPGPP(SIGt,n_input1,n_input2):
        Gp = np.zeros_like(Xp)
        Gpp = np.zeros_like(Xpp)
        TAU_ALL1, zeta_out1 = Cal_TAUS(gam,M0,ZIM_LIST,SIGt,n_input1)
        TAU_ALL2, zeta_out2 = Cal_TAUS(gam,M0,ZIM_LIST,SIGt,n_input2)
        for TAU_ALL,zeta_out,ratio in [[TAU_ALL1,zeta_out1,ratio1],[TAU_ALL2,zeta_out2,ratio2]]:
            for n,MW,N,TAU3 in TAU_ALL:
                tau_Cnk = TAU3[0]
                mintau =TAU3[1]
                tau_min=min(min(tau_Cnk),mintau)
                tau_Cnk=tau_Cnk/tau_min/10**detxy[0]
                N_chain=N/10**detxy[1]
                Mw_RN=N_chain
                tau=tau_Cnk
                Gp+=  ratio*np.sum(Wp**2*tau**2/(1+Wp**2*tau**2),axis=0)/Mw_RN
                Gpp+= ratio*np.sum(Wpp*tau/(1+Wpp**2*tau**2),axis=0)/Mw_RN
        Gp=np.reshape(Gp,(-1))
        Gpp=np.reshape(Gpp,(-1))
        return Gp,Gpp
    def GET_dSJ(SIGJ,J):
        dSJ=0.0
        if J==0:
            if SIGJ<1:dSJ=0.05
            if SIGJ>=1 and SIGJ<10:dSJ=0.25
            if SIGJ>10:dSJ=1.0
        if J==1:
            dSJ = 0.05
        return dSJ

    Wp=10**Xp;Wpp=10**Xpp
    MSE0=10000000.0
    SIG_INIT=[SIG0,SIG1]
    for i in range(200):
        J = np.random.randint(0, 2)
        SIGJ=SIG_INIT[J]
        dSJ=GET_dSJ(SIGJ,J)
        dS = dSJ*(np.random.rand()-0.5)*2
        if SIGJ+dS<0 or (SIGJ+dS>1 and J==1):
            dS=0
        SIG_INIT[J]+=dS
        Gp,Gpp=GET_GPGPP(SIG_INIT,n_input1,n_input2)
        MS=MSEG(np.log10(Gp),np.log10(Gpp),Yp,Ypp)
        if MSE0 <  MS:
            SIG_INIT[J]=SIGJ
        else:
            MSE0 =MS

    #plt.loglog(10**GP_EXP[K][0]*10**(-dw),Gpt*10**(-dG),color='k',linewidth=2)
    #plt.loglog(10**GP_EXP[K][0]*10**(-dw),Gppt*10**(-dG),'--',color='k',linewidth=2)
    #plt.loglog(10**GP_EXP[K][0]*10**(-dw), 10**GP_EXP[K][1]*10**(-dG-dGK[K]), marker=Marker_ALL[K],markersize=4, linestyle='', color='k',  markerfacecolor='white')
    #plt.loglog(10**GPP_EXP[K][0]*10**(-dw), 10**GPP_EXP[K][1]*10**(-dG-dGK[K]), marker=Marker_ALL[K],markersize=4, linestyle='', color='k', markerfacecolor='white')
    print(MSE0,SIG_INIT)
    SIG=[SIG_INIT[0],SIG_INIT[1]]
    al_It,al_IIt,al_III = get_alI_II(n_input1,SIG[0],SIG[1])
    print(al_It,al_IIt)
    al_It,al_IIt,al_III = get_alI_II(n_input2,SIG[0],SIG[1])
    print(al_It,al_IIt)



    if cali==Mstr:
        SIG=[para_dict['Sig0'],para_dict['Sig1']]
        print(SIG)


    TAU_ALL1, zeta_out1 = Cal_TAUS(gam,M0,ZIM_LIST,SIG,n_input1)
    TAU_ALL2, zeta_out2 = Cal_TAUS(gam,M0,ZIM_LIST,SIG,n_input2)

    #ni = ni*MWALL/sum(ni*MWALL)

    logwA=np.reshape(linspace(bds_M[0],bds_M[1],100),(1,100))
    Gp = np.zeros_like(logwA)
    Gpp = np.zeros_like(logwA)
    W=10**logwA
    Ws=np.reshape(W,(-1))
    m_Rouse = 2


    for TAU_ALL,zeta_out,ratio in [[TAU_ALL1,zeta_out1,ratio1],[TAU_ALL2,zeta_out2,ratio2]]:
        for n,MW,N,TAU3 in TAU_ALL:
            tau_Cnk = TAU3[0]
            mintau =TAU3[1]
            tau_min=min(min(tau_Cnk),mintau)
            tau_Cnk=tau_Cnk/tau_min/10**detxy[0]
            N_chain=N/10**detxy[1]
            Mw_RN=N_chain
            tau=tau_Cnk

            Gp+=  ratio*np.sum(W**2*tau**2/(1+W**2*tau**2),axis=0)/Mw_RN
            Gpp+= ratio*np.sum(W*tau/(1+W**2*tau**2),axis=0)/Mw_RN


    Gp=np.reshape(Gp,(-1))
    Gpp=np.reshape(Gpp,(-1))
    colorm='k'
    if cali==Mstr:
        colorm='r'
        KK=K
    if K==0:Ws = Ws #*10**0.3
    plt.loglog(Ws*10**(-dw),Gp*10**(-dG),color=colorm,linewidth=2)
    plt.loglog(Ws*10**(-dw),Gpp*10**(-dG),'--',color=colorm,linewidth=2)

    GPS_TH.append([np.log10(Ws),np.log10(Gp),np.log10(Ws),np.log10(Gpp),Mw,Me])
    K+=1


K=0
for gp_exp,gpp_exp in zip(GP_EXP,GPP_EXP):
    colorm='gray'
    if KK==K:
        colorm='r'
    plt.loglog(10**gp_exp[0]*10**(-dw), 10**gp_exp[1]*10**(-dG-dGK[K]), marker=Marker_ALL[K],markersize=4, linestyle='', color=colorm,  markerfacecolor='white')
    plt.loglog(10**gpp_exp[0]*10**(-dw), 10**gpp_exp[1]*10**(-dG-dGK[K]), marker=Marker_ALL[K],markersize=4, linestyle='', color=colorm, markerfacecolor='white')
    K+=1
#plt.gca().xaxis.set_minor_locator(AutoMinorLocator())

ax = plt.gca()
ax.xaxis.set_minor_locator(LogLocator(base=10, subs='auto'))
#ax.xaxis.set_minor_formatter(plt.NullFormatter())  # 如果不需要次要刻度的标签可以使用这行
ax.yaxis.set_minor_locator(LogLocator(base=10, subs='auto'))

# 设置刻度线的显示
ax.tick_params(which='minor', length=2, color='k')  # 设置为红色以便可以清楚看到

# for GP92
plt.xlim(10**(-2)*10**(-dw),10**7*10**(-dw))
plt.ylim(10**2*10**(-dG),10**7*10**(-dG))

#plt.xlim(10**(-5)*10**(-dw),10**7*10**(-dw))
#plt.ylim(10**2*10**(-dG),10**7*10**(-dG))
"""
# for PS6
plt.xlim(10**(-5)*10**(-dw),10**4.5*10**(-dw))
plt.ylim(10**(2)*10**(-dG),10**6*10**(-dG))
"""

#plt.legend()
plt.xlabel('$\omega/\omega_0$',fontsize=18)
#plt.ylabel('$G$`/$G_0$,$G$``/$G_0$',fontsize=18)
plt.ylabel("$G'/G_0,G''/G_0$",fontsize=18)
plt.savefig('data/'+Polytype+'_GAM1_V13_MIN.png',dpi=600)
#with open('data/'+Polytype+'GPS_TH.pkl', 'wb') as f:
#    pickle.dump(GPS_TH, f)

plt.show()
