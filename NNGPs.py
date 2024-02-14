
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
from NNG.GetTAUS_PB import *
import pickle
import argparse
from pylab import *
import ast
from cycler import cycler
from scipy.special import erf
from scipy.optimize import fsolve
import csv

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
parser.add_argument('-i', '--input', type=str, default="PB92_3", help="Input file prefix")
parser.add_argument('-f', type=str, nargs='+', default=[], help="Input experimental data") #'PBGP92.csv'
parser.add_argument('-M', type=str, nargs='+', default=['100k'], help="Molecular weights")
parser.add_argument('-B', '--bd', type=str, default='0',nargs='+', help="A list of omega's min max: [2,7] [3,9] ...")

parser.add_argument('-dw', '--dw', type=float, default=0, help="translation of omega")
parser.add_argument('-dG', '--dG', type=float, default=0, help="translation of modulus")


args = parser.parse_args()
bds = [ast.literal_eval(l) for l in args.bd]
dw=args.dw;dG=args.dG
#parser.add_argument('-s', '--optstep', type=float, default=1.0, help="size of optimized step")
#parser.add_argument('-th', '--thres', type=float, default=0.001, help="size of optimized step")

Polytype = args.input
filenames = [f for f in args.f]
Mws = [convert_molecular_weight(mw) for mw in args.M]



def Get_Sig0(al,bet,x0,Mw,Me0):
    M =Mw/Me0
    sig0= al*(np.log(M)-np.log(x0))**2+bet
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


if Polytype.endswith(']'):
    Para_ALL=ast.literal_eval(Polytype)
    Mw_true =Para_ALL[0]
    dx =Para_ALL[1]
    dy=Para_ALL[2]
    Nu=Para_ALL[3]
    sig0=Para_ALL[4]
    sig1=Para_ALL[5]
    Mw_Real=Para_ALL[6]
    Me_rate=Mw_true/Mw_Real
else:
    with open('data/'+Polytype + 'PARA.pkl', 'rb') as f:
        para_dict = pickle.load(f)
        Nu = para_dict['Nu']
        Me_rate = para_dict['Me_0/Mw_0'] #'Me_0/Mw_0': np.exp(PARA[1])/12.96/Mw_Real,
        sig0=para_dict['Sig0']
        sig1=para_dict['Sig1']
        dx = para_dict['DX']
        dy = para_dict['DY']
        Mw_true =para_dict['Mw_true']
detxy=[dx,dy]

def get_defaultBD_Mw(Mws,detxy):
    Mws_true =np.array( [mw*Me_rate for mw in Mws])
    Mw0=17.0 # -7
    detW = np.log(Mws_true/Mw0)
    W_low = -detW*1.4-6.75+detxy[0]
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
        GP_EXP.append([cols_trimmed[i*4]+dw,cols_trimmed[i*4+1]+dG])
        GPP_EXP.append([cols_trimmed[i*4+2]+dw,cols_trimmed[i*4+3]+dG])




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
det_sig1 = sig1 - func_SIG(sig0)-(Mw_true-30)/120*0.08
#Det_Sig = Det_Sig*(Det_Sig>0)
sig0_TYPE = 'GELU'
slope_sig0= (sig0-0.42)/((np.tanh((np.log(Mw_true)-4.5))+1)/2.0)
if sig0>0.42 and np.log10(Mw_true)>1.5:
    kk=solve_sig0(Mw_true,sig0)
else:
    sig0_TYPE = 'slope'

gam,M0=get_m0_gam()
K=0
GPS_TH=[]
for Mw,bds_M in zip(Mws,bds):
    #SIG0=Get_Sig0(alph,bet,M25,Mw,Me)
    if sig0_TYPE=='slope':
        SIG0 = 0.42+(np.tanh((np.log(Mw/Me)-4.5))+1)/2.0*slope_sig0
    if sig0_TYPE=='GELU':
        SIG0 = sig0_pb(kk*(np.log10(Mw/Me)-1.4)+1.4)

    SIG1=func_SIG(SIG0)+(Mw/Me-30)/120*0.08+det_sig1

    SIG =[SIG0,SIG1]
    ZIM_LIST[2] = Get_Nu(Nu,Mw_true,Mw,Me)
    print(Mw_true,Me,Mw)
    #print(SIG0,SIG1,sig0_TYPE)
    n_real=gam*np.log(Mw/M0/Me)

    n_input1 = int(gam*np.log(Mw/M0/Me))
    n_input2 = int(gam*np.log(Mw/M0/Me))+1

    ratio1 = abs(n_real-n_input2)
    ratio2 = abs(n_real-n_input1)
    TAU_ALL1, zeta_out1 = Cal_TAUS(gam,M0,ZIM_LIST,SIG,n_input1)
    TAU_ALL2, zeta_out2 = Cal_TAUS(gam,M0,ZIM_LIST,SIG,n_input2)

    #ni = ni*MWALL/sum(ni*MWALL)

    logwA=np.reshape(linspace(bds_M[0],bds_M[1],100),(1,100))
    Gp = np.zeros_like(logwA)
    Gpp = np.zeros_like(logwA)
    W=10**logwA
    Ws=np.reshape(W,(-1))
    m_Rouse = 2
    i=0

    for TAU_ALL,zeta_out,ratio in [[TAU_ALL1,zeta_out1,ratio1],[TAU_ALL2,zeta_out2,ratio2]]:
        for n,MW,N,TAU3 in TAU_ALL:
            tau_Cnk = TAU3[0]
            tau_Rouse =TAU3[1]
            tau_min=min(min(tau_Cnk),min(tau_Rouse))
            tau_Cnk=tau_Cnk/tau_min/10**detxy[0]
            tau_Rouse=tau_Rouse/tau_min/10**detxy[0]

            N_chain=np.sum(zeta_out)/10**detxy[1]
            #Mw_RN=N_chain
            Mw_RN=N/10**detxy[1]

            tau=tau_Cnk

            Gp+=  ratio*np.sum(W**2*tau**2/(1+W**2*tau**2),axis=0)/Mw_RN
            Gpp+= ratio*np.sum(W*tau/(1+W**2*tau**2),axis=0)/Mw_RN

            tau=tau_Rouse
            Gp +=  ratio*m_Rouse*np.sum(W**2*tau**2/(1+W**2*tau**2),axis=0)/Mw_RN
            Gpp += ratio*m_Rouse*np.sum(W*tau/(1+W**2*tau**2),axis=0)/Mw_RN

    Gp=np.reshape(Gp,(-1))
    Gpp=np.reshape(Gpp,(-1))
    plt.loglog(Ws,Gp)
    plt.loglog(Ws,Gpp)
    GPS_TH.append([np.log10(Ws),np.log10(Gp),np.log10(Ws),np.log10(Gpp),Mw,Me])

K=0
for gp_exp,gpp_exp in zip(GP_EXP,GPP_EXP):
    plt.loglog(10**gp_exp[0], 10**gp_exp[1], marker=Marker_ALL[K],markersize=3, linestyle='', color='k',  markerfacecolor='white')
    plt.loglog(10**gpp_exp[0], 10**gpp_exp[1], marker=Marker_ALL[K],markersize=3, linestyle='', color='k', markerfacecolor='white')
    K+=1

        #plt.ylim(2,6)
        #plt.xlim(-5,5)

#plt.legend()
plt.xlabel('$\omega\\tau_1$')
plt.ylabel('G`,G``')
plt.savefig('data/'+Polytype+'.png',dpi=600)
print('data/'+Polytype+'.png saved')

data_for_csv = []
# 重组数据
J=0;col=['w','Gp','w','Gpp']
columns=[]
for item in GPS_TH:
    # 首先添加前四个元素
    for i in range(4):
        data_for_csv.append(item[i])
        columns.append(col[i]+'_'+args.M[J])
    J+=1

data_for_csv=np.array(data_for_csv).T
df = pd.DataFrame(data_for_csv, columns=columns)

csv_filename = 'data/' + Polytype + 'Theory.csv'
df.to_csv(csv_filename, index=False)
#GPS_TH.append([np.log10(Ws),np.log10(Gp),np.log10(Ws),np.log10(Gpp),Mw,Me])
with open('data/'+Polytype+'Theory.pkl', 'wb') as f:
    pickle.dump(GPS_TH, f)
print('data/'+Polytype+'Theory.csv saved')
print('data/'+Polytype+'Theory.pkl saved')
plt.show()
