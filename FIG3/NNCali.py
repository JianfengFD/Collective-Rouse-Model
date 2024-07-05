
import torch
import torch.nn as nn
import numpy as np
import pickle
import pandas as pd
from NNC.Interpolate50 import *
from NNC.GENGS import *
import matplotlib.pyplot as plt
from pylab import *
import time
import pickle
import argparse



BIG_MSE=0.01
dP =[0.01,0.005,0.01,0.01,0.02,0.02]

# Handle command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file', type=str, default="GPGPP.csv", help="Input data file.")
parser.add_argument('-M', '--molecular-weight', type=str, default="540k", help="Molecular weight.")
parser.add_argument('-o', '--output', type=str, default="NNGPs", help="Output filename prefix.")
parser.add_argument('-s', '--optstep', type=float, default=1.0, help="size of optimized step")
parser.add_argument('-th', '--thres', type=float, default=0.001, help="size of optimized step")

parser.add_argument('-dw', '--dw', type=float, default=0, help="translation of omega")
parser.add_argument('-dG', '--dG', type=float, default=0, help="translation of modulus")
parser.add_argument('-sz', '--SYMZ', type=bool, default=True, help="Switch between symmetric Z and non-symmetric z")
args = parser.parse_args()
ratedP= args.optstep
dw=args.dw;dG=args.dG
dP=[dp*ratedP for dp in dP]
Thres_MSE=args.thres
SYMZ = args.SYMZ

filename = args.output
# Convert molecular weight to float format
if 'k' in args.molecular_weight:
    Mw_Real = float(args.molecular_weight[:-1])
else:
    Mw_Real = float(args.molecular_weight)/1e3


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, apply_pooling=False):
        super(ResidualBlock, self).__init__()
        self.apply_pooling = apply_pooling
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.skip = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride)
        if apply_pooling:
            self.maxpool = nn.MaxPool1d(2)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        identity = self.skip(identity)
        if self.apply_pooling:
            out = self.maxpool(out)
            identity = self.maxpool(identity)
        out += identity
        out = self.relu(out)
        return out

# 定义两个ResNet模型
# 创建RES1模型时添加L2正则化
class RES1(nn.Module):
    def __init__(self):
        super(RES1, self).__init__()
        self.model = nn.Sequential(
            ResidualBlock(2, 32, apply_pooling=True),
            ResidualBlock(32, 64),
            ResidualBlock(64, 128, apply_pooling=True),
            nn.Flatten(),
            nn.Linear(50 * 128, 256),  # 201经过两次pooling变为50
            nn.ReLU(),
            nn.Linear(256, 4)
        )

    def forward(self, x):
        return self.model(x)

class RES2(nn.Module):
    def __init__(self):
        super(RES2, self).__init__()
        self.model = nn.Sequential(
            ResidualBlock(2, 32, apply_pooling=True),
            ResidualBlock(32, 64),
            ResidualBlock(64, 128, apply_pooling=True),
            nn.Flatten(),
            nn.Linear(50 * 128, 256),  # 同上
            nn.ReLU(),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        return self.model(x)




# 载入预训练模型

model1 = RES1()
model2 = RES2()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#model1.load_state_dict(torch.load('data/model1_epochN_500.pt', map_location=device))
#model2.load_state_dict(torch.load('data/model2_epochN_500.pt', map_location=device))

model1.load_state_dict(torch.load('data/model1_epochNN_450.pt', map_location=device))
model2.load_state_dict(torch.load('data/model2_epochNN_450.pt', map_location=device))


model1.eval()
model2.eval()

YM = torch.tensor([[0.42,1.6],[0.0,0.3],[0.3,0.5],[0,np.log(130)-np.log(5)],[0,8],[0,8]])
Ainv = (YM[:,1] - YM[:,0]).to(device)
Binv = YM[:,0].to(device)






# Load data
if args.file.endswith('.csv'):
    WGP = pd.read_csv('data/'+args.file, header=None)
elif args.file.endswith('.txt'):
    WGP = pd.read_csv('data/'+args.file, header=None, delimiter="\t")
elif args.file.endswith('.xlsx'):
    WGP = pd.read_excel('data/'+args.file, header=None)
else:
    raise ValueError("Unsupported file type.")
# Auto-detect starting row
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

data = WGP.iloc[start_row:, 0:4].applymap(to_float_or_nan).values
# Trim data for each column individually based on the first NaN occurrence
cols_trimmed = []
for col in data.T:  # Transpose to iterate over columns
    valid_idx = np.where(np.isnan(col))[0]
    if valid_idx.size > 0:
        first_invalid = valid_idx[0]
        cols_trimmed.append(col[:first_invalid])
    else:
        cols_trimmed.append(col)

X1 = cols_trimmed[0]+dw
Y1 = cols_trimmed[1]+dG
X2 = cols_trimmed[2]+dw
Y2 = cols_trimmed[3]+dG


X=np.zeros((2,201))
X[0,:]=Makeit201(X1, Y1)
X[1,:]=Makeit201(X2, Y2)

X_processed =torch.tensor(X)

# Move the tensor to the same device as the model
X_processed = X_processed.to(device).float()


# 使用模型进行预测
with torch.no_grad():
    X_processed = X_processed.float()
    X_processed=torch.tensor(X_processed, dtype=torch.float32)
    Y_pred1 = model1(X_processed.unsqueeze(0))
    Y_pred1 = Y_pred1 * Ainv[:4] + Binv[:4]
    print(Y_pred1)

    Y_pred2 = model2(X_processed.unsqueeze(0))
    Y_pred2 = Y_pred2 * Ainv[4:] + Binv[4:]

print(Binv,Ainv)
Y_pred1=Y_pred1.cpu().numpy()[0]
Y_pred2=Y_pred2.cpu().numpy()[0]

sig0=Y_pred1[0]
sig1_low=func_SIG(sig0)
Y0=[0,sig1_low,0.3,np.log(5)]
sig0,sig1,nu,Mw=[Y_pred1[i]+Y0[i] for i in range(4)]
Mw =np.exp(Mw)
W1,Gp,W2,Gpp = GEN_GS(nu,Mw,sig0,sig1,[-7,-1],[-7,-1],SYMZ)

W1,Gp,W2,Gpp =[np.reshape(W1,(-1)),np.reshape(Gp,(-1)),np.reshape(W2,(-1)),np.reshape(Gpp,(-1))]
print(sig0,sig1,nu,Mw)
print(Y_pred2)
#Y_true2=[8.1,6.65]
W1=W1 + Y_pred2[0]
W2=W2 + Y_pred2[0]
Gp+=Y_pred2[1]
Gpp+=Y_pred2[1]
DX= Y_pred2[0];DY=Y_pred2[1]
Gp,Gpp=GEN_GS_X(nu,Mw,sig0,sig1,DX,DY,X1,X2,SYMZ)
plt.plot(X1,Y1,'<')
plt.plot(X2,Y2,'D')
plt.plot(X1,Gp,color='lightblue',linestyle='--',label='Est. by ML')
plt.plot(X2,Gpp,color='lightblue',linestyle='--')
plt.xlabel('$\omega$')
plt.ylabel("$G',G''$")
plt.show(block=False)
time.sleep(2)
plt.ion()

def MSEG(Gp,Gpp,YG1=Y1,YG2=Y2):
    return np.mean((Gp - YG1) ** 2)+np.mean((Gpp - YG2) ** 2)


PARA = [nu,np.log(Mw),sig0,sig1,DX,DY]
dP =[0.01,0.005,0.01,0.01,0.02,0.02]
Gp,Gpp=GEN_GS_X(PARA[0],np.exp(PARA[1]),PARA[2],PARA[3],
            PARA[4],PARA[5],X1,X2,SYMZ)
mse = MSEG(Gp,Gpp)

print('alpha','Mw','sig0','beta','DX','DY')
print(PARA[0],np.exp(PARA[1]),PARA[2],PARA[3],
            PARA[4],PARA[5])

for i in range(1,10000):

    if mse<Thres_MSE:
        break

    J = np.random.randint(0, 6)
    if mse>BIG_MSE and np.random.rand()<0.8:
        J= np.random.randint(4, 6)

    PARAJ = PARA[J]
    PARA[J]+= dP[J]*(np.random.rand()-0.5)*2
    Gp,Gpp=GEN_GS_X(PARA[0],np.exp(PARA[1]),PARA[2],PARA[3],
                PARA[4],PARA[5],X1,X2,SYMZ)
    mse_NEW = MSEG(Gp,Gpp)
    if mse <  MSEG(Gp,Gpp):
        PARA[J]=PARAJ
    else:
        mse =mse_NEW
    if i%100==0:
        #plt.clf()
        print('iter/total iter:',i,'/ 20000;','MSE:',mse)
        print('al','Mw','sig0','sig1','DX','DY')
        print(PARA[0],np.exp(PARA[1]),PARA[2],PARA[3],
                    PARA[4],PARA[5])
        print('Ctrl+c to terminate the refining if you are satisfied with cali.')
        if mse>0.1:icolor=230
        if mse<=0.1:icolor = abs(int(230+(np.log10(mse)+1)*100))
        plt.plot(X1, Gp, color=(icolor / 255, icolor / 255, icolor / 255),
                    label='MSE='+str(int(mse*100000)/100000))
        plt.plot(X2, Gpp, color=(icolor / 255, icolor / 255, icolor / 255))
        plt.pause(1)
        params_dict = {
            'Nu': PARA[0],
            'Me_0/Mw_0': np.exp(PARA[1])/Mw_Real,
            'Sig0': PARA[2],
            'Sig1': PARA[3],
            'DX': PARA[4],
            'DY': PARA[5],
            'Mw_true': np.exp(PARA[1])
        }
        with open('data/'+filename+'PARA.pkl', 'wb') as f:
            pickle.dump(params_dict, f)
if mse>0.1:icolor=230
if mse<=0.1:icolor = abs(int(230+(np.log10(mse)+1)*100))
plt.plot(X1, Gp, color=(icolor / 255, icolor / 255, icolor / 255),
            label='MSE='+str(int(mse*100000)/100000))
plt.plot(X2, Gpp, color=(icolor / 255, icolor / 255, icolor / 255))
plt.pause(1)
plt.legend(loc='best')
plt.savefig('data/'+filename+'ML_MSEs.png',dpi=600)
print(i,mse)

print('al','Mw','sig0','sig1','DX','DY')
print(PARA[0],np.exp(PARA[1]),PARA[2],PARA[3],
            PARA[4],PARA[5])
plt.close()

plt.plot(X1,Y1,'<')
plt.plot(X2,Y2,'D')
plt.plot(X1,Gp,'k')
plt.plot(X2,Gpp,'k')
plt.xlabel('$\omega$')
plt.ylabel("$G',G''$")
plt.pause(2)
plt.savefig('data/'+filename+'ML_FINETUNE.png',dpi=600)

# Save the parameters to a pickle file /12.96
params_dict = {
    'Nu': PARA[0],
    'Me_0/Mw_0': np.exp(PARA[1])/Mw_Real,
    'Sig0': PARA[2],
    'Sig1': PARA[3],
    'DX': PARA[4],
    'DY': PARA[5],
    'Mw_true': np.exp(PARA[1])
}
with open('data/'+filename+'PARA.pkl', 'wb') as f:
    pickle.dump(params_dict, f)
print('Model parameters data/'+filename+'PARA.pkl saved')
