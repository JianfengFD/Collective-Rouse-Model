
import numpy as np
import matplotlib.pyplot as plt

def interpolate_points(X, Y, m=10):
    """
    对X和Y之间的每两点进行线性插值，插入m个点
    """
    Xmn = []
    Ymn = []

    for i in range(len(X) - 1):
        # 线性插值x坐标
        x_interp = np.linspace(X[i], X[i+1], m+2)[:-1]  # 去除最后一个点以避免重复
        # 线性插值y坐标
        y_interp = np.linspace(Y[i], Y[i+1], m+2)[:-1]

        Xmn.extend(x_interp)
        Ymn.extend(y_interp)

    # 将最后的点加入
    Xmn.append(X[-1])
    Ymn.append(Y[-1])

    return np.array(Xmn), np.array(Ymn)

def nearest_value(Xmn, XN):
    """
    对于XN中的每个点找到Xmn中最近的点的索引
    """
    return np.argmin(np.abs(Xmn[:,None] - XN), axis=0)

def Makeit50(X, Y, m=10, seg=49):
    # 插值
    Xmn, Ymn = interpolate_points(X, Y, m)

    # 创建XN
    XN = np.linspace(X[0], X[-1], seg+1)
    # 通过最近邻找到对应的Y值
    idx = nearest_value(Xmn, XN)
    YN = Ymn[idx]

    return XN, YN

def Makeit201(X, Y, m=10, seg=200):
    # 插值
    Xmn, Ymn = interpolate_points(X, Y, m)
    I1 = int((X[0]+10+0.0001)*10)
    I2 = int((X[-1]+10+0.0001)*10)
    I_Max=max(I1,I2)
    I_Min=min(I1,I2)
    # 创建XN
    XN = np.linspace(-10, 10, seg+1)
    # 通过最近邻找到对应的Y值
    idx = nearest_value(Xmn, XN)
    YN = Ymn[idx]
    for i in range(0,I_Min):
        YN[i]=0
    for i in range(I_Max,201):
        YN[i]=0


    return YN

def Makeit201_omg(X, Y,omg_bd, m=10, seg=200):
    # 插值
    Xmn, Ymn = interpolate_points(X, Y, m)
    I1 = int((omg_bd[0]+10+0.0001)*10)
    I2 = int((omg_bd[1]+10+0.0001)*10)
    I_Max=max(I1,I2)
    I_Min=min(I1,I2)
    # 创建XN
    XN = np.linspace(-10, 10, seg+1)
    # 通过最近邻找到对应的Y值
    idx = nearest_value(Xmn, XN)
    YN = Ymn[idx]
    for i in range(0,I_Min):
        YN[i]=0
    for i in range(I_Max,201):
        YN[i]=0


    return YN
