#!/usr/bin/env python
#_*_ coding:utf-8 _*_

from numpy import *

def loadDataSet(filename):#读取数据
    dataMat = []
    labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]),float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat,labelMat#返回数据特征和类别

def selectJrand(i,m):#在0-m中随机选择一个不是i的整数
    j=i
    while(j==i):
        j = int(random.uniform(0,m))
    return j

def clipAlpha(aj,H,L):#保证a在L和H范围内
    if aj>H:
        aj = H
    if L>aj:
        aj = L
    return aj

def kernelTrans(X,A,kTup):#核函数，输出参数，X:支持向量的特征树；A:某一行特征数据
    m,n = shape(X)
    K = mat(zeros((m,1)))
    if kTup[0]=='lin':#线性函数
        K = X * A.T
    elif kTup[0] == 'rbf':#径向基函数
        for j in range(m):
            deltaRow = X[j,:] - A
            K[j] = deltaRow*deltaRow.T
        K = exp(K/(-1*kTup[1]**2))
    else:
        raise NameError("Houston We Have a Problem -- That Kernel is not recognized")
    return K

class optStruct:
    def __init__(self,dataMatIn,classLables,C,toler,kTup):
        self.X = dataMatIn
        self.lableMat = classLables
        self.C = C #软间隔参数C,参数越大，非线性拟合能力越强
        self.tol = toler #停止阙值
        self.m = shape(dataMatIn)[0] #数据行数
        self.alphas = mat(zeros((self.m,1)))
        self.b = 0 #初始设为0
        self.eCache = mat(zeros((self.m,1))) #缓存
        self.K = mat(zeros((self.m,self.m))) #核函数的计算结果
        for i in range(self.m):
            self.K[:,i] = kernelTrans(self.X,self.X[i,:],kTup)

def calcEk(oS,k): #计算Ek
    fxk = float(multiply(oS.alphas,oS.lableMat).T*oS.K[:,k]+oS.b)
    Ek = fxk - float(oS.lableMat[k])
    return Ek

def selectJ(i,oS,Ei):#随机选择aj，并返回其E值
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    oS.eCache[i] = [1,Ei]
    vaildEcacheList = nonzero(oS.eCache[:,0].A)[0]
    if (len(vaildEcacheList)) > 1:
        for k in vaildEcacheList:
            if k == 1:
                continue
            Ek = calcEk(oS,k)
            deltaE = abs(Ei - Ek)
            if (deltaE > maxDeltaE):
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxK,Ej
    else:
        j = selectJrand(i,oS.m)
        Ej = calcEk(oS,j)
    return j,Ej

def updateEk(oS,k):#更新os数据
    Ek = calcEk(oS,k)
    oS.eCache[k] = [1,Ek]

#首先检验ai是否满足KKT条件，如果不满足，随机选择aj进行优化，更新ai,aj,b值
def innerL(i,oS):#输入参数i和所有参数数据
    Ei = calcEk(oS,i)#计算E值
    if ((oS.lableMat[i]*Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.lableMat[i]*Ei > oS.tol) and
    )
