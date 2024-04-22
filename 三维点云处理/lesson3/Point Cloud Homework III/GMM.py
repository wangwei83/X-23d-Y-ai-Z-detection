# 文件功能：实现 GMM 算法

import numpy as np
from numpy import *
import pylab
import random,math

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal

#plt.style.use('seaborn')  #seaborn-v0_8-darkgrid
plt.style.use('seaborn-v0_8-darkgrid') 

class GMM(object):
    def __init__(self, n_clusters, max_iter=50,tol=0.001):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.nll = np.inf
    
    # 屏蔽开始
    def init(self,data):
        self.data=data
        N,dim = self.data.shape
        self.Mu = self.data[np.random.choice(range(N), self.n_clusters, replace=False)]
        #self.Var = np.array([np.eye(dim)] * self.n_clusters)
        self.Var = np.array([np.eye(dim) * 1.0 for _ in range(self.n_clusters)])  # 使协方差矩阵稍大一点
		
        self.W = np.ones([N, self.n_clusters]) / self.n_clusters
        self.pi = np.ones(self.n_clusters) / self.n_clusters

    def update_W(self):
        n_points = len(self.data)
        self.pdfs = np.zeros((n_points, self.n_clusters))
        for i in range(self.n_clusters):
            self.pdfs[:, i] = self.pi[i] * multivariate_normal.pdf(self.data, mean=self.Mu[i], cov=self.Var[i])
        self.W = self.pdfs / self.pdfs.sum(axis=1).reshape(-1, 1)

    # 更新pi
    def update_pi(self):
        self.Nk=self.W.sum(axis=0)
        self.pi = self.Nk/self.Nk.sum()
        
    # 更新Mu
    def update_Mu(self):
        for i in range(self.n_clusters):
            self.Mu[i] = np.average(self.data,axis=0,weights=self.W[:,i])

    # 更新Var
    def update_Var(self):
        reg_cov = 1e-6 * np.eye(self.data.shape[1])  # 添加小的正则化项
        for i in range(self.n_clusters):
            self.Var[i] = np.average((self.data-self.Mu[i])**2,axis=0,weights=self.W[:,i])

    def cal_nll(self):
        return np.sum(np.log(self.pdfs.sum(axis=1)))

    def fit(self, data):
        # 作业3
        # 屏蔽开始
        self.init(data)
        iter_num = 1
        
        while iter_num <= self.max_iter:
            self.update_W()
            self.update_pi()
            self.update_Mu()
            self.update_Var()

            last_nll = self.nll
            self.nll = self.cal_nll()
            if np.abs(last_nll-self.nll)<self.tol:
                print("iter_num:",iter_num)
                break
            iter_num+=1
        # 屏蔽结束
    
    def predict(self, data):
        # 屏蔽开始
        w_data=np.zeros((data.shape[0],self.n_clusters))
        for i in range(self.n_clusters):
            w_data[:, i] = self.pi[i] * multivariate_normal.pdf(x=data, mean=self.Mu[i], cov=self.Var[i])
        return np.argmax(w_data, axis=1)

# 生成仿真数据
def generate_X(true_Mu, true_Var):
    # 第一簇的数据
    num1, mu1, var1 = 400, true_Mu[0], true_Var[0]
    X1 = np.random.multivariate_normal(mu1, np.diag(var1), num1)
    # 第二簇的数据
    num2, mu2, var2 = 600, true_Mu[1], true_Var[1]
    X2 = np.random.multivariate_normal(mu2, np.diag(var2), num2)
    # 第三簇的数据
    num3, mu3, var3 = 1000, true_Mu[2], true_Var[2]
    X3 = np.random.multivariate_normal(mu3, np.diag(var3), num3)
    # 合并在一起
    X = np.vstack((X1, X2, X3))
    # 显示数据
    plt.figure(figsize=(10, 8))
    plt.axis([-10, 15, -5, 15])
    plt.scatter(X1[:, 0], X1[:, 1], s=5)
    plt.scatter(X2[:, 0], X2[:, 1], s=5)
    plt.scatter(X3[:, 0], X3[:, 1], s=5)
    plt.show()
    return X

if __name__ == '__main__':
    # 生成数据
    true_Mu = [[0.5, 0.5], [5.5, 2.5], [1, 7]]
    true_Var = [[1, 3], [2, 2], [6, 2]]
    X = generate_X(true_Mu, true_Var)

    gmm = GMM(n_clusters=3)
    gmm.fit(X)
    cat = gmm.predict(X)
    print(cat)
    # 初始化

    

