import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

#plt.style.use('seaborn')  #seaborn-v0_8-darkgrid
plt.style.use('seaborn-v0_8-darkgrid') 

class GMM:
    def __init__(self, n_clusters, max_iter=50, tol=0.001):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.nll = np.inf
    
    def init(self, data):
        self.data = data
        N, dim = self.data.shape
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

    def update_pi(self):
        self.Nk = self.W.sum(axis=0)
        self.pi = self.Nk / self.Nk.sum()
        
    def update_Mu(self):
        for i in range(self.n_clusters):
            self.Mu[i] = np.average(self.data, axis=0, weights=self.W[:, i])

    def update_Var(self):
        for i in range(self.n_clusters):
            self.Var[i] = np.average((self.data - self.Mu[i])**2, axis=0, weights=self.W[:, i])

    def cal_nll(self):
        return np.sum(np.log(self.pdfs.sum(axis=1)))

    def fit(self, data):
        self.init(data)
        iter_num = 1
        while iter_num <= self.max_iter:
            self.update_W()
            self.update_pi()
            self.update_Mu()
            self.update_Var()
            last_nll = self.nll
            self.nll = self.cal_nll()
            if np.abs(last_nll - self.nll) < self.tol:
                print("Convergence reached at iteration", iter_num)
                break
            iter_num += 1

    def predict(self, data):
        w_data = np.zeros((data.shape[0], self.n_clusters))
        for i in range(self.n_clusters):
            w_data[:, i] = self.pi[i] * multivariate_normal.pdf(x=data, mean=self.Mu[i], cov=self.Var[i])
        return np.argmax(w_data, axis=1)

# ���ɷ�������
def generate_X(true_Mu, true_Var):
    # ��һ�ص�����
    num1, mu1, var1 = 400, true_Mu[0], true_Var[0]
    X1 = np.random.multivariate_normal(mu1, np.diag(var1), num1)
    # �ڶ��ص�����
    num2, mu2, var2 = 600, true_Mu[1], true_Var[1]
    X2 = np.random.multivariate_normal(mu2, np.diag(var2), num2)
    # �����ص�����
    num3, mu3, var3 = 1000, true_Mu[2], true_Var[2]
    X3 = np.random.multivariate_normal(mu3, np.diag(var3), num3)
    # �ϲ���һ��
    X = np.vstack((X1, X2, X3))
    # ��ʾ����
    plt.figure(figsize=(10, 8))
    plt.axis([-10, 15, -5, 15])
    plt.scatter(X1[:, 0], X1[:, 1], s=5)
    plt.scatter(X2[:, 0], X2[:, 1], s=5)
    plt.scatter(X3[:, 0], X3[:, 1], s=5)
    plt.show()
    return X
if __name__ == '__main__':
    # 
    true_Mu = [[0.5, 0.5], [5.5, 2.5], [1, 7]]
    true_Var = [[1, 3], [2, 2], [6, 2]]
    X = generate_X(true_Mu, true_Var)

    gmm = GMM(n_clusters=3)
    gmm.fit(X)
    cat = gmm.predict(X)
    print(cat)
