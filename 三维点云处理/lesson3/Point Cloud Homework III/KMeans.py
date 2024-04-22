# 文件功能： 实现 K-Means 算法

import numpy as np

class K_Means(object):
    # k是分组数；tolerance‘中心点误差’；max_iter是迭代次数
    def __init__(self, n_clusters=2, tolerance=0.0001, max_iter=300):
        self.k_ = n_clusters
        self.tolerance_ = tolerance
        self.max_iter_ = max_iter

    def fit(self, data):
        # 作业1
        # 屏蔽开始
        idx = np.random.choice(len(data),self.k_,replace=False)
        self.k_centers = data[idx]

        n = data.shape[0]
        i = 0
        while i < self.max_iter_:
            dis_mat = np.empty((n,self.k_))
            for j,center in enumerate(self.k_centers):
                dis_mat[:,j]=np.linalg.norm(data-center.reshape(1,-1),axis=1)
            self.labels = np.argmin(dis_mat,axis=1)

            old_k_centers = self.k_centers.copy()
            for cluster_label in range(self.k_):
                mid_points = data[self.labels==cluster_label]
                if len(mid_points) > 0:
                    self.k_centers[cluster_label, :] = np.mean(mid_points, axis=0)
            
            distortion = np.linalg.norm(self.k_centers-old_k_centers,axis=1).sum()
            if distortion < self.tolerance_:
                break
            i = i+1

        # 屏蔽结束

    def predict(self, p_datas):
        result = []
        # 作业2
        # 屏蔽开始
        n = p_datas.shape[0]
        dis_mat = np.empty((n, self.k_))
        for j, center in enumerate(self.k_centers):
            dis_mat[:,j]=np.linalg.norm(p_datas-center.reshape(1,-1),axis=1)
        result = np.argmin(dis_mat,axis=1)
        # 屏蔽结束
        return result

if __name__ == '__main__':
    x = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])
    k_means = K_Means(n_clusters=2)
    k_means.fit(x)

    cat = k_means.predict(x)
    print(cat)

