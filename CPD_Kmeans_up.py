import numpy as np
import faiss
from sklearn.metrics import pairwise_distances_argmin


import numpy as np
import faiss


class CPD_KMeans:

    def __init__(self, data, k, max_iter=100, tol=1e-6, random_state=None):
        self.data = data
        self.k = k
        self.max_iter = max_iter
        self.tol = tol

        self.n, self.m = data.shape
        self.centers = None
        self.labels_ = None

        if random_state is not None:
            np.random.seed(random_state)

    # --------------------------------------------------
    # KMeans++ 初始化
    # --------------------------------------------------
    def init_cluster_center(self):
        centers = np.zeros((self.k, self.m))
        centers[0] = self.data[np.random.randint(self.n)]
        dist_sq = np.sum((self.data - centers[0]) ** 2, axis=1)

        for i in range(1, self.k):
            probs = dist_sq / dist_sq.sum()
            r = np.random.rand()
            centers[i] = self.data[np.searchsorted(np.cumsum(probs), r)]

            new_dist = np.sum((self.data - centers[i]) ** 2, axis=1)
            dist_sq = np.minimum(dist_sq, new_dist)

        return centers

    # --------------------------------------------------
    # 最近中心分配（Voronoi）
    # --------------------------------------------------
    def assign_clusters(self, centers):
        index = faiss.IndexFlatL2(self.m)
        index.add(centers.astype(np.float32))
        _, labels = index.search(self.data.astype(np.float32), 1)
        return labels.ravel()

    # --------------------------------------------------
    # Potential Difference（簇规模势差）
    # --------------------------------------------------
    def calculate_potential_difference(self, labels, centers):
        cluster_sizes = np.bincount(labels, minlength=self.k)
        size_diff = cluster_sizes[None, :] - cluster_sizes[:, None]

        dist_vec = centers[None, :, :] - centers[:, None, :]
        mask = size_diff < 0
        dist_vec[mask] *= -1

        norms = np.linalg.norm(dist_vec, axis=2, keepdims=True)
        norms[norms == 0] = 1e-10
        unit_vec = dist_vec / norms

        w = np.abs(size_diff / self.n)[..., None]
        PSD = np.sum(unit_vec * w, axis=1)

        return PSD

    # --------------------------------------------------
    # 样本权重计算（PSD 驱动）
    # --------------------------------------------------
    def calculate_weight(self, labels, centers, PSD):
        weight = np.ones(self.n)

        for i in range(self.k):
            idx = np.where(labels == i)[0]
            if len(idx) == 0:
                continue

            X = self.data[idx] - centers[i]
            psd = PSD[i]
            psd_norm = np.linalg.norm(psd)
            if psd_norm == 0:
                continue

            x_norm = np.linalg.norm(X, axis=1)
            x_norm[x_norm == 0] = 1e-10

            cos = (X @ psd) / (x_norm * psd_norm)

            # 当前中心到其他中心的平均距离
            center_dist = np.linalg.norm(centers - centers[i], axis=1)
            mean_center_dist = center_dist.sum() / (self.k - 1)

            weight[idx] = np.where(
                cos > 0,
                1.0 + 0.5 * np.sqrt(mean_center_dist) * psd_norm,
                1.0
            )

        return weight

    # --------------------------------------------------
    # 加权中心更新
    # --------------------------------------------------
    def compute_cluster_centers(self, labels, weight):
        centers = np.zeros((self.k, self.m))

        for j in range(self.k):
            idx = labels == j
            if not np.any(idx):
                continue

            w = weight[idx][:, None]
            centers[j] = np.sum(self.data[idx] * w, axis=0) / np.sum(w)

        return centers

    # --------------------------------------------------
    # 收敛判定
    # --------------------------------------------------
    def has_converged(self, old, new):
        return np.linalg.norm(new - old) < self.tol

    # --------------------------------------------------
    # 主训练流程
    # --------------------------------------------------
    def train(self):
        centers = self.init_cluster_center()

        for it in range(self.max_iter):
            old_centers = centers.copy()

            labels = self.assign_clusters(centers)
            PSD = self.calculate_potential_difference(labels, centers)
            weight = self.calculate_weight(labels, centers, PSD)
            centers = self.compute_cluster_centers(labels, weight)

            if self.has_converged(old_centers, centers):
                # print(f"Converged at iteration {it}")
                break

        self.centers = centers
        self.labels_ = self.assign_clusters(centers)
        return self.centers, self.labels_

