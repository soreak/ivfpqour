import math

from sklearn.neighbors import NearestNeighbors
import time
import numpy as np
from sklearn.cluster import KMeans
from collections import deque
from pynndescent import NNDescent

from .CPD_Kmeans_up import CPD_KMeans


# =========================
# Tree Node
# =========================
class TreeNode:
    def __init__(self, level=0):
        self.level = level
        self.children = []
        self.ids = None
        self.center = None

    @property
    def is_leaf(self):
        return len(self.children) == 0


# =========================
# Hierarchical KMeans
# =========================
class HierarchicalKMeans:
    def __init__(self, branching_factor, max_leaf_size, random_state=42):
        self.K = branching_factor
        self.max_leaf_size = max_leaf_size
        self.random_state = random_state
        self.root = TreeNode(level=0)
        self.leaves = []

    def _kmeans(self, X, K):
        km = KMeans(
            n_clusters=K,
            n_init=5,
            max_iter=50,
            random_state=self.random_state,
        )
        labels = km.fit_predict(X)
        return labels, km.cluster_centers_

    def _CPD(self, X, K):
        CPD = CPD_KMeans(X, K)
        cluster_centers, labels = CPD.train()
        return cluster_centers, labels

    def _should_split(self, X):
        return len(X) > self.max_leaf_size

    def fit(self, X):
        ids = np.arange(len(X))
        self.root = TreeNode(level=0)

        queue = deque([(self.root, X, ids,np.mean(X))])

        while queue:
            node, X_node, ids_node,center = queue.popleft()

            node.ids = ids_node
            node.center=center

            if not self._should_split(X_node):
                node.center = np.mean(X_node, axis=0)
                self.leaves.append(node)
                continue

            # K = min(self.K, len(X_node))
            K = min(
                self.K,
                max(2, math.ceil(len(X_node) / self.max_leaf_size))
            )

            labels, centers = self._kmeans(X_node, K)
            # centers,labels, = self._kmeans(X_node, K)
            child_centers = []
            for k in range(K):
                mask = labels == k
                if not np.any(mask):
                    continue
                child = TreeNode(level=node.level + 1)
                node.children.append(child)
                queue.append((child, X_node[mask], ids_node[mask],centers[k]))

        for i, leaf in enumerate(self.leaves):
            leaf.node_id = i

        return self


    def export_ivf(self):
        centers = []
        inverted = {}

        for cid, leaf in enumerate(self.leaves):
            centers.append(leaf.center)
            inverted[cid] = leaf.ids.tolist()

        return {
            "center_vectors": np.vstack(centers),
            "inverted": inverted,
        }

    def top_k_center(self, K):
        """
        为每个叶子中心，计算其在所有中心中的 top-K 最近邻
        使用 sklearn 的 NearestNeighbors，适合大规模中心
        返回：
            neighbors: dict
                key: center_id
                value: List[int]，长度为 K 的最近中心 id
        """
        if K <= 0:
            raise ValueError("K must be positive.")

        centers = np.vstack([leaf.center for leaf in self.leaves]).astype(np.float32)
        C, D = centers.shape

        if K >= C:
            K = C - 1  # 不包含自身

        # 使用 NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=K + 1, algorithm='auto', metric='euclidean').fit(centers)
        distances, indices = nbrs.kneighbors(centers)  #

        # 每个中心去掉自身（通常第一个就是自己）
        center_neighbors = {}
        for i in range(C):
            # 如果第一个就是自己，去掉
            topk = indices[i]
            if topk[0] == i:
                topk = topk[1:K + 1]
            else:
                topk = topk[:K]
            center_neighbors[i] = topk.tolist()

        return center_neighbors


    def search_nearest_leaf(self, q):
        """
        贪心下降，返回距离查询向量最近的叶子节点。

        时间复杂度：O(K * depth)，上界 O(|leaves|)
        """
        node = self.root

        while not node.is_leaf:
            best_child = None
            best_dist = float("inf")
            for child in node.children:
                d = np.linalg.norm(q - child.center)
                if d < best_dist:
                    best_dist = d
                    best_child = child
            node = best_child
        return node.node_id


# =========================
# Hierarchical Inverted Index
# =========================
class HierarchicalInvIndex:
    def __init__(self, X, branching_factor, max_leaf_size,n_probe_centers,random_state=42,):
        self.X = X
        self.N, self.D = X.shape

        self.hkmeans = HierarchicalKMeans(
            branching_factor, max_leaf_size, random_state
        )

        self.center_vectors = None
        self.inverted = None
        self.center_monotonic_neighbors = None
        self.center_neighbors = None
        self.graph_neighbors = None
        self.center_top_k=None
        self.n_probe_centers=n_probe_centers

    def run(self):
        self.hkmeans.fit(self.X)
        out = self.hkmeans.export_ivf()
        #保存叶子节点的top_K的最近邻
        self.center_top_k=self.hkmeans.top_k_center(self.n_probe_centers)
        self.center_vectors = out["center_vectors"]
        self.inverted = out["inverted"]

    def load_graph_neighbors(self, npz_path=None, metric="euclidean", graph_k=50, random_state=42):
        """
        加载/构建 graph_neighbors

        - 若 npz_path 非空：从 npz 中读取 neighbors / neighbor_indices
        - 若 npz_path 为空(None/"")：用 self.X 构建 NNDescent，并把整张邻接图写入 self.graph_neighbors

        参数：
          npz_path: str|None
          metric: NNDescent 使用的距离（默认 euclidean）
          graph_k: NNDescent 图的邻居数（默认 32）
          random_state: 随机种子
        """
        # -----------------------
        # 情况 A：从文件读取
        # -----------------------
        if npz_path is not None and str(npz_path).strip() != "":
            data = np.load(npz_path)
            if "neighbors" in data:
                self.graph_neighbors = data["neighbors"].astype(np.int32)
            elif "neighbor_indices" in data:
                self.graph_neighbors = data["neighbor_indices"].astype(np.int32)
            else:
                raise KeyError("npz must contain 'neighbors' or 'neighbor_indices'")
            return

        # -----------------------
        # 情况 B：npz_path 为空 -> 用 self.X 构建 NNDescent 图
        # -----------------------
        if self.X is None:
            raise RuntimeError("self.X is None, cannot build NNDescent graph.")

        X = np.asarray(self.X, dtype=np.float32, order="C")
        if X.ndim != 2:
            raise ValueError(f"self.X must be 2D array, got shape={X.shape}")

        # 你项目里如果已经有 create_nndescent，就优先用它
        # 否则直接用 pynndescent.NNDescent
        try:
            # 若你已有封装函数 create_nndescent(data=..., metric=..., graph_k=..., random_state=...)
            nnd = create_nndescent(data=X, metric=metric, graph_k=graph_k, random_state=random_state)
        except NameError:
            from pynndescent import NNDescent
            nnd = NNDescent(X, metric=metric, n_neighbors=graph_k, random_state=random_state)

        # 从 nnd 中取整张图（邻居索引矩阵）
        # 常见：neighbor_graph -> (indices, distances)
        if hasattr(nnd, "neighbor_graph"):
            g = nnd.neighbor_graph
            indices = g[0] if isinstance(g, (tuple, list)) else g
            self.graph_neighbors = np.asarray(indices, dtype=np.int32)
            return

        # 兼容一些实现：_neighbor_graph
        if hasattr(nnd, "_neighbor_graph"):
            g = nnd._neighbor_graph
            indices = g[0] if isinstance(g, (tuple, list)) else g
            self.graph_neighbors = np.asarray(indices, dtype=np.int32)
            return

        raise RuntimeError("Cannot extract neighbor graph from NNDescent object (no neighbor_graph/_neighbor_graph).")

    @staticmethod
    def _stable_unique_1d(a: np.ndarray) -> np.ndarray:
        """保持顺序去重（stable unique），返回 1D ndarray"""
        if a.size == 0:
            return a
        _, idx = np.unique(a, return_index=True)
        idx.sort()
        return a[idx]

    def add_results(self, new_results, target, overwrite=False, dtype=np.int32):
        if not hasattr(self, target) or getattr(self, target) is None:
            setattr(self, target, {})
        tgt = getattr(self, target)

        for cid, arr in enumerate(new_results):
            # 统一成 1D ndarray（尽量不拷贝）
            a = np.asarray(arr)
            if a.dtype != dtype:
                a = a.astype(dtype, copy=False)
            a = a.ravel()

            # 构建阶段就去重（保持顺序）
            a = self._stable_unique_1d(a)

            if overwrite or cid not in tgt or tgt[cid] is None:
                tgt[cid] = a
            else:
                # merge：先拼接，再 stable unique（构建期做一次即可）
                old = tgt[cid]
                if not isinstance(old, np.ndarray):
                    old = np.asarray(old, dtype=dtype).ravel()
                merged = np.concatenate([old, a])
                tgt[cid] = self._stable_unique_1d(merged)



    # def query(self, X, K, neighbors_per_center=None, monotonic_top_k=None):
    #     """ 使用 中心的单调邻居 + graph_neighbors 进行近邻查询（最后一步用欧式距离精排）。
    #     流程：
    #         1. 在 self.center_vectors 中找到最近的若干个中心（默认取 min(K, #centers) 个）；
    #         2. 对这些中心： - 查询 self.center_monotonic_neighbors 得到它们的 monotonic 邻居（若 neighbors_per_center 限制数量）；
    #         - 对每个 monotonic 邻居 vid，去 self.graph_neighbors[vid] 中取 monotonic_top_k 个最近邻， 将这些二级近邻加入候选集（而不是直接把 vid 自身加入候选）；
    #         3. 在候选集中，用 **欧式距离**（在 self.X 上计算）找到最近的 K 个向量。
    #     参数：
    #         X: np.ndarray, shape (D,) 的单个查询向量（与 self.X 同维度）
    #         K: int, 最终返回的最近邻数量
    #         n_probe_centers: int 或 None，表示查询时使用的中心数量。 若为 None，则使用 n_probe_centers = min(K, num_centers)。
    #         neighbors_per_center: int 或 None，表示每个中心最多使用多少个
    #         center_monotonic_neighbors；None 表示全部使用。
    #         monotonic_top_k: int 或 None，对每个 monotonic 邻居 vid，从 self.graph_neighbors[vid] 中取前 monotonic_top_k 个近邻。 None 表示使用该行全部近邻。
    #     返回：
    #         indices: np.ndarray, shape (K,)，在 self.X 中的索引 distances: np.ndarray, shape (K,)，对应的欧式距离 """
    #     # X = np.asarray(X, dtype=np.float32)
    #     N = self.X.shape[0]
    #     t0 = time.perf_counter()
    #     # graph_neighbors 形状与检查
    #     graph_neighbors = self.graph_neighbors
    #     G = graph_neighbors.shape[1]
    #     t1 = time.perf_counter()
    #     top_cluaster_id = self.hkmeans.search_nearest_leaf(X)
    #     # 然后通过top_k_center找到最近的top_k个中心
    #     center_ids=self.center_top_k[top_cluaster_id]
    #
    #     # 2) 候选集：center_monotonic_neighbors ->（按需要筛选）-> graph_neighbors 二级扩展
    #     t2 = time.perf_counter()
    #     cmn = self.center_monotonic_neighbors
    #     cn = self.center_neighbors  # 你要新增的中心邻居结构（格式同 cmn）
    #
    #     def _get_neighbors(container, cid):
    #         """支持 dict 或 list/array 两种形态"""
    #         if container is None:
    #             return []
    #         if isinstance(container, dict):
    #             return container.get(cid, [])
    #         else:
    #             if cid < 0 or cid >= len(container):
    #                 return []
    #             return container[cid]
    #
    #     def _add_graph_expansion(vids_1d: np.ndarray):
    #         """把 vids 的 graph_neighbors 前 mono_k_use 个加入候选集"""
    #         for vid in vids_1d:
    #             if 0 <= vid < N:
    #                 nbrs = graph_neighbors[vid, :mono_k_use]
    #                 for nid in nbrs:
    #                     if 0 <= nid < N:
    #                         candidate_set.add(int(nid))
    #
    #     # monotonic_top_k 的实际使用值
    #     if monotonic_top_k is None or monotonic_top_k <= 0:
    #         mono_k_use = G
    #     else:
    #         mono_k_use = min(monotonic_top_k, G)
    #
    #     candidate_set = set()
    #
    #     for cid in center_ids:
    #         cid = int(cid)
    #
    #         # =========================
    #         # 1) 单调邻居 cmn[cid]
    #         # =========================
    #         neigh = _get_neighbors(cmn, cid)
    #
    #         # 先过滤非法 id
    #         if neigh.size > 0:
    #             neigh = neigh[(neigh >= 0) & (neigh < N)]
    #
    #         if neigh.size > 0:
    #             # 若要限制 neighbors_per_center，则按 ||X - self.X[vid]|| 选最近的
    #             if neighbors_per_center is not None and neighbors_per_center > 0 and neigh.size > neighbors_per_center:
    #                 neigh_vecs = self.X[neigh]  # (M, D)
    #                 diffs = neigh_vecs - X[None, :]  # (M, D)
    #                 d2 = np.sum(diffs * diffs, axis=1)  # (M,)
    #                 sel = np.argpartition(d2, neighbors_per_center)[:neighbors_per_center]
    #                 neigh = neigh[sel]
    #
    #             # 单调邻居 -> graph 二级扩展
    #             _add_graph_expansion(neigh)
    #
    #         # =========================
    #         # 2) 中心邻居 cn[cid]
    #         #    加入候选集（也受 mono_k_use 控制）
    #         # =========================
    #         cneigh = _get_neighbors(cn, cid)
    #
    #
    #         if cneigh.size > 0:
    #             cneigh = cneigh[(cneigh >= 0) & (cneigh < N)]
    #             if cneigh.size > 0:
    #                 # 中心邻居 -> graph 二级扩展
    #                 _add_graph_expansion(cneigh)
    #
    #     # 如果候选集为空，则退化为全库搜索
    #     if not candidate_set:
    #         candidate_ids = np.arange(N, dtype=np.int64)
    #     else:
    #         candidate_ids = np.fromiter(candidate_set, dtype=np.int64)
    #
    #     # 3) 候选集上用欧式距离 top-K 选择（替代全排序）
    #     t3 = time.perf_counter()
    #     cand_vectors = self.X[candidate_ids]
    #     diffs = cand_vectors - X[None, :]
    #     d2 = np.sum(diffs * diffs, axis=1)  # 计算平方距离 (M,)
    #
    #     # 只做 top-K 选择，不维护其余
    #     if d2.size > K:
    #         topk_idx = np.argpartition(d2, K)[:K]  # 选出最小 K 个
    #     else:
    #         topk_idx = np.arange(d2.size)
    #
    #     top_indices = candidate_ids[topk_idx]
    #     top_distances = np.sqrt(d2[topk_idx])  # 转为真实距离用于返回
    #
    #     if top_indices.shape[0] < K:
    #         pad = K - top_indices.shape[0]
    #         top_indices = np.concatenate([top_indices, np.full(pad, -1, dtype=np.int64)])
    #         top_distances = np.concatenate([top_distances, np.full(pad, np.inf, dtype=np.float32)])
    #
    #     t4 = time.perf_counter()
    #     print(f"total={(t4 - t0) * 1000:.3f} ms | "
    #           f"s0={(t1 - t0) * 1000:.3f} ms "
    #           f"s1={(t2 - t1) * 1000:.3f} ms "
    #           f"s2={(t3 - t2) * 1000:.3f} ms "
    #           f"s3={(t4 - t3) * 1000:.3f} ms ")
    #
    #     return top_indices, top_distances

    def query(self, X, K, neighbors_per_center=None, monotonic_top_k=None,
              theta_center=0.9, theta_neigh=0.9, eps=1e-12, cap10=False,
              use_center_neighbors_as_seed=True):
        """
        使用：
          1) hkmeans 定位最近叶子 -> 取 center_top_k[leaf] 得到候选中心 center_ids
          2) 对这些中心计算可信度（F_until_theta_by_center_ids），选出“满足 theta_center 的最小 m 个中心”，并按可信度从高到低处理
          3) 对每个中心的单调邻居（cmn[cid]）做可信度筛选（theta_neigh=0.9）得到 selected_neigh
             - 第一个中心算完后取 selected_neigh 中离查询点最远的距离作为阈值 tau2
             - 后续中心在算可信度前先用 tau2 做距离剪枝，减少计算量
          4) 对选中的单调邻居做 graph 二级扩展，得到 candidate_set
          5) 在 candidate_set 上用欧氏距离做 top-K（argpartition）返回

        说明：
          - theta_neigh 你要求固定 0.9（这里默认 0.9，你也可传参）
          - neighbors_per_center：作为“单调邻居阶段”的 max_m（最多选多少个），不会再做额外排序/全排序
          - monotonic_top_k：控制 graph 扩展每个 vid 取多少个邻居
          - use_center_neighbors_as_seed：是否把 center_neighbors 也参与“单调邻居可信度筛选”的候选（一般先关掉）
        """


        # -----------------------------
        # 基本检查
        # -----------------------------
        t0 = time.perf_counter()
        if X.ndim != 1:
            raise ValueError(f"只支持单个查询向量，当前 X.ndim={X.ndim}，请传入 shape=(D,)")

        if K <= 0:
            raise ValueError("K 必须大于 0。")

        if self.X is None or self.graph_neighbors is None:
            raise RuntimeError("self.X 或 self.graph_neighbors 为空，请先构建/加载。")

        N = self.X.shape[0]
        if K > N:
            K = N

        graph_neighbors = self.graph_neighbors
        if graph_neighbors.shape[0] != N:
            raise ValueError(f"graph_neighbors 第一维 {graph_neighbors.shape[0]} 与 N={N} 不一致。")
        G = graph_neighbors.shape[1]

        if self.center_top_k is None:
            raise RuntimeError("self.center_top_k 为空，请先构建每个叶子对应的 top-k 中心列表。")
        if self.center_vectors is None:
            raise RuntimeError("self.center_vectors 为空，请先生成中心向量。")
        if self.center_monotonic_neighbors is None:
            raise RuntimeError("self.center_monotonic_neighbors 为空，请先 add_results 填充。")

        cmn = self.center_monotonic_neighbors
        cn = getattr(self, "center_neighbors", None)

        # -----------------------------
        # 1) 找最近叶子 + 拿到候选中心 center_ids
        # -----------------------------
        t1 = time.perf_counter()
        top_cluster_id = self.hkmeans.search_nearest_leaf(X)
        center_ids = self.center_top_k[top_cluster_id]

        # -----------------------------
        # 1.5) 中心可信度：从 center_ids 里选出满足 theta_center 的最小 m 个中心
        #      并按可信度从高到低处理
        # -----------------------------
        t1b = time.perf_counter()
        out_center = F_until_theta_by_center_ids(
            p=X,
            centers_ids=center_ids,
            centers_vectors=self.center_vectors,  # ✅中心向量表
            theta=theta_center,
            eps=eps,
            cap10=cap10,
            max_m=None,  # 不额外限制中心数量，让 theta 决定
        )
        selected_center_ids = out_center["selected_ids"]
        center_scores = out_center["F_selected"]

        if selected_center_ids.size == 0:
            selected_center_ids = center_ids
            center_scores = np.ones((center_ids.size,), dtype=np.float32)

        # 按可信度从高到低排序（你要求“分数最高的中心开始”）
        order = np.argsort(-center_scores)
        selected_center_ids = selected_center_ids[order]
        center_scores = center_scores[order]

        # -----------------------------
        # 2) 候选集生成：单调邻居可信度 + tau2 剪枝 + graph 二级扩展
        # -----------------------------
        t2 = time.perf_counter()

        def _get_neighbors(container, cid):
            if container is None:
                return np.empty((0,), dtype=np.int64)
            if isinstance(container, dict):
                arr = container.get(cid, None)
            else:
                arr = container[cid] if (0 <= cid < len(container)) else None
            if arr is None:
                return np.empty((0,), dtype=np.int64)
            return np.asarray(arr, dtype=np.int64).ravel()

        if monotonic_top_k is None or monotonic_top_k <= 0:
            mono_k_use = G
        else:
            mono_k_use = min(int(monotonic_top_k), G)

        candidate_set = set()
        tau2 = None  # ✅用第一个中心选中的邻居最远距离平方作为阈值

        def _add_graph_expansion(vids_1d):
            # vids_1d: iterable of int ids
            for vid in vids_1d:
                if 0 <= vid < N:
                    nbrs = graph_neighbors[vid, :mono_k_use]
                    for nid in nbrs:
                        if 0 <= nid < N:
                            candidate_set.add(int(nid))

        # 单调邻居阶段最多选多少个：用 neighbors_per_center 作为 max_m
        max_m_neigh = None
        if neighbors_per_center is not None and neighbors_per_center > 0:
            max_m_neigh = int(neighbors_per_center)

        for cid in selected_center_ids:
            cid = int(cid)

            # (a) 取该中心的单调邻居
            neigh = _get_neighbors(cmn, cid)
            if neigh.size == 0:
                continue
            neigh = neigh[(neigh >= 0) & (neigh < N)]
            if neigh.size == 0:
                continue

            # 可选：把 center_neighbors 也并入“候选邻居池”
            if use_center_neighbors_as_seed and cn is not None:
                cneigh = _get_neighbors(cn, cid)
                if cneigh.size > 0:
                    cneigh = cneigh[(cneigh >= 0) & (cneigh < N)]
                    if cneigh.size > 0:
                        neigh = np.concatenate([neigh, cneigh])
                        # 稳定去重
                        neigh = np.array(list(dict.fromkeys(neigh.tolist())), dtype=np.int64)

            # (b) tau2 剪枝：后续中心先按距离阈值过滤，减少可信度计算量
            if tau2 is not None:
                vecs = self.X[neigh]
                d2_all = np.sum((vecs - X[None, :]) ** 2, axis=1)
                mask = d2_all <= tau2
                if not np.any(mask):
                    continue
                neigh = neigh[mask]
                if neigh.size == 0:
                    continue

            # (c) 单调邻居可信度筛选（theta_neigh=0.9）
            out_neigh = F_until_theta_by_center_ids(
                p=X,
                centers_ids=neigh,
                centers_vectors=self.X,  # ✅注意：这里是数据向量表 self.X
                theta=theta_neigh,  # ✅你要求 0.9
                eps=eps,
                cap10=cap10,
                max_m=max_m_neigh,  # ✅最多选 neighbors_per_center 个
            )
            selected_neigh = out_neigh["selected_ids"]
            if selected_neigh.size == 0:
                continue

            # (d) 第一个中心：设置 tau2（选中邻居里“离查询最远”的距离平方）
            if tau2 is None:
                tau2 = float(np.max(out_neigh["dist2_selected"]))  # dist2_selected 按距离近->远累加得到的前缀

            # (e) graph 二级扩展
            _add_graph_expansion(selected_neigh)

        # 候选集为空兜底
        if not candidate_set:
            candidate_ids = np.arange(N, dtype=np.int64)
        else:
            candidate_ids = np.fromiter(candidate_set, dtype=np.int64)

        # -----------------------------
        # 3) 候选集欧氏距离 top-K（不全排序）
        # -----------------------------
        t3 = time.perf_counter()
        cand_vectors = self.X[candidate_ids]
        diffs = cand_vectors - X[None, :]
        d2 = np.sum(diffs * diffs, axis=1)  # squared dist

        if d2.size > K:
            topk_idx = np.argpartition(d2, K)[:K]
        else:
            topk_idx = np.arange(d2.size)

        top_indices = candidate_ids[topk_idx].astype(np.int64)
        top_distances = np.sqrt(d2[topk_idx]).astype(np.float32)

        # 需要的话可做最终排序（按距离升序）；不做也能用，但通常评测要排序
        ord2 = np.argsort(top_distances)
        top_indices = top_indices[ord2]
        top_distances = top_distances[ord2]

        if top_indices.shape[0] < K:
            pad = K - top_indices.shape[0]
            top_indices = np.concatenate([top_indices, np.full(pad, -1, dtype=np.int64)])
            top_distances = np.concatenate([top_distances, np.full(pad, np.inf, dtype=np.float32)])

        t4 = time.perf_counter()
        # print(f"total={(t4 - t0) * 1000:.3f} ms | "
        #       f"s0={(t1 - t0) * 1000:.3f} ms "
        #       f"s0b={(t1b - t1) * 1000:.3f} ms "
        #       f"s1={(t2 - t1b) * 1000:.3f} ms "
        #       f"s2={(t3 - t2) * 1000:.3f} ms "
        #       f"s3={(t4 - t3) * 1000:.3f} ms")

        return top_indices, top_distances


# =========================
# 可信度
# =========================

from typing import Any, Dict, Optional

def F_until_theta_by_center_ids(
    p: np.ndarray,
    centers_ids: np.ndarray,
    centers_vectors: np.ndarray,
    theta: float,
    eps: float = 1e-12,
    cap10: bool = True,
    max_m: Optional[int] = None,
) -> Dict[str, Any]:
    """
    给定候选“中心/邻居”的全局 id 列表 centers_ids，从 centers_vectors[centers_ids] 取向量，
    计算 F_until_theta，并返回“全局 id 版本”的结果。

    centers_vectors: 通常是 self.center_vectors 或 self.X
    centers_ids:     对应 centers_vectors 的行索引（全局 id）
    """
    p = np.asarray(p, dtype=np.float32).reshape(-1)
    centers_ids = np.asarray(centers_ids, dtype=np.int64).ravel()

    if centers_ids.size == 0:
        return {
            "m": 0,
            "selected_ids": np.empty((0,), dtype=np.int64),
            "F_selected": np.empty((0,), dtype=np.float32),
            "R_selected": np.empty((0,), dtype=np.float32),
            "denom_sumF": 0.0,
            "dist2_selected": np.empty((0,), dtype=np.float32),
        }

    C = np.asarray(centers_vectors[centers_ids], dtype=np.float32)  # (l, d)
    l, d = C.shape
    if p.shape[0] != d:
        raise ValueError(f"p 的维度 {p.shape[0]} 与 centers 的维度 {d} 不一致")
    if not (0.0 < theta < 1.0):
        raise ValueError("theta 建议在 (0,1) 之间")

    diff = C - p[None, :]
    dist2 = np.sum(diff * diff, axis=1)  # (l,)
    order = np.argsort(dist2)  # 近->远

    mu = float(np.mean(dist2)) if l > 0 else 0.0

    # 第1遍：分母
    denom = 0.0
    for i in range(l):
        Fi = mu / (dist2[i] + eps)
        if cap10:
            Fi = 10.0 if Fi > 10.0 else Fi
        denom += Fi

    if denom <= 0.0:
        return {
            "m": 0,
            "selected_ids": np.empty((0,), dtype=np.int64),
            "F_selected": np.empty((0,), dtype=np.float32),
            "R_selected": np.empty((0,), dtype=np.float32),
            "denom_sumF": float(denom),
            "dist2_selected": np.empty((0,), dtype=np.float32),
        }

    # 第2遍：前缀累加直到 R>theta
    num = 0.0
    sel_ids = []
    F_sel = []
    R_sel = []
    d2_sel = []

    for idx in order:
        Fi = mu / (dist2[idx] + eps)
        if cap10:
            Fi = 10.0 if Fi > 10.0 else Fi

        num += Fi
        Rm = num / denom

        sel_ids.append(int(centers_ids[idx]))     # ✅返回全局 id
        F_sel.append(float(Fi))
        R_sel.append(float(Rm))
        d2_sel.append(float(dist2[idx]))

        if max_m is not None and len(sel_ids) >= int(max_m):
            break
        if Rm > theta:
            break

    return {
        "m": len(sel_ids),
        "selected_ids": np.asarray(sel_ids, dtype=np.int64),
        "F_selected": np.asarray(F_sel, dtype=np.float32),
        "R_selected": np.asarray(R_sel, dtype=np.float32),
        "denom_sumF": float(denom),
        "dist2_selected": np.asarray(d2_sel, dtype=np.float32),
    }


# =========================
# NNDescent utils
# =========================

def generate_synthetic_data(
        dimension,
        database_size,
        query_size
):
    """生成合成数据集用于测试"""
    print("生成合成数据集...")
    # 生成有一定聚类结构的随机向量，而不是完全随机
    np.random.seed(42)

    # 创建5个聚类中心
    centers = np.random.random((5, dimension)).astype('float32')

    # 为每个聚类中心生成数据点
    database_vectors = np.zeros((database_size, dimension), dtype='float32')
    for i in range(database_size):
        center_idx = i % 5
        database_vectors[i] = centers[center_idx] + 0.1 * np.random.random(dimension).astype('float32')

    # 生成查询向量
    query_vectors = np.zeros((query_size, dimension), dtype='float32')
    for i in range(query_size):
        center_idx = i % 5
        query_vectors[i] = centers[center_idx] + 0.1 * np.random.random(dimension).astype('float32')

    return database_vectors, query_vectors


def _monotonic_filter_for_center(
        center_vec,
        candidate_indices,
        candidate_distances,
        data,
        angle_threshold_deg=45.0,
):
    """
    对某个簇中心的候选邻居做“单调性筛选”：
    - 希望选出的邻居在方向上尽量分散（角度大），保证覆盖不同方向。
    - 实现：贪心 + 角度约束。
    - 不做填充，不限制数量，返回所有满足角度约束的单调邻居。
    """
    if len(candidate_indices) == 0:
        return np.array([], dtype=np.int32), np.array([], dtype=np.float32)

    # 角度阈值 -> cos 阈值（越小越分散）
    theta = np.deg2rad(angle_threshold_deg)
    cos_threshold = np.cos(theta)

    # 按距离从小到大排序
    order = np.argsort(candidate_distances)
    candidate_indices = candidate_indices[order]
    candidate_distances = candidate_distances[order]

    center = center_vec.astype(np.float32)

    selected = []  # 选中的邻居索引（在 data 中的下标）
    selected_dirs = []  # 已选方向向量（单位向量）

    for idx, dist in zip(candidate_indices, candidate_distances):
        v = data[idx].astype(np.float32) - center
        norm = np.linalg.norm(v)
        if norm == 0:
            # 和中心重合，跳过
            continue
        direction = v / norm

        if not selected_dirs:
            # 第一个最近邻直接收
            selected.append(idx)
            selected_dirs.append(direction)
            continue

        # 和已选邻居的方向做最大余弦相似度
        cos_sim = max(float(np.dot(direction, d)) for d in selected_dirs)

        # 余弦相似度小于阈值 → 方向足够“分散”，接受
        if cos_sim < cos_threshold:
            selected.append(idx)
            selected_dirs.append(direction)

    # 输出最终选择的邻居及其距离（完全由角度筛选决定，不做补齐/截断）
    if not selected:
        return np.array([], dtype=np.int32), np.array([], dtype=np.float32)

    selected = np.array(selected, dtype=np.int32)
    dist_map = {i: d for i, d in zip(candidate_indices, candidate_distances)}
    selected_distances = np.array([dist_map[i] for i in selected], dtype=np.float32)

    return selected, selected_distances


def create_nndescent(
        data,
        metric="euclidean",
        graph_k=32,
        random_state=42,
):
    start_time_0 = time.time()
    # 在 data 上构建 NNDescent 图
    nnd = NNDescent(
        data,
        metric=metric,
        n_neighbors=graph_k,
        random_state=np.random.RandomState(random_state),
        verbose=False,
    )
    build_time_0 = time.time() - start_time_0
    print(f"[INFO] Build time: {build_time_0:.4f} seconds")

    return nnd


def find_neighbors_for_centers(
        nnd,
        centroids,
        top_k,
):
    init_indices, init_distances = nnd.query(centroids, k=top_k)
    return init_indices, init_distances


def find_monotonic_neighbors_for_centers(
        nnd,
        data,
        centroids,
        m_top_k,
        m=3,
        expand_iters=1,
        angle_threshold_deg=45.0,
):
    """
    使用 NNDescent 在 data 上构图，再以 centroids 为查询，返回每个簇中心的"单调邻居"。

    参数：
    - nnd: NNDescent 对象
    - data: np.ndarray, shape (N, D), 原始数据
    - centroids: np.ndarray, shape (C, D), 簇类中心
    - topk: int, 需要返回的单调最近邻数量
    - m: int, 初始与扩展阶段的候选放大倍数（先取 topk * m 再做单调筛选）
    - expand_iters: int, 邻居扩展迭代次数
    - angle_threshold_deg: 单调性筛选时的最小夹角（度）

    返回：
    - monotonic_neighbors: list of np.ndarray, 长度为 C
      每个元素是对应簇中心在 data 中的单调邻居索引数组
    """
    data = np.asarray(data, dtype=np.float32)
    centroids = np.asarray(centroids, dtype=np.float32)

    N, dim = data.shape
    C = centroids.shape[0]

    neighbor_graph_indices, neighbor_graph_distances = nnd.neighbor_graph

    # 2. 以 centroids 为查询，先查 topk * m 个最近邻
    initial_k = m_top_k * m
    init_indices, init_distances = find_neighbors_for_centers(nnd, centroids, top_k=initial_k)

    # 存放最终单调邻居 - 改为列表存储不同长度的数组
    monotonic_neighbors = []

    for ci in range(C):
        center_vec = centroids[ci]

        # 当前迭代的"选中邻居"和"候选"
        current_indices = init_indices[ci]
        current_distances = init_distances[ci]

        # 先用初始查询做一次单调筛选
        selected_indices, selected_distances = _monotonic_filter_for_center(
            center_vec,
            current_indices,
            current_distances,
            data,
            angle_threshold_deg=angle_threshold_deg,
        )

        # 3. 迭代扩展：用已选邻居的邻居作为候选，再做单调筛选
        for _ in range(expand_iters):
            # 聚合所有已选邻居的邻居
            candidate_set = set()
            for idx in selected_indices:
                if idx < 0 or idx >= N:
                    continue
                # graph_k 个邻居
                for nb in neighbor_graph_indices[idx]:
                    candidate_set.add(int(nb))

            # 去掉已经选过的
            candidate_set.difference_update(selected_indices.tolist())

            if not candidate_set:
                # 没有更多候选，提前结束
                break

            candidate_list = np.array(sorted(candidate_set), dtype=np.int32)

            # 计算这些候选与中心的距离
            diffs = data[candidate_list] - center_vec
            cand_distances = np.sqrt(np.sum(diffs * diffs, axis=1)).astype(np.float32)

            # 按距离取前 topk*m 作为候选
            if candidate_list.shape[0] > initial_k:
                order = np.argsort(cand_distances)[:initial_k]
                candidate_list = candidate_list[order]
                cand_distances = cand_distances[order]

            # 再做一次单调筛选
            selected_indices, selected_distances = _monotonic_filter_for_center(
                center_vec,
                candidate_list,
                cand_distances,
                data,
                angle_threshold_deg=angle_threshold_deg,
            )

        # 保存该中心最终的单调邻居 - 直接存储数组
        monotonic_neighbors.append(selected_indices)

    return monotonic_neighbors





def test_hkmeans_index(
        data_npz_path="../data/sampled_300000_data_and_neighbors.npz",
        nnd_npz_path="../data/sampled_300000_data_and_neighbors_nndescent_topk100.npz",
        # file_path = "../data/sift-128-euclidean.hdf5",

        # ===== HKMeans 参数 =====
        branching_factor=8, #8-16
        max_leaf_size=20, #20-30

        # ===== 图 & 单调邻居参数 =====
        graph_k=32,
        m_top_k=30,  # 中心周围的单调邻居数量
        expand_iters=1,
        angle_threshold_deg=45.0,

        # ===== 评测参数 =====
        # K_eval=30,
        K_eval=50,
        n_queries=50,
        n_probe_centers=30,
        neighbors_per_center=30,
        monotonic_top_k = 50

):
    """
    使用 HierarchicalKMeans + 单调邻居 + NNDescent 图 的完整测试流程
    """

    # -------------------------------------------------
    # 1. 加载数据 + ground truth
    # -------------------------------------------------
    print("[STEP] 加载数据与真实最近邻:", data_npz_path)
    data_npz = np.load(data_npz_path)

    if "sampled_data" not in data_npz or "neighbor_indices" not in data_npz:
        raise KeyError(f"{data_npz_path} 中必须包含 'sampled_data' 和 'neighbor_indices' 字段。")

    X = data_npz["sampled_data"].astype(np.float32)  # (N, D)
    true_neighbors = data_npz["neighbor_indices"].astype(np.int32)

    N, D = X.shape
    print(f"  data shape = {X.shape}, true_neighbors shape = {true_neighbors.shape}")

    # -------------------------------------------------
    # 2. 初始化并构建 HierarchicalInvIndex
    # -------------------------------------------------
    print("[STEP] 初始化并构建 HierarchicalInvIndex ...")

    index = HierarchicalInvIndex(
        X=X,
        branching_factor=branching_factor,
        max_leaf_size=max_leaf_size,
        random_state=42,
        n_probe_centers=n_probe_centers
    )

    index.run()

    print(f"  叶子中心数量 = {index.center_vectors.shape[0]}")

    # -------------------------------------------------
    # 3. 构建 NNDescent 图（基于原始数据）
    # -------------------------------------------------
    print("[STEP] 构建 NNDescent 图，用于生成中心的单调邻居 ...")

    nnd = create_nndescent(
        data=X,
        metric="euclidean",
        graph_k=graph_k,
        random_state=42,
    )

    # -------------------------------------------------
    # 4. 计算每个中心的 monotonic neighbors
    # -------------------------------------------------
    print("[STEP] 计算每个中心的 monotonic neighbors ...")

    monotonic_neighbors = find_monotonic_neighbors_for_centers(
        nnd=nnd,
        data=X,
        centroids=index.center_vectors,
        m_top_k=m_top_k,
        m=3,
        expand_iters=expand_iters,
        angle_threshold_deg=angle_threshold_deg,
    )

    index.add_results(
        new_results=monotonic_neighbors,
        target="center_monotonic_neighbors",
        overwrite=True,
    )

    neighbors, indices = find_neighbors_for_centers(
        nnd=nnd,
        centroids=index.center_vectors,
        top_k=m_top_k,
    )

    index.add_results(
        new_results=neighbors,
        target="center_neighbors",
        overwrite=True,
    )

    print("  center_monotonic_neighbors 已写入")

    # -------------------------------------------------
    # 5. 加载 NNDescent graph neighbors（用于 query 阶段）
    # -------------------------------------------------
    print("[STEP] 加载 NNDescent 近邻到 index.graph_neighbors:", nnd_npz_path)
    #
    #
    index.load_graph_neighbors(nnd_npz_path)

    # -------------------------------------------------
    # 6. 评估 recall@K
    # -------------------------------------------------
    num_queries = min(n_queries, N)
    print(f"[STEP] 开始评估（前 {num_queries} 个点，K = {K_eval}） ...")

    def recall_at_k(pred, gt, K):
        pred = np.asarray(pred, dtype=np.int64)[:K]
        gt = np.asarray(gt, dtype=np.int64)[:K]
        if pred.size == 0 or gt.size == 0:
            return 0.0
        return len(set(pred.tolist()) & set(gt.tolist())) / float(len(gt))

    recalls = []

    times = []

    for qi in range(num_queries):
        q_vec = X[qi]
        start = time.time()

        indices, _ = index.query(
            q_vec,
            K=K_eval,
            neighbors_per_center=neighbors_per_center,
            monotonic_top_k=monotonic_top_k,
        )
        times.append(time.time() - start)

        indices = indices[indices >= 0]
        gt_knn = true_neighbors[qi, :K_eval]

        r = recall_at_k(indices, gt_knn, K_eval)
        recalls.append(r)

        # ✅ 前 N 次的平均召回率
        if (qi + 1) % 10 == 0:
            avg_r = float(np.mean(recalls))
            print(
                f"  Query 1–{qi + 1}/{num_queries}: "
                f"avg recall@{K_eval} = {avg_r:.5f}"
            )

    mean_recall = float(np.mean(recalls))

    print("\n[RESULT] 平均召回率（相对真实最近邻）:")
    print(f" HKMeans + monotonic  recall@{K_eval}: {mean_recall:.4f}")
    print("平均查询时间:", np.mean(times))

    return index


if __name__ == "__main__":
    test_hkmeans_index(
        branching_factor=8,
        max_leaf_size=100,  # 影响分区数大小
        n_probe_centers=30
    )

"""
ANN Retrieval System
│
├── ① 数据空间划分框架（Partition Framework）
│
├── ② 倒排索引框架（Inverted Index Framework）
│
├── ③ 邻居图框架（Graph Framework）
│
├── ④ 候选筛选 / 扩展框架（Candidate Generation Framework）
│
├── ⑤ 查询与评测框架（Query & Evaluation Framework）
                    query
                      ↓
                    中心路由
                      ↓
                    候选合并
                      ↓
                    精排
        ┌───────────────┐
        │   Query q     │
        └───────┬───────┘
                │
        ┌───────▼───────┐
        │ HKMeans 路由  │
        │ n_probe_cent. │
        └───────┬───────┘
                │
     ┌──────────▼──────────┐
     │ Inverted Lists      │
     │ (Leaf → sample IDs) │
     └──────────┬──────────┘
                │
   ┌────────────▼────────────┐
   │ Graph / Monotonic Expand│
   └────────────┬────────────┘
                │
        ┌───────▼───────┐
        │  Re-ranking   │
        └───────┬───────┘
                │
        ┌───────▼───────┐
        │   Top-K       │
        └───────────────┘

                    """