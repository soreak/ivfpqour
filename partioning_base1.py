
import time
import numpy as np
from sklearn.cluster import KMeans
from collections import deque
from pynndescent import NNDescent

from CPD_Kmeans_up import CPD_KMeans


# =========================
# Tree Node
# =========================
class TreeNode:
    def __init__(self, level=0):
        self.level = level
        self.children = []
        self.ids = None
        self.center = None


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
        CPD = CPD_KMeans(X,K)
        cluster_centers,labels = CPD.train()
        return cluster_centers,labels

    def _should_split(self, X):
        return len(X) > self.max_leaf_size

    def fit(self, X):
        ids = np.arange(len(X))
        self.leaves = []
        self.root = TreeNode(level=0)

        queue = deque([(self.root, X, ids)])

        while queue:
            node, X_node, ids_node = queue.popleft()

            if not self._should_split(X_node):
                node.ids = ids_node
                node.center = X_node.mean(axis=0)
                self.leaves.append(node)
                continue

            K = min(self.K, len(X_node))
            # labels, centers = self._kmeans(X_node, K)
            centers,labels, = self._CPD(X_node, K)

            for k in range(K):
                mask = labels == k
                if not np.any(mask):
                    continue
                child = TreeNode(level=node.level + 1)
                node.children.append(child)
                queue.append((child, X_node[mask], ids_node[mask]))

        return self

    def export_for_invpq(self):
        centers = []
        inverted = {}

        for cid, leaf in enumerate(self.leaves):
            centers.append(leaf.center)
            inverted[cid] = leaf.ids.tolist()

        return {
            "center_vectors": np.vstack(centers),
            "inverted": inverted,
        }


# =========================
# Hierarchical Inverted Index
# =========================
class HierarchicalInvIndex:
    def __init__(self, X, branching_factor, max_leaf_size, random_state=42):
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
        self.kmeans_center_graph = None

    def run(self):
        self.hkmeans.fit(self.X)
        out = self.hkmeans.export_for_invpq()
        self.center_vectors = out["center_vectors"]
        self.inverted = out["inverted"]

    def init_kmeans_graph(self):
        self.kmeans_center_graph = NNDescent(
            self.center_vectors,
            n_neighbors=30,
            random_state=np.random.RandomState(42),
            verbose=False,
        )
        return  self.kmeans_center_graph

    def load_graph_neighbors(self, npz_path):
        data = np.load(npz_path)
        if "neighbors" in data:
            self.graph_neighbors = data["neighbors"].astype(np.int64)
        elif "neighbor_indices" in data:
            self.graph_neighbors = data["neighbor_indices"].astype(np.int64)
        else:
            raise KeyError("npz must contain 'neighbors' or 'neighbor_indices'")

    def add_results(self, new_results, target, overwrite=False):
        if not hasattr(self, target) or getattr(self, target) is None:
            setattr(self, target, {})
        tgt = getattr(self, target)

        for cid, arr in enumerate(new_results):
            lst = np.asarray(arr, dtype=np.int32).tolist()
            if overwrite or cid not in tgt:
                tgt[cid] = list(dict.fromkeys(lst))
            else:
                s = set(tgt[cid])
                tgt[cid].extend(v for v in lst if v not in s)



    def query(self, X, K, n_probe_centers=None, neighbors_per_center=None, monotonic_top_k=None):
        """ 使用 中心的单调邻居 + graph_neighbors 进行近邻查询（最后一步用欧式距离精排）。
        流程：
            1. 在 self.center_vectors 中找到最近的若干个中心（默认取 min(K, #centers) 个）；
            2. 对这些中心： - 查询 self.center_monotonic_neighbors 得到它们的 monotonic 邻居（若 neighbors_per_center 限制数量）；
            - 对每个 monotonic 邻居 vid，去 self.graph_neighbors[vid] 中取 monotonic_top_k 个最近邻， 将这些二级近邻加入候选集（而不是直接把 vid 自身加入候选）；
            3. 在候选集中，用 **欧式距离**（在 self.X 上计算）找到最近的 K 个向量。
        参数：
            X: np.ndarray, shape (D,) 的单个查询向量（与 self.X 同维度）
            K: int, 最终返回的最近邻数量
            n_probe_centers: int 或 None，表示查询时使用的中心数量。 若为 None，则使用 n_probe_centers = min(K, num_centers)。
            neighbors_per_center: int 或 None，表示每个中心最多使用多少个
            center_monotonic_neighbors；None 表示全部使用。
            monotonic_top_k: int 或 None，对每个 monotonic 邻居 vid，从 self.graph_neighbors[vid] 中取前 monotonic_top_k 个近邻。 None 表示使用该行全部近邻。
        返回：
            indices: np.ndarray, shape (K,)，在 self.X 中的索引 distances: np.ndarray, shape (K,)，对应的欧式距离 """

        t0 = time.perf_counter()
        #X = np.asarray(X, dtype=np.float32)
        if X.ndim != 1:
            raise ValueError(f"只支持单个查询向量，当前 X.ndim = {X.ndim}，请传入 shape=(D,) 的向量。")

        if K <= 0:
            raise ValueError("K 必须大于 0。")


        if self.center_vectors is None:
            raise RuntimeError("center_vectors 为空，请先调用 generate_all_centers() 生成中心。")

        if self.center_monotonic_neighbors is None:
            raise RuntimeError(
                "center_monotonic_neighbors 为空，请先用 add_results(target='center_monotonic_neighbors', ...) 填充。"
            )
        if self.center_neighbors is None:
            raise RuntimeError(
                "center_neighbors 为空，请先用 add_results(target='center_neighbors', ...) 填充。"
            )

        if self.graph_neighbors is None:
            raise RuntimeError(
                "graph_neighbors 为空，请先调用 load_graph_neighbors(...) 加载 NNDescent 邻居。"
            )

        # 维度检查
        if X.shape[0] != self.D:
            raise ValueError(f"查询向量维度 {X.shape[0]} 与数据库维度 {self.D} 不一致。")

        N = self.N
        if K > N:
            K = N

        # graph_neighbors 形状与检查
        graph_neighbors = self.graph_neighbors
        if graph_neighbors.shape[0] != N:
            raise ValueError(f"graph_neighbors 第一维 {graph_neighbors.shape[0]} 与 N={N} 不一致。")
        G = graph_neighbors.shape[1]

        # 1) 找到最近的中心
        centers = self.center_vectors  # (C, D)
        C = centers.shape[0]

        if n_probe_centers is None:
            n_probe_centers = min(K, C)
        else:
            n_probe_centers = min(n_probe_centers, C)

        # diff_cent = centers - X[None, :]
        # dist_cent = np.sum(diff_cent * diff_cent, axis=1)

        q = X
        if q.ndim == 1:
            q2 = X[None, :]  # (1, D)
        elif q.ndim == 2:
            q2 = q
        else:
            raise ValueError(f"X.ndim must be 1 or 2, got {q.ndim}")

        t1 = time.perf_counter()
        center_ids, center_dists = self.kmeans_center_graph.query(q2, k=n_probe_centers)
        center_ids = center_ids[0]  # (n_probe_centers,)
        # if n_probe_centers < C:
            # center_ids = np.argpartition(dist_cent, n_probe_centers)[:n_probe_centers]
        # else:
        #     center_ids = np.arange(C, dtype=np.int64)



        # 2) 候选集：center_monotonic_neighbors ->（按需要筛选）-> graph_neighbors 二级扩展

        t2 = time.perf_counter()
        cmn = self.center_monotonic_neighbors

        cn = self.center_neighbors  # 你要新增的中心邻居结构（格式同 cmn）

        def _get_neighbors(container, cid):
            """支持 dict 或 list/array 两种形态"""
            if container is None:
                return []
            if isinstance(container, dict):
                return container.get(cid, [])
            else:
                if cid < 0 or cid >= len(container):
                    return []
                return container[cid]

        def _add_graph_expansion(vids_1d: np.ndarray):
            """把 vids 的 graph_neighbors 前 mono_k_use 个加入候选集"""
            for vid in vids_1d:
                if 0 <= vid < N:
                    nbrs = graph_neighbors[vid, :mono_k_use]
                    for nid in nbrs:
                        if 0 <= nid < N:
                            candidate_set.add(int(nid))

        # monotonic_top_k 的实际使用值
        if monotonic_top_k is None or monotonic_top_k <= 0:
            mono_k_use = G
        else:
            mono_k_use = min(monotonic_top_k, G)

        candidate_set = set()

        for cid in center_ids:
            cid = int(cid)

            # =========================
            # 1) 单调邻居 cmn[cid]
            # =========================
            neigh = _get_neighbors(cmn, cid)
            neigh = np.asarray(neigh, dtype=np.int64).ravel()

            # 先过滤非法 id
            if neigh.size > 0:
                neigh = neigh[(neigh >= 0) & (neigh < N)]

            if neigh.size > 0:
                # 若要限制 neighbors_per_center，则按 ||X - self.X[vid]|| 选最近的
                if neighbors_per_center is not None and neighbors_per_center > 0 and neigh.size > neighbors_per_center:
                    neigh_vecs = self.X[neigh]  # (M, D)
                    diffs = neigh_vecs - X[None, :]  # (M, D)
                    d2 = np.sum(diffs * diffs, axis=1)  # (M,)
                    sel = np.argpartition(d2, neighbors_per_center)[:neighbors_per_center]
                    neigh = neigh[sel]

                # 单调邻居 -> graph 二级扩展
                _add_graph_expansion(neigh)

            # =========================
            # 2) 中心邻居 cn[cid]
            #    加入候选集（也受 mono_k_use 控制）
            # =========================
            cneigh = _get_neighbors(cn, cid)
            cneigh = np.asarray(cneigh, dtype=np.int64).ravel()

            if cneigh.size > 0:
                cneigh = cneigh[(cneigh >= 0) & (cneigh < N)]
                if cneigh.size > 0:
                    # 中心邻居 -> graph 二级扩展
                    _add_graph_expansion(cneigh)

        # 如果候选集为空，则退化为全库搜索
        if not candidate_set:
            candidate_ids = np.arange(N, dtype=np.int64)
        else:
            candidate_ids = np.fromiter(candidate_set, dtype=np.int64)



        # 3) 候选集上用欧式距离 top-K 选择（替代全排序）
        t3 = time.perf_counter()
        cand_vectors = self.X[candidate_ids]
        diffs = cand_vectors - X[None, :]
        d2 = np.sum(diffs * diffs, axis=1)  # 计算平方距离 (M,)

        # 只做 top-K 选择，不维护其余
        if d2.size > K:
            topk_idx = np.argpartition(d2, K)[:K]  # 选出最小 K 个
        else:
            topk_idx = np.arange(d2.size)

        top_indices = candidate_ids[topk_idx]
        top_distances = np.sqrt(d2[topk_idx])  # 转为真实距离用于返回




        if top_indices.shape[0] < K:
            pad = K - top_indices.shape[0]
            top_indices = np.concatenate([top_indices, np.full(pad, -1, dtype=np.int64)])
            top_distances = np.concatenate([top_distances, np.full(pad, np.inf, dtype=np.float32)])


        t4 = time.perf_counter()
        print(f"total={(t4 - t0) * 1000:.3f} ms | "
              f"s0={(t1 - t0) * 1000:.3f} ms "
              f"s1={(t2 - t1) * 1000:.3f} ms "
              f"s2={(t3 - t2) * 1000:.3f} ms "
              f"s3={(t4 - t3) * 1000:.3f} ms ")
        return top_indices, top_distances





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

    selected = []        # 选中的邻居索引（在 data 中的下标）
    selected_dirs = []   # 已选方向向量（单位向量）

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
    #在 data 上构建 NNDescent 图
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
    init_indices, init_distances = find_neighbors_for_centers(nnd,centroids, top_k=initial_k)


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

    # ===== HKMeans 参数 =====
    branching_factor=8,
    max_leaf_size=100,

    # ===== 图 & 单调邻居参数 =====
    graph_k=32,
    m_top_k=30,
    expand_iters=1,
    angle_threshold_deg=45.0,

    # ===== 评测参数 =====
    K_eval=30,
    n_queries=50,
    n_probe_centers=3,
    neighbors_per_center = 3,
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

    X = data_npz["sampled_data"].astype(np.float32)               # (N, D)
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
        random_state=42
    )

    index.run()
    kmeans_center_graph=index.init_kmeans_graph()


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

    neighbors,indices = find_neighbors_for_centers(
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


    aa=1


    #用假数据 让pynndescent先进行jit编译
    rand_q = np.random.randn(D).astype(np.float32)
    # === 和真实 query 完全一致的控制 ===
    q = np.asarray(rand_q, dtype=np.float32)
    if q.ndim == 1:
        q2 = q[None, :]
    elif q.ndim == 2:
        q2 = q
    else:
        raise ValueError(f"X.ndim must be 1 or 2, got {q.ndim}")

    C = index.center_vectors.shape[0]
    k_warm = n_probe_centers
    if k_warm is None:
        k_warm = min(K_eval, C)
    else:
        k_warm = min(int(k_warm), C)
    _ = kmeans_center_graph.query(q2, k=k_warm)




    for qi in range(num_queries):
        q_vec = X[qi]
        if aa==1:
            t = time.perf_counter()

        indices, _ = index.query(
            q_vec,
            K=K_eval,
            n_probe_centers=n_probe_centers,
            neighbors_per_center = neighbors_per_center,
            monotonic_top_k = monotonic_top_k,
        )
        if aa == 1:
            dt = time.perf_counter()-t
            print(f"查询时间：{dt*1000:.3f} ms")
            aa=0

        indices = indices[indices >= 0]
        gt_knn = true_neighbors[qi, :K_eval]

        r = recall_at_k(indices, gt_knn, K_eval)
        recalls.append(r)

        # ✅ 前 N 次的平均召回率
        if (qi + 1) % 10 == 0:
            avg_r = float(np.mean(recalls))
            print(
                f"  Query 1–{qi+1}/{num_queries}: "
                f"avg recall@{K_eval} = {avg_r:.5f}"
            )

    mean_recall = float(np.mean(recalls))

    print("\n[RESULT] 平均召回率（相对真实最近邻）:")
    print(f" HKMeans + monotonic  recall@{K_eval}: {mean_recall:.4f}")

    return index


if __name__ == "__main__":
    test_hkmeans_index(
        branching_factor=8,
        max_leaf_size=50,
        n_probe_centers=10,
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