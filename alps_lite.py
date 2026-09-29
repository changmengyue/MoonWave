"""
TALMA-on-ALPS —— 轻量版 (纯 Python 标准库, 零额外依赖)
================================================================
把论文核心思路移植成本系统(MediaPipe)可用的轻量工具:
  1) ALPS   : 肢体夹角姿态结构 (Angle-of-Limb-based Posture Structure)
             —— 用"肢体矢量的方向夹角(余弦相似度)"刻画姿势,
               对 平移 / 缩放 / 机位旋转 都更鲁棒。
  2) 机位归一: 方向向量归一化 → 角度特征不依赖摄像机摆放。
  3) TALMA  : 自适应软约束 DTW (adaptive soft-regulation DTW)
             —— 在允许的时间弯曲下对齐两条姿态序列, 并可选地对
               "偏离线性/锚点比例" 的路径施加软惩罚。

仅依赖标准库 (math / json / itertools), 因此可直接在 2C4G 服务器
或本项目环境运行, 不引入 numpy/matplotlib/scikit-learn。

用法(在项目里):
    from alps_lite import compute_alps_feature, sequence_features, dtw_align, evaluate_segment
    feat = compute_alps_feature(frame_landmarks)   # frame_landmarks: 33 个 (x,y,z)
* 本项目 MediaPipe 关键点为 33 点, 以下常量与之对应。
"""

import math


# ---------------- MediaPipe 33 点关键点索引 ----------------
_NOSE = 0
_L_SHO, _L_ELB, _L_WRI = 11, 13, 15
_R_SHO, _R_ELB, _R_WRI = 12, 14, 16
_L_HIP, _L_KNE, _L_ANK, _L_HEEL, _L_TOE = 23, 25, 27, 29, 31
_R_HIP, _R_KNE, _R_ANK, _R_HEEL, _R_TOE = 24, 26, 28, 30, 32

# 躯干 / 手臂 / 腿部, 每条为有向边 (a, b), 肢体向量 v = p_b - p_a
# MediaPipe 的 11/13/15/23/25/27/29/31 为人体左侧, 12/14/16/24/26/28/30/32 为右侧
_EDGES = {
    "full": [
        (_L_HIP, _L_SHO), (_R_HIP, _R_SHO),          # 躯干左右
        (_L_SHO, _L_ELB), (_L_ELB, _L_WRI),          # 左臂
        (_R_SHO, _R_ELB), (_R_ELB, _R_WRI),          # 右臂
        (_L_HIP, _L_KNE), (_L_KNE, _L_ANK),          # 左腿
        (_L_ANK, _L_HEEL), (_L_HEEL, _L_TOE),
        (_R_HIP, _R_KNE), (_R_KNE, _R_ANK),          # 右腿
        (_R_ANK, _R_HEEL), (_R_HEEL, _R_TOE),
    ],
    "left": [
        (_L_HIP, _L_SHO), (_L_SHO, _L_ELB), (_L_ELB, _L_WRI),
        (_L_HIP, _L_KNE), (_L_KNE, _L_ANK), (_L_ANK, _L_HEEL), (_L_HEEL, _L_TOE),
    ],
    "right": [
        (_R_HIP, _R_SHO), (_R_SHO, _R_ELB), (_R_ELB, _R_WRI),
        (_R_HIP, _R_KNE), (_R_KNE, _R_ANK), (_R_ANK, _R_HEEL), (_R_HEEL, _R_TOE),
    ],
}


def _norm(v):
    """方向归一化到单位向量 —— 机位/缩放无关的关键。"""
    n = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    if n == 0.0:
        return (0.0, 0.0, 0.0)
    return (v[0] / n, v[1] / n, v[2] / n)


def _cos(a, b):
    """单位向量的余弦相似度 = 点积。"""
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _limb_vectors(landmarks):
    """由 33 点 landmark 计算每类(全身/左/右)的归一化肢体向量列表。"""
    out = {}
    for name, edges in _EDGES.items():
        vecs = []
        for a, b in edges:
            ax, ay, az = landmarks[a][0], landmarks[a][1], landmarks[a][2]
            bx, by, bz = landmarks[b][0], landmarks[b][1], landmarks[b][2]
            vecs.append(_norm((bx - ax, by - ay, bz - az)))
        out[name] = vecs
    return out


def compute_alps_matrix(vecs):
    """肢体向量两两夹角(余弦相似度)矩阵 —— ALPS 姿态结构。"""
    n = len(vecs)
    M = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            M[i][j] = round(_cos(vecs[i], vecs[j]), 6)
    return M


def compute_alps_feature(landmarks):
    """
    单帧 ALPS 特征。
    :param landmarks: 长度为 33 的序列, 每元素为 (x, y, z)。
    :return: {
        "full":  n_l x n_l 矩阵, "left": ..., "right": ...,
        "feature": 展平成 1 维列表(全身+左+右),   # 排序稳定, 可直接做相似度
    }
    """
    vecs = _limb_vectors(landmarks)
    feat = []
    mat = {}
    for k in ("full", "left", "right"):
        M = compute_alps_matrix(vecs[k])
        mat[k] = M
        for row in M:
            feat.extend(row)
    return {"full": mat["full"], "left": mat["left"], "right": mat["right"], "feature": feat}


def sequence_features(frames):
    """
    多帧 -> 特征序列(每帧一个 1 维 ALPS 特征列表)。
    :param frames: 列表, 每个元素是 33 个 (x,y,z)。
    :return: list[list[float]]
    """
    return [compute_alps_feature(f)["feature"] for f in frames]


def _len(v):
    return math.sqrt(sum(x * x for x in v))


def _cos1(a, b):
    """通用(未归一)向量的余弦相似度。"""
    na, nb = _len(a), _len(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    s = sum(x * y for x, y in zip(a, b))
    return s / (na * nb)


# ---------------- TALMA: 自适应软约束 DTW ----------------
def dtw_align(a, b, penalty=0.0, anchors=None, diagonal=None):
    """
    两条特征序列的 DTW 对齐 + 可选软约束。
    :param a, b: 帧级特征列表(等长特征向量)。
    :param penalty: 软约束强度 (>=0)。>0 时惩罚偏离"比例/锚点"的对齐。
    :param anchors: 可选, 期望锚点比例列表(0~1), 用于软约束; None 时按线性。
    :param diagonal: 备用简单软约束: 惩罚 |i/m - j/n| * diagonal。
    :return: (path, cost_matrix, similarity)
        path: [(i,j)...] 从 (0,0) 到 (m-1,n-1) 的最优路径。
        cost_matrix: 累加成本矩阵。
        similarity: 0~1, 匹配成功帧的平均相似度(1-cost)。
    """
    m, n = len(a), len(b)
    anchors = anchors if anchors else _uniform_anchors(m)
    costs = [[1.0 - _cos1(a[i], b[j]) for j in range(n)] for i in range(m)]
    D = [[float("inf")] * n for _ in range(m)]
    prev = [[None] * n for _ in range(m)]
    D[0][0] = costs[0][0]

    for i in range(m):
        for j in range(n):
            if i == 0 and j == 0:
                continue
            best = float("inf")
            bk = None
            for di, dj in ((-1, 0), (0, -1), (-1, -1)):
                pi, pj = i + di, j + dj
                if 0 <= pi < m and 0 <= pj < n and D[pi][pj] < best:
                    best = D[pi][pj]
                    bk = (pi, pj)
            pen = _penalty(i, j, m, n, anchors, penalty, diagonal)
            D[i][j] = best + costs[i][j] + pen
            prev[i][j] = bk

    path = []
    i, j = m - 1, n - 1
    while (i, j) != (0, 0):
        path.append((i, j))
        bk = prev[i][j]
        if bk is None:
            break
        i, j = bk
    path.append((0, 0))
    path.reverse()

    matched = 0.0
    for i, j in path:
        matched += (1.0 - costs[i][j])
    similarity = (matched / len(path)) if path else 0.0
    return path, D, round(similarity, 4)


def _uniform_anchors(m):
    if m <= 1:
        return [0.0]
    return [i / (m - 1) for i in range(m)]


def _penalty(i, j, m, n, anchors, penalty, diagonal):
    """惩罚值(加在成本上)。penalty<=0 且无 diagonal 时返回 0。"""
    p = 0.0
    if penalty > 0.0:
        exp_j = anchors[i] * (n - 1)
        dev = abs(j - exp_j) / max(n, 1)
        p += dev * penalty
    if diagonal:
        p += abs(i / max(m, 1) - j / max(n, 1)) * diagonal
    return p


def filter_one_to_one(path):
    """把 DTW 路径规整为一对一匹配(同一 mentor 帧保留最先出现的 patient 帧)。"""
    seen = {}
    for i, j in path:
        if i not in seen:
            seen[i] = j
    return sorted(seen.items())


def similarity(a, b, top_percent=None):
    """整体/ Top-N% 平均余弦相似度, 返回 0~1。"""
    if not a or not b:
        return 0.0
    s = [_cos1(a[i], b[min(i, len(b) - 1)]) for i in range(min(len(a), len(b)))]
    if top_percent:
        k = max(1, int(len(s) * top_percent))
        s = sorted(s, reverse=True)[:k]
    return round(sum(s) / len(s), 4) if s else 0.0


def evaluate_segment(a, b, penalty=0.0, top_percent=0.3):
    """汇总: DTW 对齐 -> 一对一匹配 + 综合相似度。"""
    path, D, sim = dtw_align(a, b, penalty=penalty)
    o2o = filter_one_to_one(path)
    return {
        "path": path,
        "one_to_one": o2o,
        "similarity": sim,
        "n_ref": len(a),
        "n_pat": len(b),
        "top_similarity": similarity(a, b, top_percent) if top_percent else None,
    }
