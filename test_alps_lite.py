"""
alps_lite.py 的功能自测(纯 Python, 标准库)。
运行: python test_alps_lite.py
"""
import random
from alps_lite import (
    compute_alps_feature,
    sequence_features,
    dtw_align,
    filter_one_to_one,
    evaluate_segment,
    similarity,
)

random.seed(0)


def make_frame(arm_angle_deg=0.0, speed=1.0, t=0.0):
    """构造一帧 33 点 landmark: 模拟右臂抬起, 其余保持中性骨骼。"""
    import math
    lm = [(0.0, 0.0, 0.0) for _ in range(33)]
    # 基础骨骼坐标(归一化图像坐标, y 向下)
    base = {
        11: (0.30, 0.30, 0.0), 12: (0.70, 0.30, 0.0),
        13: (0.30, 0.50, 0.0), 14: (0.70, 0.50, 0.0),
        15: (0.30, 0.70, 0.0), 16: (0.70, 0.70, 0.0),
        23: (0.40, 0.60, 0.0), 24: (0.60, 0.60, 0.0),
        25: (0.40, 0.85, 0.0), 26: (0.60, 0.85, 0.0),
        27: (0.40, 1.00, 0.0), 28: (0.60, 1.00, 0.0),
        29: (0.40, 1.02, 0.0), 30: (0.60, 1.02, 0.0),
        31: (0.41, 1.05, 0.0), 32: (0.59, 1.05, 0.0),
    }
    for k, v in base.items():
        lm[k] = list(v)
    # 右肘/右腕随 arm_angle 抬起(绕右肩 12 旋转平面内)
    ang = math.radians(arm_angle_deg)
    sx, sy = lm[12][0], lm[12][1]
    # 简化: 让右腕绕右肩画弧
    lm[16] = [sx + 0.3 * math.cos(ang), sy + 0.3 * math.sin(ang) * 0.5, 0.0]
    lm[14] = [sx + 0.15 * math.cos(ang) + 0.1, sy + 0.15 * math.sin(ang) * 0.5, 0.0]
    # 加一点噪声模拟真实抖动
    for k in (14, 16, 13, 15):
        lm[k][0] += random.uniform(-0.005, 0.005)
        lm[k][1] += random.uniform(-0.005, 0.005)
    return [(x, y, z) for x, y, z in lm]


def main():
    print("== 1) 单帧特征结构 ==")
    f = make_frame(45.0)
    feat = compute_alps_feature(f)
    print("full 矩阵维度:", len(feat["full"]), "x", len(feat["full"][0]),
          "| left:", len(feat["left"][0]), "| right:", len(feat["right"][0]),
          "| 特征向量长度:", len(feat["feature"]))
    assert len(feat["feature"]) == len(feat["full"])**2 + len(feat["left"])**2 + len(feat["right"])**2
    print("OK")

    print("== 2) 序列 + DTW 对齐 ==")
    # 参考序列: 手臂从 0 -> 90 度(慢)
    ref = [make_frame(a, t=i) for i, a in enumerate(range(0, 91, 10))]
    # 患者序列: 同样的动作但更快/慢一点 (0->90, 但跳过几个中间帧模拟时间弯曲)
    pat = [make_frame(a, t=i) for i, a in enumerate(range(0, 91, 15))]
    ref_seq = sequence_features(ref)
    pat_seq = sequence_features(pat)

    path, D, sim = dtw_align(ref_seq, pat_seq)
    print("DTW 相似度(同动作):", sim)
    o2o = filter_one_to_one(path)
    print("一对一匹配对:", o2o)
    # 单调 + 起止校验
    assert all(path[i][0] <= path[i + 1][0] and path[i][1] <= path[i + 1][1] for i in range(len(path) - 1))
    assert path[0] == (0, 0) and path[-1] == (len(ref_seq) - 1, len(pat_seq) - 1)
    print("OK")

    print("== 3) 不同动作相似度应当更低 ==")
    # 一个完全不动(手臂保持 0 度)的序列
    still = [make_frame(0.0, t=i) for i in range(6)]
    still_seq = sequence_features(still)
    same = evaluate_segment(ref_seq, ref_seq)          # 完全相同 -> 高
    diff = evaluate_segment(ref_seq, still_seq)        # 动作不同 -> 低
    print("相同动作相似度:", same["similarity"], "| 不同动作相似度:", diff["similarity"])
    assert same["similarity"] > diff["similarity"], "相同动作应更相似"

    print("== 4) evaluate_segment 汇总 ==")
    res = evaluate_segment(ref_seq, pat_seq, penalty=0.2, top_percent=0.3)
    print("one_to_one 长度:", len(res["one_to_one"]), "| n_ref:", res["n_ref"],
          "| n_pat:", res["n_pat"], "| top_similarity:", res["top_similarity"])
    print("\n全部通过")


if __name__ == "__main__":
    main()
