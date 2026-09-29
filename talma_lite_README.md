# TALMA-on-ALPS 轻量版（接入本系统）

把论文 **ALPS / 相机无关 / TALMA(自适应软约束DTW)** 的**思路**移植成本系统可用的小工具，
**纯 Python 标准库、零额外依赖**、不引入 numpy/matplotlib/scikit-learn，也不下载视频/3D 模型。
非常适合当前 2C4G 云服务器。

## 新增文件

| 文件 | 作用 |
|---|---|
| `alps_lite.py` | 核心算法（单文件）：ALPS 角度特征 + 机位归一 + DTW 时序对齐 |
| `talma_lite_api.py` | 把上面的算法暴露成 FastAPI 接口 |
| `test_alps_lite.py` | 自测（`python test_alps_lite.py`，已验证通过） |

## 核心能力（alps_lite.py）

- `compute_alps_feature(landmarks)`：单帧 → 全身/左/右的**肢体夹角矩阵** + 展平特征向量。
  - 用「肢体方向向量的余弦相似度」刻画姿势，对**平移/缩放/机位旋转**鲁棒（即论文 CAFE 思想）。
- `sequence_features(frames)`：多帧 → 特征序列。
- `dtw_align(a, b, penalty, anchors, diagonal)`：**自适应软约束 DTW**。允许时间弯曲对齐两条姿态序列，
  可选对「偏离线性/锚点比例」的路径施压（TALMA 思想）。
- `filter_one_to_one(path)`：把路径规整成一一对应匹配。
- `evaluate_segment(ref, pat, penalty, top_percent)`：一站式返回 匹配对 + 相似度。

## 接口（talma_lite_api.py）

```
GET  /api/talma/health
POST /api/talma/alps_feature        (multipart 图片 -> 该帧 ALPS 特征)
POST /api/talma/alps_from_landmarks (JSON {landmarks:[[x,y,z]*33]} -> 特征)
POST /api/talma/align               (JSON {reference:[..], patient:[..]} -> 对齐+匹配+相似度)
```

## 如何挂到主后端（zitaishibie.py）

在 `app = FastAPI(...)` 之后加两行：

```python
from talma_lite_api import router as talma_router
app.include_router(talma_router)
```

> 注意：`zitaishibie.py` 是 **UTF-8（无 BOM）编码**，直接插入这三行（都是 ASCII）即可，
> 用支持 UTF-8 的编辑器保存，中文注释不会乱码。你可已挂载，或按此处手动加。

## 独立测试

```bash
python talma_lite_api.py     # 只含这些接口的 uvicorn，端口 8001
```

## 价值小结（对应你的 5 个问题）

- 不启用「跨视频匹配」也**有提升**：可用 `compute_alps_feature` 得到更鲁棒的姿势描述子，
  或对「患者动作序列 vs 标准动作序列」做 DTW 对齐打分/计次 —— 都由原来的**逐帧阈值**
  升级到**整段动作**的时序对比。
- 体积：源码约 **90KB**（`alps_lite.py` 单文件），**零新依赖**，打包体积基本不变。
- 2C4G 服务器：纯 Python，CPU/内存开销极小，可放心部署。
