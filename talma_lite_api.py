"""
TALMA-on-ALPS 轻量版 —— FastAPI 路由
================================================================
提供两个新接口, 复用 alps_lite 的 ALPS 角度特征 + TALMA-DTW 时序对齐:

    GET  /api/talma/health              存活检查
    POST /api/talma/alps_feature        上传图片 -> 该帧 ALPS 特征(机位鲁棒描述子)
    POST /api/talma/alps_from_landmarks 直接给 landmark -> ALPS 特征(不依赖 MediaPipe)
    POST /api/talma/align               给两条 landmark 序列 -> DTW 对齐 + 匹配 + 相似度

挂载到主应用(zitaishibie.py 的 app 之后加一行):
    from talma_lite_api import router as talma_router
    app.include_router(talma_router)

或独立跑:
    python talma_lite_api.py     # 会启一个只含这几个接口的 uvicorn(测试用)

注意: mediapipe 只在 "上传图片" 接口内**延迟导入**, 所以本模块在
没有 mediapipe 的环境也能 import 成功(只影响图片接口)。
"""
import json

import alps_lite

try:
    from fastapi import APIRouter, File, UploadFile, Form
    from pydantic import BaseModel
    _FASTAPI_OK = True
except Exception:                       # 仅为了在无 FastAPI 时可被 import 而不崩
    _FASTAPI_OK = False
    APIRouter = File = UploadFile = Form = BaseModel = None


# ---------------- Pydantic 模型 ----------------
if _FASTAPI_OK:
    class LandmarkFrame(BaseModel):
        x: float
        y: float
        z: float = 0.0

    class AlignRequest(BaseModel):
        reference: list          # [[x,y,z]*33, ...] 每条为 33 点
        patient: list
        penalty: float = 0.0
        top_percent: float = 0.3
        anchors: list = None
        diagonal: float = 0.0


# ---------------- 路由 ----------------
if _FASTAPI_OK:
    router = APIRouter(prefix="/api/talma", tags=["talma"])

    @router.get("/health")
    def health():
        return {"code": 200, "message": "talma_lite ok", "data": {"module": "alps_lite"}}

    @router.post("/alps_from_landmarks")
    def alps_from_landmarks(payload: dict):
        """直接传入 landmarks(33 个 [x,y,z]) -> 返回该帧 ALPS 特征。"""
        try:
            lms = payload.get("landmarks")
            if not lms or len(lms) < 33:
                return {"code": 400, "message": "landmarks 需为 33 个 [x,y,z]", "data": None}
            feat = alps_lite.compute_alps_feature([tuple(float(v) for v in p) for p in lms[:33]])
            return {"code": 200, "message": "OK", "data": feat}
        except Exception as e:
            return {"code": 500, "message": f"特征计算失败: {e}", "data": None}

    @router.post("/alps_feature")
    async def alps_feature(file: UploadFile = File(...)):
        """上传图片 -> MediaPipe(延迟导入)取 landmark -> ALPS 特征。"""
        try:
            import cv2
            import mediapipe as mp
            import numpy as np
        except Exception as e:
            return {"code": 500, "message": f"需要 mediapipe/cv2/numpy: {e}", "data": None}
        try:
            data = await file.read()
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                return {"code": 400, "message": "无法解码图片", "data": None}
            mp_pose = mp.solutions.pose
            with mp_pose.Pose(static_image_mode=True, model_complexity=1) as pose:
                res = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                if not res.pose_landmarks:
                    return {"code": 400, "message": "未检测到人体姿态", "data": None}
                lms = [(lm.x, lm.y, lm.z) for lm in res.pose_landmarks.landmark]
            feat = alps_lite.compute_alps_feature(lms)
            return {"code": 200, "message": "OK", "data": feat}
        except Exception as e:
            return {"code": 500, "message": f"特征计算失败: {e}", "data": None}

    @router.post("/align")
    def align(req: AlignRequest):
        """两条 landmark 序列 -> 帧级 ALPS 特征 -> DTW 对齐 + 匹配 + 相似度。"""
        try:
            ref_seq = alps_lite.sequence_features([_frames_to_landmarks(f) for f in req.reference])
            pat_seq = alps_lite.sequence_features([_frames_to_landmarks(f) for f in req.patient])
            res = alps_lite.evaluate_segment(ref_seq, pat_seq, penalty=req.penalty, top_percent=req.top_percent)
            return {"code": 200, "message": "OK", "data": res}
        except Exception as e:
            return {"code": 500, "message": f"对齐失败: {e}", "data": None}
else:
    router = None


def _frames_to_landmarks(frame):
    """把 [[x,y,z] x 33] 转成 [(x,y,z) x 33]; 不足 33 补 0。"""
    out = []
    for p in frame:
        out.append((float(p[0]), float(p[1]), float(p[2])))
    while len(out) < 33:
        out.append((0.0, 0.0, 0.0))
    return out[:33]


if __name__ == "__main__":
    import uvicorn
    # 独立运行(仅含这些接口), 便于测试。实际使用请挂到 zitaishibie.py。
    from fastapi import FastAPI
    app = FastAPI(title="TALMA-lite 测试服务")
    app.include_router(router)
    uvicorn.run(app, host="0.0.0.0", port=8001)
