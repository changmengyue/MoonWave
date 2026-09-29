# zitaishibie.py
# 运动康复智能指导系统后端
# 运行命令：python zitaishibie.py
#
# 说明: 服务端摄像头/MJPEG 视频流链路已移除(前端改为浏览器本地 getUserMedia
# + MediaPipe 姿态估计, 回退为逐帧上传), 静态文件改为白名单方式提供。

# ========== 导入模块 ==========
from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict
import cv2
import mediapipe as mp
import numpy as np
import pymysql
from datetime import datetime, timedelta
import uuid
import hashlib
import json
import math
import os
import re
import alps_lite
from threading import Lock
import uvicorn
import threading
import webbrowser

# ========== FastAPI 应用初始化 ==========
app = FastAPI(title="运动康复智能指导系统后端")

# 认证走 Authorization 头而非 Cookie, 不需要 credentials
# CORS 来源: 默认 "*" 便于本地开发。生产环境请用环境变量收紧为实际域名, 例如:
#   CORS_ALLOW_ORIGINS="https://your-domain.example,https://www.your-domain.example"
_cors_env = os.getenv("CORS_ALLOW_ORIGINS", "*").strip()
_cors_origins = ["*"] if _cors_env in ("", "*") else [o.strip() for o in _cors_env.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

from talma_lite_api import router as talma_router
app.include_router(talma_router)

# ========== Pydantic 模型 ==========
class EvaluateRequest(BaseModel):
    action_name: str
    angle_data: Dict[str, float]
    side: str = None   # 无侧别动作的评估侧别(left/right), 缺省自动

class UserAuthRequest(BaseModel):
    username: str
    password: str

class UserUpdateRequest(BaseModel):
    username: str = None
    avatar: str = None

class SaveRecordRequest(BaseModel):
    action: str
    score: float
    detail: dict = None   # 关节数值/时序相似度等明细(可选)

class ActionLibRequest(BaseModel):
    name: str
    body_part: str
    disease_stage: str
    target_muscles: str
    standard_points: str
    contraindications: str
    standard_pose: str
    error_pose: str
    recommendation: str
    difficulty: int
    reference_source: str = None   # 标准来源/审核人(选填, 用于标注阈值出处)

class SequenceScoreRequest(BaseModel):
    reference: list = None          # 参考(标准/教练) landmark 序列
    reference_action: str = None    # 或给定动作名, 自动读取当前用户已保存模板
    patient: list = None            # 患者序列(可选; 缺省用实时滚动窗口)
    use_rolling: bool = True
    penalty: float = 0.2
    top_percent: float = 0.3

# ========== MediaPipe 初始化 ==========
# TALMA: headless 服务器强制 CPU, 避免无 GPU 环境 GL/EGL 崩溃
os.environ.setdefault('MEDIAPIPE_DISABLE_GPU', '1')
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=0,          # TALMA性能: 0=最快档(已内置lite模型), 2核CPU显著降负载
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
pose_lock = Lock()   # TALMA: MediaPipe 单例并发保护(图片/逐帧接口共享)

# ========== 全局变量 ==========
# ========== TALMA-on-ALPS 轻量版: 实时 ALPS 特征缓存 ==========
latest_alps_seq = []            # 最近 N 帧的 ALPS 特征(滚动)
latest_alps_feature = None      # 最新一帧的 ALPS 特征(机位鲁棒描述子)
alps_seq_lock = Lock()
ALPS_SEQ_LEN = 120              # 滚动窗口帧数(约 4 秒 @30fps)
DTW_SEQ_CAP = 40                # DTW 对齐前降采样上限帧数(40x40 代替 120x120, 约10倍提速)


def _downsample_seq(seq, cap=DTW_SEQ_CAP):
    """等间隔降采样到最多 cap 帧, 控制 DTW 计算量。"""
    if not seq or len(seq) <= cap:
        return list(seq or [])
    step = -(-len(seq) // cap)   # 向上取整, 保证采样后不超过 cap
    return seq[::step]


def _trim_idle_frames(seq, min_len=5, thr=0.005):
    """剔除滚动窗口首尾的静止(空闲)帧。

    滚动窗口固定 120 帧, 动作前后往往是静止站立, 直接做 DTW 会让
    时序相似度系统性偏高(静止帧互相"白送"相似度)。此处按相邻帧特征
    变化找出有运动的区段, 只保留首尾运动帧之间的部分; 运动过少则
    原样返回。"""
    if not seq or len(seq) <= min_len:
        return list(seq or [])
    def _delta(a, b):
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
    deltas = [0.0] + [_delta(seq[i], seq[i - 1]) for i in range(1, len(seq))]
    moving = [i for i, d in enumerate(deltas) if d > thr]
    if len(moving) < min_len:
        return list(seq)
    start = max(min(moving) - 1, 0)
    end = min(max(moving) + 1, len(seq) - 1)
    trimmed = seq[start:end + 1]
    return trimmed if len(trimmed) >= min_len else list(seq)


def _resolve_side(action_name: str, side: str = None) -> str:
    """侧别解析: 动作名含"左/右"时以名称为准; 其余动作(臀桥/手臂上举等
    无侧别动作)采用前端选择的患侧, 未选择时保持历史行为(右侧)。"""
    if "左" in action_name:
        return "left"
    if "右" in action_name:
        return "right"
    if side in ("left", "right"):
        return side
    return "right"


def _update_alps(landmarks):
    """把当前帧 MediaPipe landmark 的 ALPS 特征写入滚动缓存。"""
    global latest_alps_feature
    try:
        pts = [(float(lm.x), float(lm.y), float(lm.z)) for lm in landmarks]
        feat = alps_lite.compute_alps_feature(pts)
        with alps_seq_lock:
            latest_alps_feature = feat
            latest_alps_seq.append(feat["feature"])
            if len(latest_alps_seq) > ALPS_SEQ_LEN:
                del latest_alps_seq[: len(latest_alps_seq) - ALPS_SEQ_LEN]
    except Exception:
        pass


# 数据库连接: 全部通过环境变量配置(切勿把真实密码写进代码或提交到仓库)
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'user': os.getenv('DB_USER', 'rehab_user'),
    'password': os.getenv('DB_PASS', ''),            # 由环境变量 DB_PASS 提供
    'database': os.getenv('DB_NAME', 'rehab_db'),
    'charset': 'utf8mb4',
    'port': int(os.getenv('DB_PORT', '3306')),
    'autocommit': True
}

def get_db_connection():
    return pymysql.connect(**DB_CONFIG)


# ========== 认证: token 落库 + 7 天过期 ==========
TOKEN_TTL_DAYS = 7

def hash_password(password: str, salt: str = None) -> str:
    if not salt:
        salt = uuid.uuid4().hex
    hash_text = hashlib.sha256((salt + password).encode('utf-8')).hexdigest()
    return f"{salt}${hash_text}"

def verify_password(password: str, stored: str) -> bool:
    try:
        salt, stored_hash = stored.split('$', 1)
        return hashlib.sha256((salt + password).encode('utf-8')).hexdigest() == stored_hash
    except Exception:
        return False

def create_token(username: str) -> str:
    token = uuid.uuid4().hex
    try:
        conn = get_db_connection()
        c = conn.cursor()
        expires_at = (datetime.now() + timedelta(days=TOKEN_TTL_DAYS)).strftime("%Y-%m-%d %H:%M:%S")
        c.execute("REPLACE INTO user_tokens (token, username, created_at, expires_at) VALUES (%s, %s, %s, %s)",
                  (token, username, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), expires_at))
        conn.close()
    except Exception as e:
        print(f"保存 token 失败：{e}")
    return token

def get_username_by_token(token: str):
    if not token:
        return None
    try:
        conn = get_db_connection()
        c = conn.cursor()
        c.execute("SELECT username, expires_at FROM user_tokens WHERE token = %s", (token,))
        row = c.fetchone()
        if not row:
            conn.close()
            return None
        username, expires_at = row[0], row[1]
        # 旧数据无过期时间视为有效; 已过期则删除并拒绝
        if expires_at:
            try:
                if datetime.strptime(str(expires_at), "%Y-%m-%d %H:%M:%S") < datetime.now():
                    c.execute("DELETE FROM user_tokens WHERE token = %s", (token,))
                    conn.close()
                    return None
            except Exception:
                pass
        conn.close()
        return username
    except Exception as e:
        print(f"查询 token 失败：{e}")
        return None

def parse_authorization(request: Request):
    auth_value = request.headers.get('authorization') or request.headers.get('Authorization')
    if not auth_value:
        return None
    auth_value = auth_value.strip()
    if auth_value.lower().startswith('bearer '):
        return auth_value[7:].strip()
    return auth_value

def get_current_user(request: Request):
    token = parse_authorization(request)
    return get_username_by_token(token) if token else None

# ========== 角度计算函数 ==========
def calc_angle(a, b, c):
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])
    cosine_angle = (ba[0]*bc[0] + ba[1]*bc[1]) / (math.hypot(*ba) * math.hypot(*bc) + 1e-6)
    cosine_angle = max(min(cosine_angle, 1.0), -1.0)
    angle = math.acos(cosine_angle)
    return math.degrees(angle)

# ========== 数据库初始化 ==========
def _ensure_column(c, table: str, column: str, ddl: str):
    """补列(兼容 MySQL/MariaDB): information_schema 判断列是否存在, 缺失再 ALTER。"""
    c.execute(
        "SELECT COUNT(*) FROM information_schema.COLUMNS "
        "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = %s AND COLUMN_NAME = %s",
        (table, column))
    if c.fetchone()[0] == 0:
        c.execute(f"ALTER TABLE `{table}` ADD COLUMN `{column}` {ddl}")


def init_db():
    conn = get_db_connection()
    c = conn.cursor()

    c.execute('''CREATE TABLE IF NOT EXISTS records
                 (`user` VARCHAR(255), action VARCHAR(255), score DOUBLE, time VARCHAR(255), detail MEDIUMTEXT)''')

    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INT AUTO_INCREMENT PRIMARY KEY,
                  username VARCHAR(255) NOT NULL UNIQUE,
                  password_hash VARCHAR(255) NOT NULL,
                  created_at VARCHAR(255) NOT NULL,
                  avatar MEDIUMTEXT)''')
    # 兼容已存在的旧表: 补 avatar / expires_at / detail 列
    try:
        _ensure_column(c, 'users', 'avatar', 'MEDIUMTEXT')
        _ensure_column(c, 'user_tokens', 'expires_at', 'VARCHAR(255)')
        _ensure_column(c, 'records', 'detail', 'MEDIUMTEXT')
        _ensure_column(c, 'rehab_action_lib', 'reference_source', 'VARCHAR(255)')
    except Exception as e:
        print(f"补列跳过: {e}")

    c.execute('''CREATE TABLE IF NOT EXISTS user_tokens
                 (token VARCHAR(64) PRIMARY KEY,
                  username VARCHAR(255) NOT NULL,
                  created_at VARCHAR(255) NOT NULL,
                  expires_at VARCHAR(255))''')

    c.execute('''CREATE TABLE IF NOT EXISTS rehab_action_lib
                 (id VARCHAR(36) PRIMARY KEY,
                  name VARCHAR(255) NOT NULL UNIQUE,
                  body_part VARCHAR(255) NOT NULL,
                  disease_stage VARCHAR(255) NOT NULL,
                  target_muscles TEXT NOT NULL,
                  standard_points TEXT NOT NULL,
                  contraindications TEXT NOT NULL,
                  standard_pose TEXT NOT NULL,
                  error_pose TEXT NOT NULL,
                  recommendation TEXT NOT NULL,
                  difficulty INT NOT NULL,
                  create_time VARCHAR(255) NOT NULL,
                  reference_source VARCHAR(255))''')

    default_actions = []

    arm_lift_id = str(uuid.uuid4())
    arm_lift_standard_pose = {
        "joint_angles": {"elbow": [30, 160], "shoulder": [20, 140]},
        "keypoints": {"shoulder": [11, 13], "elbow": [13, 15], "wrist": [15]}
    }
    arm_lift_error_pose = {
        "elbow_too_high": {"angle": ">160", "tip": "手臂不要抬过高，避免肩关节拉伤"},
        "elbow_too_low": {"angle": "<30", "tip": "手臂抬升不足，达不到康复效果"}
    }
    arm_lift_recommendation = {
        "times_per_set": 15, "sets": 3,
        "rhythm": "2秒上举，1秒停留，2秒放下",
        "breath": "上举呼气，放下吸气"
    }
    default_actions.append((
        arm_lift_id, "手臂上举", "肩", "肩周炎/肩关节术后恢复期（1-3个月）",
        "三角肌、斜方肌、肱二头肌",
        "1. 身体直立，双脚与肩同宽；2. 手臂缓慢上举，肘部微屈；3. 举至最高点停留1秒；4. 缓慢放下",
        "1. 肩关节脱位史禁用；2. 急性肩周炎发作期禁用",
        json.dumps(arm_lift_standard_pose, ensure_ascii=False),
        json.dumps(arm_lift_error_pose, ensure_ascii=False),
        json.dumps(arm_lift_recommendation, ensure_ascii=False),
        2, datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))

    knee_bend_id = str(uuid.uuid4())
    knee_bend_standard_pose = {
        "joint_angles": {"knee": [70, 120], "hip": [60, 100]},
        "keypoints": {"hip": [23], "knee": [25], "ankle": [27]}
    }
    knee_bend_error_pose = {
        "knee_too_bent": {"angle": ">120", "tip": "膝盖弯曲过度，易损伤半月板"},
        "knee_not_bent_enough": {"angle": "<70", "tip": "膝盖弯曲不足，康复效果差"}
    }
    knee_bend_recommendation = {
        "times_per_set": 20, "sets": 4,
        "rhythm": "3秒屈膝，2秒停留，3秒伸直",
        "breath": "屈膝呼气，伸直吸气"
    }
    default_actions.append((
        knee_bend_id, "屈膝训练", "膝", "膝关节滑膜炎/半月板修复术后（2-6个月）",
        "股四头肌、腘绳肌、腓肠肌",
        "1. 背靠墙站立，双脚离墙30cm；2. 缓慢屈膝，膝盖不超过脚尖；3. 屈膝至90度停留2秒",
        "1. 膝关节积液期禁用；2. 骨质疏松患者避免屈膝超过90度",
        json.dumps(knee_bend_standard_pose, ensure_ascii=False),
        json.dumps(knee_bend_error_pose, ensure_ascii=False),
        json.dumps(knee_bend_recommendation, ensure_ascii=False),
        3, datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))

    knee_flex_id = str(uuid.uuid4())
    knee_flex_standard = {
        "joint_angles": {"main_joint": [0, 90]},
        "keypoints": [23, 25, 27]
    }
    knee_flex_error = {
        "below_std": {"angle": "<0", "tip": "屈膝未达标"},
        "above_std": {"angle": ">90", "tip": "屈膝过度"}
    }
    knee_flex_recommend = {
        "times_per_set": 15, "sets": 3,
        "rhythm": "2秒完成一次动作，1秒停留",
        "breath": "发力呼气，放松吸气"
    }
    default_actions.append((
        knee_flex_id, "膝关节屈曲", "膝", "膝关节术后康复",
        "下肢肌群", "髋-膝-踝夹角<90°", "暂无明确禁忌（遵医嘱）",
        json.dumps(knee_flex_standard, ensure_ascii=False),
        json.dumps(knee_flex_error, ensure_ascii=False),
        json.dumps(knee_flex_recommend, ensure_ascii=False),
        2, datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))

    left_leg_raise_id = str(uuid.uuid4())
    left_leg_raise_standard = {
        "joint_angles": {"main_joint": [165, 180]},
        "keypoints": [24, 26, 28]
    }
    left_leg_raise_error = {
        "below_std": {"angle": "<165", "tip": "左腿未抬直"},
        "above_std": {"angle": ">180", "tip": "左腿过伸"}
    }
    left_leg_raise_recommend = {
        "times_per_set": 10, "sets": 3,
        "rhythm": "2秒抬起，1秒停留",
        "breath": "抬起呼气，放下吸气"
    }
    default_actions.append((
        left_leg_raise_id, "左腿直腿抬高", "髋/膝", "膝关节术后康复",
        "股四头肌", "髋-膝-踝夹角>165°", "暂无明确禁忌",
        json.dumps(left_leg_raise_standard, ensure_ascii=False),
        json.dumps(left_leg_raise_error, ensure_ascii=False),
        json.dumps(left_leg_raise_recommend, ensure_ascii=False),
        2, datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))

    right_arm_raise_id = str(uuid.uuid4())
    right_arm_raise_standard = {
        "joint_angles": {"main_joint": [140, 200]},
        "keypoints": [11, 13, 15]
    }
    right_arm_raise_error = {
        "below_std": {"angle": "<140", "tip": "右臂未举过肩"},
        "above_std": {"angle": ">200", "tip": "右臂上举过度"}
    }
    right_arm_raise_recommend = {
        "times_per_set": 15, "sets": 3,
        "rhythm": "2秒上举，1秒停留",
        "breath": "上举呼气，放下吸气"
    }
    default_actions.append((
        right_arm_raise_id, "右臂上举", "肩/臂", "脑卒中上肢康复",
        "上肢肌群", "肩-肘-腕角>140°", "暂无明确禁忌",
        json.dumps(right_arm_raise_standard, ensure_ascii=False),
        json.dumps(right_arm_raise_error, ensure_ascii=False),
        json.dumps(right_arm_raise_recommend, ensure_ascii=False),
        2, datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))

    for side, side_cn in [("left", "左"), ("right", "右")]:
        kp_shoulder, kp_elbow, kp_wrist = (11,13,15) if side=="left" else (12,14,16)
        standard_pose = {
            "joint_angles": {"shoulder_flexion": [20, 180]},
            "keypoints": {"shoulder": [kp_shoulder], "elbow": [kp_elbow], "wrist": [kp_wrist]}
        }
        error_pose = {
            "flexion_too_small": {"angle": "<10", "tip": "患侧上肢未能抬起或活动范围小于10°"},
            "flexion_too_large": {"angle": ">180", "tip": "肩关节前屈超过正常范围"}
        }
        recommendation = {
            "times_per_set": 10, "sets": 5,
            "rhythm": "抬起3秒，末端维持3-5秒，放下3-5秒",
            "breath": "抬起时呼气，放下时吸气"
        }
        default_actions.append((
            str(uuid.uuid4()), f"{side_cn}肩关节仰卧被动前屈", "肩",
            "肩关节术后/肩周炎恢复期（早期康复）",
            "三角肌前束、胸大肌、肱二头肌",
            "1. 仰卧于床；2. 健侧手握住患侧手腕；3. 患侧上肢不发力，健侧手缓慢举起；4. 达最大范围维持3-5秒",
            "1. 急性炎症期禁用；2. 骨折未愈合禁用",
            json.dumps(standard_pose, ensure_ascii=False),
            json.dumps(error_pose, ensure_ascii=False),
            json.dumps(recommendation, ensure_ascii=False),
            1, datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ))

    for side, side_cn in [("left", "左"), ("right", "右")]:
        kp_shoulder, kp_elbow, kp_wrist = (11,13,15) if side=="left" else (12,14,16)
        standard_pose = {
            "joint_angles": {"elbow_flexion": [30, 150]},
            "keypoints": {"shoulder": [kp_shoulder], "elbow": [kp_elbow], "wrist": [kp_wrist]}
        }
        error_pose = {
            "flexion_insufficient": {"angle": "<90", "tip": "肘关节屈曲不足90°"},
            "hyperextension": {"angle": ">180", "tip": "肘关节过伸"}
        }
        recommendation = {
            "times_per_set": 12, "sets": 3,
            "rhythm": "屈曲3秒，末端维持2-3秒，伸直3秒",
            "breath": "屈曲时吸气，伸直时呼气"
        }
        default_actions.append((
            str(uuid.uuid4()), f"{side_cn}肘关节坐位主动屈伸", "肘",
            "肘关节骨折术后/关节炎恢复期",
            "肱二头肌、肱肌",
            "1. 坐于靠背椅；2. 缓慢屈肘将手掌向肩部靠拢；3. 保持2-3秒后伸直",
            "1. 骨折未愈合禁用；2. 急性炎症禁用",
            json.dumps(standard_pose, ensure_ascii=False),
            json.dumps(error_pose, ensure_ascii=False),
            json.dumps(recommendation, ensure_ascii=False),
            2, datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ))

    for side, side_cn in [("left", "左"), ("right", "右")]:
        kp_hip, kp_knee, kp_ankle = (23,25,27) if side=="left" else (24,26,28)
        standard_pose = {
            "joint_angles": {"knee_flexion": [0, 150]},
            "keypoints": {"hip": [kp_hip], "knee": [kp_knee], "ankle": [kp_ankle]}
        }
        error_pose = {
            "flexion_insufficient": {"angle": "<90", "tip": "膝关节屈曲不足90°"},
            "hyperextension": {"angle": ">190", "tip": "膝关节过伸"}
        }
        recommendation = {
            "times_per_set": 15, "sets": 3,
            "rhythm": "屈曲4秒，维持5-10秒，伸直3秒",
            "breath": "屈曲时呼气，伸直时吸气"
        }
        default_actions.append((
            str(uuid.uuid4()), f"{side_cn}膝关节坐位主动屈伸", "膝",
            "膝关节术后/骨关节炎恢复期",
            "股四头肌、腘绳肌",
            "1. 坐于高凳；2. 患侧脚底放水瓶辅助滑动；3. 缓慢屈膝至最大角度维持5-10秒；4. 缓慢伸直",
            "1. 置换术后早期禁用；2. 急性创伤禁用",
            json.dumps(standard_pose, ensure_ascii=False),
            json.dumps(error_pose, ensure_ascii=False),
            json.dumps(recommendation, ensure_ascii=False),
            2, datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ))

    for side, side_cn in [("left", "左"), ("right", "右")]:
        kp_knee, kp_ankle, kp_heel, kp_toe = (25,27,29,31) if side=="left" else (26,28,30,32)
        standard_pose = {
            "joint_angles": {
                "ankle_dorsiflexion": [0, 20],
                "ankle_plantarflexion": [0, 45]
            },
            "keypoints": {
                "knee": [kp_knee], "ankle": [kp_ankle],
                "heel": [kp_heel], "toe": [kp_toe]
            }
        }
        error_pose = {
            "dorsiflexion_insufficient": {"angle": "dorsiflexion<5", "tip": "勾脚不足"},
            "plantarflexion_insufficient": {"angle": "plantarflexion<20", "tip": "绷脚不足"}
        }
        recommendation = {
            "times_per_set": 15, "sets": 5,
            "rhythm": "勾脚尖2-3秒，维持2-3秒，绷脚尖2-3秒",
            "breath": "勾脚尖吸气，绷脚尖呼气"
        }
        default_actions.append((
            str(uuid.uuid4()), f"{side_cn}踝关节坐位勾脚与绷脚尖", "踝",
            "踝关节扭伤后/跟腱术后恢复期",
            "胫骨前肌、小腿三头肌",
            "1. 坐位伸腿；2. 用力勾脚尖维持2-3秒；3. 用力绷脚尖维持2-3秒",
            "1. 急性扭伤肿胀期禁用；2. 跟腱断裂术后急性期禁用",
            json.dumps(standard_pose, ensure_ascii=False),
            json.dumps(error_pose, ensure_ascii=False),
            json.dumps(recommendation, ensure_ascii=False),
            2, datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ))

    # 内置动作(显式列名插入, 避免表结构增列后位置错位)
    c.executemany(
        "INSERT IGNORE INTO rehab_action_lib "
        "(id, name, body_part, disease_stage, target_muscles, standard_points, "
        "contraindications, standard_pose, error_pose, recommendation, difficulty, create_time) "
        "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)", default_actions)

    # 导入 action_seed.json(方案数据引用的动作, 此前该文件从未被加载)
    try:
        seed_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "action_seed.json")
        if os.path.exists(seed_path):
            with open(seed_path, "r", encoding="utf-8") as f:
                seeds = json.load(f)
            seed_rows = [(
                s["id"], s["name"], s["body_part"], s["disease_stage"],
                s["target_muscles"], s["standard_points"], s["contraindications"],
                s["standard_pose"], s["error_pose"], s["recommendation"],
                int(s["difficulty"]), s["create_time"], s.get("reference_source")
            ) for s in seeds]
            c.executemany(
                "INSERT IGNORE INTO rehab_action_lib "
                "(id, name, body_part, disease_stage, target_muscles, standard_points, "
                "contraindications, standard_pose, error_pose, recommendation, difficulty, create_time, reference_source) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)", seed_rows)
    except Exception as e:
        print(f"导入 action_seed.json 失败: {e}")

    conn.close()


def _validate_protocol_refs():
    """启动校验: rehab_protocols.json 引用的动作是否都存在于动作库, 缺失则打印警告。"""
    try:
        conn = get_db_connection()
        c = conn.cursor()
        c.execute("SELECT name FROM rehab_action_lib")
        known = {r[0] for r in c.fetchall()}
        conn.close()
        missing = set()
        for prob, info in (_PROTOCOLS or {}).items():
            for p in info.get("plans", []):
                for a in p.get("actions", []):
                    if a.get("action") not in known:
                        missing.add(a["action"])
            for a in info.get("solo_actions", []):
                if a not in known:
                    missing.add(a)
        if missing:
            print(f"警告: 以下方案引用的动作不在动作库中, 对其评估将返回404: {', '.join(sorted(missing))}")
    except Exception as e:
        print(f"校验方案引用失败: {e}")


@app.on_event("startup")
def startup():
    init_db()
    _validate_protocol_refs()

# ========== 用户接口 ==========
@app.post("/api/user/register")
def register_user(req: UserAuthRequest):
    if not req.username or not req.password:
        return {"code": 400, "message": "用户名和密码不能为空", "data": None}
    try:
        conn = get_db_connection()
        c = conn.cursor()
        password_hash = hash_password(req.password)
        c.execute("INSERT INTO users (username, password_hash, created_at) VALUES (%s, %s, %s)",
                  (req.username, password_hash, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        conn.close()
        return {"code": 200, "message": "注册成功", "data": {"username": req.username}}
    except pymysql.IntegrityError:
        return {"code": 400, "message": "用户名已存在", "data": None}
    except Exception as e:
        return {"code": 500, "message": f"注册失败：{str(e)}", "data": None}

@app.post("/api/user/login")
def login_user(req: UserAuthRequest):
    if not req.username or not req.password:
        return {"code": 400, "message": "用户名和密码不能为空", "data": None}
    try:
        conn = get_db_connection()
        c = conn.cursor()
        c.execute("SELECT password_hash FROM users WHERE username = %s", (req.username,))
        row = c.fetchone()
        conn.close()
        if not row:
            return {"code": 404, "message": "用户名不存在", "data": None}
        if not verify_password(req.password, row[0]):
            return {"code": 401, "message": "密码错误", "data": None}
        token = create_token(req.username)
        return {"code": 200, "message": "登录成功", "data": {"token": token, "username": req.username}}
    except Exception as e:
        return {"code": 500, "message": f"登录失败：{str(e)}", "data": None}

@app.get("/api/user/info")
def user_info(request: Request):
    username = get_current_user(request)
    if not username:
        return {"code": 401, "message": "Token 无效或已过期，请重新登录", "data": None}
    return {"code": 200, "message": "用户信息获取成功", "data": {"username": username}}

@app.get("/api/user/profile")
def user_profile(request: Request):
    username = get_current_user(request)
    if not username:
        return {"code": 401, "message": "Token 无效或已过期，请重新登录", "data": None}
    try:
        conn = get_db_connection(); c = conn.cursor()
        c.execute("SELECT avatar FROM users WHERE username = %s", (username,))
        row = c.fetchone(); conn.close()
        avatar = row[0] if row and row[0] else None
        return {"code": 200, "message": "用户资料获取成功", "data": {"username": username, "avatar": avatar}}
    except Exception as e:
        return {"code": 500, "message": f"获取资料失败：{str(e)}", "data": None}

@app.post("/api/user/update")
def update_user(req: UserUpdateRequest, request: Request):
    username = get_current_user(request)
    if not username:
        return {"code": 401, "message": "Token 无效或已过期，请重新登录", "data": None}
    try:
        conn = get_db_connection(); c = conn.cursor()
        new_name = (req.username or "").strip() if req.username else None
        new_avatar = req.avatar
        has_change = False
        if new_name and new_name != username:
            c.execute("SELECT username FROM users WHERE username = %s", (new_name,))
            if c.fetchone():
                conn.close(); return {"code": 400, "message": "用户名已存在", "data": None}
            c.execute("UPDATE users SET username = %s WHERE username = %s", (new_name, username))
            c.execute("UPDATE user_tokens SET username = %s WHERE username = %s", (new_name, username))
            username = new_name
            has_change = True
        if new_avatar is not None:
            # 限制头像 base64 大小, 防止超大值
            if len(new_avatar) > 2_000_000:
                conn.close(); return {"code": 400, "message": "头像文件过大", "data": None}
            c.execute("UPDATE users SET avatar = %s WHERE username = %s", (new_avatar, username))
            has_change = True
        conn.close()
        if not has_change:
            return {"code": 200, "message": "无修改", "data": {"username": username, "avatar": new_avatar}}
        return {"code": 200, "message": "更新成功", "data": {"username": username, "avatar": new_avatar}}
    except Exception as e:
        return {"code": 500, "message": f"更新失败：{str(e)}", "data": None}

# ========== 辅助函数 ==========
def get_landmark_coords(landmark, w, h):
    return [landmark.x * w, landmark.y * h]

def analyze_pose(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    with pose_lock:
        results = pose.process(img_rgb)
    if not results.pose_landmarks:
        return None
    landmarks = results.pose_landmarks.landmark
    _update_alps(landmarks)   # TALMA-on-ALPS 轻量版: 实时缓存 ALPS 特征
    h, w = img.shape[:2]

    def coords(idx):
        return get_landmark_coords(landmarks[idx], w, h)

    left_shoulder = coords(11); left_elbow = coords(13); left_wrist = coords(15)
    left_hip = coords(23); left_knee = coords(25); left_ankle = coords(27); left_heel = coords(29); left_toe = coords(31)
    right_shoulder = coords(12); right_elbow = coords(14); right_wrist = coords(16)
    right_hip = coords(24); right_knee = coords(26); right_ankle = coords(28); right_heel = coords(30); right_toe = coords(32)

    angle_left_elbow = calc_angle(left_shoulder, left_elbow, left_wrist)
    angle_right_elbow = calc_angle(right_shoulder, right_elbow, right_wrist)
    angle_left_knee = calc_angle(left_hip, left_knee, left_ankle)
    angle_right_knee = calc_angle(right_hip, right_knee, right_ankle)
    angle_left_shoulder = calc_angle(left_hip, left_shoulder, left_elbow)
    angle_right_shoulder = calc_angle(right_hip, right_shoulder, right_elbow)
    angle_left_hip = calc_angle(left_shoulder, left_hip, left_knee)
    angle_right_hip = calc_angle(right_shoulder, right_hip, right_knee)
    angle_left_ankle_dorsiflexion = calc_angle(left_knee, left_ankle, left_toe)
    angle_right_ankle_dorsiflexion = calc_angle(right_knee, right_ankle, right_toe)
    angle_left_ankle_plantarflexion = calc_angle(left_knee, left_ankle, left_heel)
    angle_right_ankle_plantarflexion = calc_angle(right_knee, right_ankle, right_heel)
    angle_left_ankle = angle_left_ankle_dorsiflexion   # TALMA: 与dorsiflexion同公式, 复用免重复计算
    angle_right_ankle = angle_right_ankle_dorsiflexion

    return {
        "angle_left_elbow": angle_left_elbow,
        "angle_right_elbow": angle_right_elbow,
        "angle_left_knee": angle_left_knee,
        "angle_right_knee": angle_right_knee,
        "angle_left_shoulder": angle_left_shoulder,
        "angle_right_shoulder": angle_right_shoulder,
        "angle_left_hip": angle_left_hip,
        "angle_right_hip": angle_right_hip,
        "angle_left_ankle_dorsiflexion": angle_left_ankle_dorsiflexion,
        "angle_right_ankle_dorsiflexion": angle_right_ankle_dorsiflexion,
        "angle_left_ankle_plantarflexion": angle_left_ankle_plantarflexion,
        "angle_right_ankle_plantarflexion": angle_right_ankle_plantarflexion,
        "angle_left_ankle": angle_left_ankle,
        "angle_right_ankle": angle_right_ankle
    }

# ========== 评估标准缓存(此前每帧评估都查一次库) ==========
_eval_std_cache = {}
_eval_std_lock = Lock()

def _get_eval_standard(action_name: str):
    """按动作名缓存 (name, standard_pose, error_pose, recommendation), 避免每帧连 MySQL。"""
    if not action_name:
        return None
    with _eval_std_lock:
        if action_name in _eval_std_cache:
            return _eval_std_cache[action_name]
    try:
        conn = get_db_connection()
        c = conn.cursor()
        c.execute("SELECT name, standard_pose, error_pose, recommendation FROM rehab_action_lib WHERE name = %s", (action_name,))
        res = c.fetchone()
        conn.close()
    except Exception as e:
        print(f"读取动作标准失败：{e}")
        return None
    with _eval_std_lock:
        _eval_std_cache[action_name] = res
    return res


def _invalidate_eval_std(action_name: str):
    with _eval_std_lock:
        _eval_std_cache.pop(action_name, None)


def _match_error_tip(cond: str, current_angle: float) -> bool:
    """解析错误条件里的比较式, 支持 '>160' / '<30' / 'dorsiflexion<5' 等写法。"""
    m = re.search(r'(>=|<=|>|<)\s*([0-9]+(?:\.[0-9]+)?)', cond or '')
    if not m:
        return False
    op, val = m.group(1), float(m.group(2))
    if op == '>':
        return current_angle > val
    if op == '<':
        return current_angle < val
    if op == '>=':
        return current_angle >= val
    if op == '<=':
        return current_angle <= val
    return False

# ========== 内部评估函数（平滑评分优化版） ==========
def evaluate_action_internal(action_name: str, angle_data: dict, username: str = None, side: str = None):
    """内部调用评估逻辑，返回字典结果（评分平滑版）。
    username 用于按用户隔离 DTW 参考模板; side 用于无侧别动作的患侧选择。"""
    try:
        res = _get_eval_standard(action_name)
        if not res:
            return None

        action_name_db, standard_pose, error_pose, recommendation = res
        standard_pose = json.loads(standard_pose)
        error_pose = json.loads(error_pose)
        recommendation = json.loads(recommendation)

        evaluate_results = []
        overall_standard = True
        side = _resolve_side(action_name, side)
        side_cn = "左" if side == "left" else "右"

        joint_scores = []  # 存储每个关节的得分（0~100）

        for joint, std_range in standard_pose["joint_angles"].items():
            # 确定角度key
            if joint == "main_joint":
                if "臂上举" in action_name or "肩" in action_name:
                    angle_key = f"angle_{side}_shoulder"
                elif "屈膝" in action_name or "膝" in action_name:
                    angle_key = f"angle_{side}_knee"
                elif "肘" in action_name:
                    angle_key = f"angle_{side}_elbow"
                elif "髋" in action_name:
                    angle_key = f"angle_{side}_hip"
                elif "踝" in action_name:
                    angle_key = f"angle_{side}_ankle"   # TALMA: 修复 踝被错误映射为膝角度
                else:
                    angle_key = f"angle_{side}_shoulder"
            elif "knee" in joint.lower():
                angle_key = f"angle_{side}_knee"
            elif "elbow" in joint.lower():
                angle_key = f"angle_{side}_elbow"
            elif "shoulder" in joint.lower():
                angle_key = f"angle_{side}_shoulder"
            elif "hip" in joint.lower():
                angle_key = f"angle_{side}_hip"
            elif "ankle_dorsiflexion" in joint.lower():
                angle_key = f"angle_{side}_ankle_dorsiflexion"
            elif "ankle_plantarflexion" in joint.lower():
                angle_key = f"angle_{side}_ankle_plantarflexion"
            elif "ankle" in joint.lower():
                angle_key = f"angle_{side}_ankle"
            else:
                angle_key = f"angle_{joint}"

            current_angle = round(angle_data.get(angle_key, 0), 1)
            min_angle, max_angle = std_range
            is_standard = min_angle <= current_angle <= max_angle
            if not is_standard:
                overall_standard = False

            # 计算单关节得分（平滑版）
            mid = (min_angle + max_angle) / 2
            half = (max_angle - min_angle) / 2
            if half == 0:
                half = 1  # 避免除零

            if is_standard:
                # 范围内：得分 100 - 15 * (偏离中值的比例)
                deviation_ratio = abs(current_angle - mid) / half
                joint_score = 100 - 15 * deviation_ratio
            else:
                # 范围外：计算距离边界的绝对差值
                if current_angle < min_angle:
                    diff = min_angle - current_angle
                else:
                    diff = current_angle - max_angle
                # 最大偏差阈值 100°，超过则 30 分，否则线性扣分
                max_diff = 100.0
                if diff >= max_diff:
                    joint_score = 30.0
                else:
                    joint_score = 85.0 - 55.0 * (diff / max_diff)
                joint_score = max(30.0, joint_score)  # 确保不低于30

            joint_score = round(max(0, min(100, joint_score)), 1)
            joint_scores.append(joint_score)

            # 生成错误提示
            error_tip = "动作标准✅" if is_standard else "动作不标准，请调整"
            if not is_standard:
                for error_type, error_info in error_pose.items():
                    if _match_error_tip(error_info.get("angle", ""), current_angle):
                        error_tip = error_info["tip"]
                        break

            evaluate_results.append({
                "joint": joint,
                "current_angle": current_angle,
                "standard_range": f"{min_angle}-{max_angle}度",
                "is_standard": is_standard,
                "error_tip": error_tip
            })

        # 总分取所有关节得分的平均值
        if joint_scores:
            overall_score = round(sum(joint_scores) / len(joint_scores), 1)
        else:
            overall_score = 0.0

        result = {
            "action_name": action_name_db,
            "overall_standard": overall_standard,
            "score": overall_score,
            "side_used": side_cn + "侧",
            "joint_evaluate": evaluate_results,
            "recommendation": recommendation,
            "feedback": "动作达标，继续保持！" if overall_standard else "动作不标准，请按提示调整"
        }
        # TALMA-on-ALPS 轻量版: 自动增强(机位鲁棒 ALPS 描述子 + 时序一致性 + 统一质量分)
        aug = _alps_augment(action_name_db, overall_standard, overall_score, username)
        result.update(aug)
        # score 直接用增强后的 quality(无参考时 quality==score, 不改变行为);
        # 原始瞬时角度分放到 score_frame 作为兜底。
        result["score_frame"] = overall_score
        result["score"] = aug.get("quality", overall_score)
        return result
    except Exception as e:
        print(f"内部评估错误：{e}")
        return None

# ========== API 接口 ==========

@app.post("/api/pose/analyze_with_eval")
async def analyze_image_with_eval(request: Request, file: UploadFile = File(...), action_name: str = Form(...), side: str = Form(None)):
    """图片分析并评估（增强接口）。CV/DB 为阻塞操作, 丢线程池执行避免卡事件循环。"""
    try:
        contents = await file.read()
        username = get_current_user(request)

        def _work():
            img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                return ("decode", None)
            angle_data = analyze_pose(img)
            if not angle_data:
                return ("nopose", None)
            return ("ok", evaluate_action_internal(action_name, angle_data, username))

        tag, payload = await run_in_threadpool(_work)
        if tag == "decode":
            return {"code": 400, "message": "无法解码图片", "data": None}
        if tag == "nopose":
            return {"code": 400, "message": "未识别到人体姿态", "data": None}
        if not payload:
            return {"code": 404, "message": f"未找到动作 {action_name} 的评估标准", "data": None}
        return {"code": 200, "message": "评估成功", "data": payload}
    except Exception as e:
        return {"code": 500, "message": f"分析评估失败：{str(e)}", "data": None}

@app.post("/api/pose/analyze_frame")
async def analyze_frame(request: Request, file: UploadFile = File(...), action_name: str = Form(None), side: str = Form(None)):
    """方案1: 前端逐帧上传图片 -> 返回 角度 + landmark(供前端画骨架) + ALPS + 可选评估。"""
    try:
        contents = await file.read()
        username = get_current_user(request)

        def _work():
            img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                return None
            with pose_lock:
                results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if not results.pose_landmarks:
                return None
            lm = results.pose_landmarks.landmark
            h, w = img.shape[:2]

            def coords(idx):
                return get_landmark_coords(lm[idx], w, h)

            # 归一化 landmark(供前端画骨架)
            landmarks = [(round(float(p.x), 4), round(float(p.y), 4), round(float(p.z), 4)) for p in lm]
            angle_data = {
                "angle_left_elbow": round(calc_angle(coords(11), coords(13), coords(15)), 1),
                "angle_right_elbow": round(calc_angle(coords(12), coords(14), coords(16)), 1),
                "angle_left_knee": round(calc_angle(coords(23), coords(25), coords(27)), 1),
                "angle_right_knee": round(calc_angle(coords(24), coords(26), coords(28)), 1),
                "angle_left_shoulder": round(calc_angle(coords(23), coords(11), coords(13)), 1),
                "angle_right_shoulder": round(calc_angle(coords(24), coords(12), coords(14)), 1),
                "angle_left_hip": round(calc_angle(coords(11), coords(23), coords(25)), 1),
                "angle_right_hip": round(calc_angle(coords(12), coords(24), coords(26)), 1),
                "angle_left_ankle_dorsiflexion": round(calc_angle(coords(25), coords(27), coords(31)), 1),
                "angle_right_ankle_dorsiflexion": round(calc_angle(coords(26), coords(28), coords(32)), 1),
                "angle_left_ankle_plantarflexion": round(calc_angle(coords(25), coords(27), coords(29)), 1),
                "angle_right_ankle_plantarflexion": round(calc_angle(coords(26), coords(28), coords(30)), 1),
                "angle_left_ankle": round(calc_angle(coords(25), coords(27), coords(31)), 1),
                "angle_right_ankle": round(calc_angle(coords(26), coords(28), coords(32)), 1),
            }
            _update_alps(lm)   # 维持 ALPS 滚动缓存(方案1 也生效)
            return landmarks, angle_data

        work_res = await run_in_threadpool(_work)
        if not work_res:
            return {"code": 400, "message": "未检测到人体姿态", "data": None}
        landmarks, angle_data = work_res
        alps = None
        with alps_seq_lock:
            alps = latest_alps_feature
        eval_result = None
        if action_name:
            eval_result = evaluate_action_internal(action_name, angle_data, username, side)
        return {"code": 200, "message": "OK",
                "data": {"angles": angle_data, "landmarks": landmarks, "alps": alps, "eval": eval_result}}
    except Exception as e:
        return {"code": 500, "message": f"分析失败：{str(e)}", "data": None}

from collections import namedtuple
_LMK = namedtuple("_LMK", "x y z")

@app.post("/api/pose/analyze_landmarks")
def analyze_landmarks(payload: dict, request: Request):
    """
    A′(客户端姿态估计): 前端在浏览器算出 landmark(33个归一化 [x,y,z])后,
    交给后端做 判定(角度/ALPS/DTW/评分)。landmark 很小, 判定往返极快。
    同步 def 由 FastAPI 自动放入线程池, 不阻塞事件循环。
    """
    try:
        lms = payload.get("landmarks")
        action_name = payload.get("action_name")
        side = payload.get("side")
        if not lms or len(lms) < 33:
            return {"code": 400, "message": "landmarks 需为 33 个 [x,y,z]", "data": None}
        pts = [(float(p[0]), float(p[1]), float(p[2])) for p in lms[:33]]

        def coords(idx):
            return [pts[idx][0], pts[idx][1]]   # 归一化坐标算角度与像素一致(角度对缩放不变)

        angle_data = {
            "angle_left_elbow": round(calc_angle(coords(11), coords(13), coords(15)), 1),
            "angle_right_elbow": round(calc_angle(coords(12), coords(14), coords(16)), 1),
            "angle_left_knee": round(calc_angle(coords(23), coords(25), coords(27)), 1),
            "angle_right_knee": round(calc_angle(coords(24), coords(26), coords(28)), 1),
            "angle_left_shoulder": round(calc_angle(coords(23), coords(11), coords(13)), 1),
            "angle_right_shoulder": round(calc_angle(coords(24), coords(12), coords(14)), 1),
            "angle_left_hip": round(calc_angle(coords(11), coords(23), coords(25)), 1),
            "angle_right_hip": round(calc_angle(coords(12), coords(24), coords(26)), 1),
            "angle_left_ankle_dorsiflexion": round(calc_angle(coords(25), coords(27), coords(31)), 1),
            "angle_right_ankle_dorsiflexion": round(calc_angle(coords(26), coords(28), coords(32)), 1),
            "angle_left_ankle_plantarflexion": round(calc_angle(coords(25), coords(27), coords(29)), 1),
            "angle_right_ankle_plantarflexion": round(calc_angle(coords(26), coords(28), coords(30)), 1),
            "angle_left_ankle": round(calc_angle(coords(25), coords(27), coords(31)), 1),
            "angle_right_ankle": round(calc_angle(coords(26), coords(28), coords(32)), 1),
        }
        # _update_alps 需要带 .x/.y/.z 的对象
        _update_alps([_LMK(x, y, z) for x, y, z in pts])
        alps = None
        with alps_seq_lock:
            alps = latest_alps_feature
        username = get_current_user(request)
        eval_result = evaluate_action_internal(action_name, angle_data, username, side) if action_name else None
        return {"code": 200, "message": "OK",
                "data": {"angles": angle_data, "alps": alps, "eval": eval_result}}
    except Exception as e:
        return {"code": 400, "message": f"判定失败: {str(e)}", "data": None}

# ====================== TALMA-on-ALPS 轻量版: 时序得分接口 ======================
# 参考模板按用户隔离: 键为 "用户名::动作名", 避免第一个达标用户成为所有人的标准
_REF_STORE = {}            # ref_key -> 特征序列(参考模板)
_REF_FILE = "talma_refs.json"


def _ref_key(username, action_name):
    return f"{username or '_anon'}::{action_name}"


def _lm33(frame):
    """把 [[x,y,z] x 33] 规整为 [(x,y,z) x 33], 不足 33 补 0。"""
    out = [(float(p[0]), float(p[1]), float(p[2])) for p in frame]
    while len(out) < 33:
        out.append((0.0, 0.0, 0.0))
    return out[:33]


def _load_ref(username, name):
    """按 (用户, 动作名) 加载已保存的参考特征序列; 没有则 None。"""
    if not name:
        return None
    return _REF_STORE.get(_ref_key(username, name))


def _save_refs():
    """持久化参考模板到 JSON 文件。"""
    try:
        with open(_REF_FILE, "w", encoding="utf-8") as f:
            json.dump(_REF_STORE, f)
    except Exception as e:
        print(f"保存参考模板失败: {e}")


def _load_refs():
    """启动时从 JSON 文件加载参考模板。"""
    global _REF_STORE
    try:
        if os.path.exists(_REF_FILE):
            with open(_REF_FILE, "r", encoding="utf-8") as f:
                _REF_STORE = json.load(f)
    except Exception:
        pass


# 方案2: 从 rehab_action_lib 的 standard_pose 为每个动作推导"标准目标", 让判定从第一次就有基准
_ACTION_STD = {}
_ACTION_STD_BUILT = False


def _build_action_standards():
    """读取动作库 standard_pose 的 joint_angles, 为每个动作计算标准目标角度(中值)。"""
    global _ACTION_STD, _ACTION_STD_BUILT
    if _ACTION_STD_BUILT:
        return
    try:
        conn = get_db_connection()
        c = conn.cursor()
        c.execute("SELECT name, standard_pose FROM rehab_action_lib")
        rows = c.fetchall()
        conn.close()
        for name, sp_json in rows:
            try:
                sp = json.loads(sp_json)
                ja = sp.get("joint_angles", {})
                targets = {}
                for j, rng in ja.items():
                    if isinstance(rng, (list, tuple)) and len(rng) >= 2:
                        targets[j] = round((float(rng[0]) + float(rng[1])) / 2.0, 1)
                if targets:
                    _ACTION_STD[name] = targets
            except Exception:
                continue
    except Exception as e:
        print(f"构建动作标准目标失败: {e}")
    _ACTION_STD_BUILT = True


def _angle_fit(angle, target, soft=12.0):
    """当前角度与标准目标角的接近度(0~1), soft 为允许偏差。"""
    try:
        diff = abs(float(angle) - float(target))
        return max(0.0, min(1.0, 1.0 - diff / soft))
    except Exception:
        return 0.0


# 方案数据层: "病症 -> 训练方案(有序动作序列+组次数)"
_PROTOCOLS = {}


def _load_protocols():
    global _PROTOCOLS
    try:
        with open("rehab_protocols.json", "r", encoding="utf-8") as f:
            _PROTOCOLS = json.load(f)
    except Exception as e:
        print(f"加载康复方案失败: {e}")


def _alps_augment(action_name, is_standard, base_score, username=None):
    """
    评估结果自动增强(在 evaluate_action_internal 内被调用):
      - alps               : 当前帧机位/缩放鲁棒的 ALPS 姿态描述子(自动附加)
      - temporal_similarity: 实时窗口与"当前用户"该动作参考模板的 DTW 时序一致度(0~1, 无模板则 None)
      - quality            : 统一质量分(0~100), 有时序参考时=0.7*角度分+0.3*时序*100, 否则=角度分
      - auto_ref_saved     : 本次是否自动采集了该动作的参考模板
      - temporal_note      : 说明文字
    序列先降采样到 DTW_SEQ_CAP 帧再对齐, 控制纯 Python DTW 的计算量。
    """
    out = {
        "alps": None,
        "temporal_similarity": None,
        "auto_ref_saved": False,
        "temporal_note": None,
        "standard": None,
        "standard_fit": round(float(base_score) / 100.0, 3),
        "quality": round(float(base_score), 1),
    }
    try:
        # 方案2: 从动作库为标准推导/预置每个动作的标准目标(判定从第一次就有基准)
        _build_action_standards()
        out["standard"] = _ACTION_STD.get(action_name)
        with alps_seq_lock:
            feat = latest_alps_feature
            seq = _downsample_seq(_trim_idle_frames(list(latest_alps_seq)))
        if feat is not None:
            out["alps"] = feat
        if username:
            ref = _load_ref(username, action_name)
            if ref and len(seq) >= 5:
                ref = _downsample_seq(ref)
                sim = alps_lite.evaluate_segment(ref, seq, penalty=0.2, top_percent=0.3)["similarity"]
                out["temporal_similarity"] = sim
                out["quality"] = round(0.7 * float(base_score) + 0.3 * sim * 100.0, 1)
            # 首次达标且无模板: 自动采集当前滚动窗口为该用户该动作的参考(一次性引导)
            if is_standard and seq and not ref:
                _REF_STORE[_ref_key(username, action_name)] = seq
                _save_refs()
                out["auto_ref_saved"] = True
                out["temporal_note"] = "已自动采集本次达标动作为您的个人参考模板, 后续评估将自动做时序对比。"
    except Exception:
        pass
    return out


_load_refs()
_load_protocols()


@app.get("/api/pose/alps")
def get_current_alps():
    """返回当前帧的 ALPS 姿态特征(机位鲁棒描述子)。"""
    with alps_seq_lock:
        if latest_alps_feature is None:
            return {"code": 404, "message": "暂无姿态数据，请确保摄像头工作", "data": None}
        return {"code": 200, "message": "获取成功", "data": latest_alps_feature}


@app.get("/api/rehab/protocols")
def rehab_protocols():
    """返回所有"病症->训练方案"数据层(有序动作序列+组次数)。"""
    return {"code": 200, "message": "OK", "data": _PROTOCOLS}


@app.post("/api/rehab/sequence_score")
def sequence_score(req: SequenceScoreRequest, request: Request):
    """TALMA 轻量版: DTW 对齐 参考序列与患者序列(或实时窗口), 返回匹配+时序相似度。"""
    try:
        username = get_current_user(request)
        ref_seq = None
        if req.reference:
            ref_seq = alps_lite.sequence_features([_lm33(f) for f in req.reference])
        elif req.reference_action:
            saved = _load_ref(username, req.reference_action)
            if saved:
                ref_seq = saved
        if not ref_seq:
            return {"code": 400, "message": "缺少参考序列(reference 或 reference_action)", "data": None}

        if req.use_rolling and not req.patient:
            with alps_seq_lock:
                pat_seq = _downsample_seq(_trim_idle_frames(list(latest_alps_seq)))
        else:
            pat_seq = alps_lite.sequence_features([_lm33(f) for f in (req.patient or [])])
        if not pat_seq:
            return {"code": 400, "message": "缺少患者序列(patient 或实时数据)", "data": None}

        res = alps_lite.evaluate_segment(_downsample_seq(ref_seq), pat_seq, penalty=req.penalty, top_percent=req.top_percent)
        return {"code": 200, "message": "OK", "data": res}
    except Exception as e:
        return {"code": 500, "message": f"时序打分失败: {e}", "data": None}


@app.post("/api/rehab/capture_ref")
def capture_ref(request: Request, action_name: str):
    """把当前实时滚动窗口保存为当前用户某动作的参考模板, 供 sequence_score 复用。"""
    username = get_current_user(request)
    if not username:
        return {"code": 401, "message": "请先登录后采集参考模板", "data": None}
    with alps_seq_lock:
        seq = _downsample_seq(_trim_idle_frames(list(latest_alps_seq)))
    if not seq:
        return {"code": 400, "message": "当前没有采集到姿态序列", "data": None}
    _REF_STORE[_ref_key(username, action_name)] = seq
    _save_refs()
    return {"code": 200, "message": f"已保存动作 [{action_name}] 的个人参考序列", "data": {"len": len(seq)}}


# ========== 康复记录: 保存(含明细) + 查询 ==========
@app.post("/api/rehab/save")
def save_record(request: Request, req: SaveRecordRequest):
    username = get_current_user(request)
    if not username:
        return {"code": 401, "message": "请先登录后保存记录", "data": None}
    try:
        conn = get_db_connection()
        c = conn.cursor()
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        detail_json = json.dumps(req.detail, ensure_ascii=False) if req.detail else None
        c.execute("INSERT INTO records (`user`, action, score, time, detail) VALUES (%s, %s, %s, %s, %s)",
                  (username, req.action, req.score, current_time, detail_json))
        conn.close()
        return {"code": 200, "message": "记录保存成功", "data": {"time": current_time}}
    except Exception as e:
        return {"code": 500, "message": f"保存失败：{str(e)}", "data": None}

@app.get("/api/rehab/query")
def query_record(request: Request):
    username = get_current_user(request)
    if not username:
        return {"code": 401, "message": "请先登录后查询记录", "data": None}
    try:
        conn = get_db_connection()
        c = conn.cursor()
        c.execute("SELECT action, score, time, detail FROM records WHERE `user` = %s ORDER BY time DESC", (username,))
        records = []
        for r in c.fetchall():
            detail = None
            if r[3]:
                try:
                    detail = json.loads(r[3])
                except Exception:
                    detail = None
            records.append({"action": r[0], "score": r[1], "time": r[2], "detail": detail})
        conn.close()
        return {"code": 200, "message": "查询成功", "data": records}
    except Exception as e:
        return {"code": 500, "message": f"查询失败：{str(e)}", "data": None}

# ========== 动作库: 查询 / 详情 / 新增 / 修改 / 删除(写操作需登录) ==========
_SUPPORTED_JOINTS = {
    "shoulder", "elbow", "wrist", "hip", "knee", "ankle",
    "ankle_dorsiflexion", "ankle_plantarflexion",
    "main_joint", "shoulder_flexion", "elbow_flexion", "knee_flexion"
}


def _validate_action_payload(payload: ActionLibRequest):
    """校验 JSON 字段与关节类型; 返回错误信息或 None。"""
    try:
        json.loads(payload.standard_pose)
        json.loads(payload.error_pose)
        json.loads(payload.recommendation)
    except Exception:
        return "标准角度、错误提示或训练建议 JSON 格式错误"
    try:
        joint_angles = json.loads(payload.standard_pose).get("joint_angles", {})
    except Exception:
        return "standard_pose 解析失败"
    unsupported = [j for j in joint_angles.keys() if j not in _SUPPORTED_JOINTS]
    if unsupported:
        return f"不支持的关节类型：{','.join(unsupported)}"
    return None


def _action_row_to_dict(r):
    return {
        "id": r[0], "name": r[1], "body_part": r[2], "disease_stage": r[3],
        "target_muscles": r[4], "standard_points": r[5], "contraindications": r[6],
        "standard_pose": json.loads(r[7]), "error_pose": json.loads(r[8]),
        "recommendation": json.loads(r[9]), "difficulty": r[10], "create_time": r[11],
        "reference_source": (r[12] if len(r) > 12 else None)
    }


@app.get("/api/action_lib/list")
def list_action_lib(body_part: str = None):
    try:
        conn = get_db_connection()
        c = conn.cursor()
        if body_part:
            c.execute("SELECT * FROM rehab_action_lib WHERE body_part = %s", (body_part,))
        else:
            c.execute("SELECT * FROM rehab_action_lib")
        actions = [_action_row_to_dict(r) for r in c.fetchall()]
        conn.close()
        return {"code": 200, "message": "查询成功", "data": actions}
    except Exception as e:
        return {"code": 500, "message": f"查询失败：{str(e)}", "data": None}

@app.get("/api/action_lib/detail/{action_id}")
def get_action_detail(action_id: str):
    try:
        conn = get_db_connection()
        c = conn.cursor()
        c.execute("SELECT * FROM rehab_action_lib WHERE id = %s", (action_id,))
        r = c.fetchone()
        conn.close()
        if not r:
            return {"code": 404, "message": "动作不存在", "data": None}
        return {"code": 200, "message": "查询成功", "data": _action_row_to_dict(r)}
    except Exception as e:
        return {"code": 500, "message": f"查询失败：{str(e)}", "data": None}

@app.post("/api/action_lib/add")
def add_action(req: ActionLibRequest, request: Request):
    username = get_current_user(request)
    if not username:
        return {"code": 401, "message": "请先登录后管理动作库", "data": None}
    err = _validate_action_payload(req)
    if err:
        return {"code": 400, "message": err, "data": None}
    try:
        action_id = str(uuid.uuid4())
        create_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        conn = get_db_connection()
        c = conn.cursor()
        c.execute("""
            INSERT INTO rehab_action_lib
            (id, name, body_part, disease_stage, target_muscles, standard_points,
             contraindications, standard_pose, error_pose, recommendation, difficulty, create_time, reference_source)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """, (action_id, req.name, req.body_part, req.disease_stage, req.target_muscles,
              req.standard_points, req.contraindications, req.standard_pose,
              req.error_pose, req.recommendation, req.difficulty, create_time, req.reference_source))
        conn.close()
        _invalidate_eval_std(req.name)
        return {"code": 200, "message": "新增动作成功", "data": {"action_id": action_id}}
    except pymysql.IntegrityError:
        return {"code": 400, "message": "同名动作已存在", "data": None}
    except Exception as e:
        return {"code": 500, "message": f"新增失败：{str(e)}", "data": None}

@app.put("/api/action_lib/update/{action_id}")
def update_action(action_id: str, req: ActionLibRequest, request: Request):
    username = get_current_user(request)
    if not username:
        return {"code": 401, "message": "请先登录后管理动作库", "data": None}
    err = _validate_action_payload(req)
    if err:
        return {"code": 400, "message": err, "data": None}
    try:
        conn = get_db_connection()
        c = conn.cursor()
        c.execute("SELECT name FROM rehab_action_lib WHERE id = %s", (action_id,))
        row = c.fetchone()
        if not row:
            conn.close()
            return {"code": 404, "message": "动作不存在", "data": None}
        old_name = row[0]
        if req.name != old_name:
            c.execute("SELECT id FROM rehab_action_lib WHERE name = %s AND id != %s", (req.name, action_id))
            if c.fetchone():
                conn.close()
                return {"code": 400, "message": "同名动作已存在", "data": None}
        c.execute("""
            UPDATE rehab_action_lib
            SET name=%s, body_part=%s, disease_stage=%s, target_muscles=%s,
                standard_points=%s, contraindications=%s, standard_pose=%s,
                error_pose=%s, recommendation=%s, difficulty=%s, reference_source=%s
            WHERE id=%s
        """, (req.name, req.body_part, req.disease_stage, req.target_muscles,
              req.standard_points, req.contraindications, req.standard_pose,
              req.error_pose, req.recommendation, req.difficulty, req.reference_source, action_id))
        conn.close()
        _invalidate_eval_std(old_name)
        _invalidate_eval_std(req.name)
        return {"code": 200, "message": "修改动作成功", "data": {"action_id": action_id}}
    except Exception as e:
        return {"code": 500, "message": f"修改失败：{str(e)}", "data": None}

@app.delete("/api/action_lib/delete/{action_id}")
def delete_action(action_id: str, request: Request):
    username = get_current_user(request)
    if not username:
        return {"code": 401, "message": "请先登录后管理动作库", "data": None}
    try:
        conn = get_db_connection()
        c = conn.cursor()
        c.execute("SELECT name FROM rehab_action_lib WHERE id = %s", (action_id,))
        row = c.fetchone()
        if not row:
            conn.close()
            return {"code": 404, "message": "动作不存在", "data": None}
        c.execute("DELETE FROM rehab_action_lib WHERE id = %s", (action_id,))
        conn.close()
        _invalidate_eval_std(row[0])
        return {"code": 200, "message": f"已删除动作 {row[0]}", "data": {"action_id": action_id}}
    except Exception as e:
        return {"code": 500, "message": f"删除失败：{str(e)}", "data": None}


@app.post("/api/rehab/evaluate")
def evaluate_action(req: EvaluateRequest, request: Request):
    eval_result = evaluate_action_internal(req.action_name, req.angle_data, get_current_user(request), req.side)
    if not eval_result:
        return {"code": 404, "message": f"未找到{req.action_name}的康复标准", "data": None}
    return {"code": 200, "message": "动作评估成功", "data": eval_result}


# ========== 静态文件白名单(不再把整个项目目录暴露为站点) ==========
_ALLOWED_STATIC = {"index.html", "login.html", "logo1.jpg"}

def _serve_static(name: str):
    if name not in _ALLOWED_STATIC or not os.path.exists(name):
        return JSONResponse(status_code=404,
                            content={"code": 404, "message": "Not Found", "data": None})
    return FileResponse(name)

@app.get("/")
def serve_index():
    return _serve_static("index.html")

@app.get("/index.html")
def serve_index_html():
    return _serve_static("index.html")

@app.get("/login.html")
def serve_login_html():
    return _serve_static("login.html")

@app.get("/logo1.jpg")
def serve_logo():
    return _serve_static("logo1.jpg")

# ====================== 一键启动 + 自动打开前端 ======================
if __name__ == "__main__":
    HOST = "127.0.0.1"
    PORT = 8000

    def _open_browser():
        try:
            webbrowser.open(f"http://{HOST}:{PORT}/")
        except Exception:
            pass

    if os.getenv('NO_BROWSER', '0') != '1':
        threading.Timer(2.0, _open_browser).start()

    uvicorn.run(app, host=HOST, port=PORT, log_level="error")
