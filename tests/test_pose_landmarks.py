"""A′ 方案的核心接口：浏览器传 33 个关键点，后端算角度与评分。"""

import pytest

LANDMARKS = [[0.5, 0.5, 0.0]] * 33

ANGLE_KEYS = {
    "angle_left_elbow", "angle_right_elbow",
    "angle_left_knee", "angle_right_knee",
    "angle_left_shoulder", "angle_right_shoulder",
    "angle_left_hip", "angle_right_hip",
    "angle_left_ankle", "angle_right_ankle",
    "angle_left_ankle_dorsiflexion", "angle_right_ankle_dorsiflexion",
    "angle_left_ankle_plantarflexion", "angle_right_ankle_plantarflexion",
}


def test_analyze_landmarks_returns_all_angles(client):
    r = client.post("/api/pose/analyze_landmarks", json={"landmarks": LANDMARKS})
    body = r.json()
    assert body["code"] == 200
    angles = body["data"]["angles"]
    assert set(angles) == ANGLE_KEYS
    assert all(0.0 <= v <= 180.0 for v in angles.values()), angles


def test_analyze_landmarks_works_without_login(client):
    """A′ 接口不要求登录：未登录时 eval 为空，但角度照算。"""
    r = client.post("/api/pose/analyze_landmarks", json={"landmarks": LANDMARKS})
    assert r.json()["code"] == 200
    assert r.json()["data"]["eval"] is None


@pytest.mark.parametrize(
    "payload",
    [
        {},                                              # 完全没有 landmarks
        {"landmarks": LANDMARKS[:10]},                   # 点数不足
        {"landmarks": [["x", "y", "z"]] * 33},           # 非数值
        {"landmarks": [[0.1, 0.2]] * 33},                # 少一个维度
    ],
)
def test_analyze_landmarks_rejects_bad_payload(client, payload):
    r = client.post("/api/pose/analyze_landmarks", json=payload)
    assert r.json()["code"] == 400


def test_analyze_landmarks_with_unknown_action_keeps_angles(client, db):
    r = client.post(
        "/api/pose/analyze_landmarks",
        json={"landmarks": LANDMARKS, "action_name": "不存在的动作"},
    )
    body = r.json()
    assert body["code"] == 200
    assert body["data"]["angles"]
    assert body["data"]["eval"] is None      # 找不到标准就只回角度，不报错


def test_straight_limb_angle_is_close_to_180(client):
    """三点共线时角度应接近 180°，用于验证 calc_angle 的几何计算。"""
    landmarks = [[0.5, 0.5, 0.0] for _ in range(33)]
    landmarks[11] = [0.5, 0.2, 0.0]     # 左肩
    landmarks[13] = [0.5, 0.4, 0.0]     # 左肘
    landmarks[15] = [0.5, 0.6, 0.0]     # 左腕
    r = client.post("/api/pose/analyze_landmarks", json={"landmarks": landmarks})
    assert abs(r.json()["data"]["angles"]["angle_left_elbow"] - 180.0) <= 1.0
