"""康复动作库的增删改查与鉴权。

表结构里 standard_pose / error_pose / recommendation 三列存的是 JSON 字符串，
接口负责序列化与反序列化，这里用一行真实形状的数据来锁住这个约定。
"""

import json

ACTION_ROW = (
    "id-001", "深蹲", "knee", "early", "股四头肌", "膝盖不超过脚尖", "膝关节置换术后",
    json.dumps({"joint_angles": {"knee_flexion": [60, 100]}}, ensure_ascii=False),
    json.dumps({"knee_flexion": "膝盖超过脚尖"}, ensure_ascii=False),
    json.dumps(["保持背部挺直"], ensure_ascii=False),
    2, "2026-01-01 00:00:00", "《康复评定学》",
)

VALID_PAYLOAD = {
    "name": "站姿提踵",
    "body_part": "ankle",
    "disease_stage": "late",
    "target_muscles": "小腿三头肌",
    "standard_points": "脚跟抬到最高",
    "contraindications": "跟腱断裂急性期",
    "standard_pose": json.dumps({"joint_angles": {"ankle_dorsiflexion": [10, 30]}}),
    "error_pose": json.dumps({"ankle_dorsiflexion": "幅度不足"}),
    "recommendation": json.dumps(["扶墙保持平衡"]),
    "difficulty": 1,
}


def test_list_actions_returns_rows(client, db):
    db.push(ACTION_ROW)
    r = client.get("/api/action_lib/list")
    body = r.json()
    assert body["code"] == 200
    assert len(body["data"]) == 1
    item = body["data"][0]
    assert item["name"] == "深蹲"
    assert item["standard_pose"]["joint_angles"]["knee_flexion"] == [60, 100]
    assert item["difficulty"] == 2


def test_list_actions_can_filter_by_body_part(client, db):
    db.push(ACTION_ROW)
    r = client.get("/api/action_lib/list", params={"body_part": "knee"})
    assert r.json()["code"] == 200
    assert any("WHERE body_part = %s" in sql for sql in db.executed)


def test_action_detail_not_found(client, db):
    db.push()
    r = client.get("/api/action_lib/detail/nope")
    assert r.json()["code"] == 404


def test_add_action_requires_login(client, db):
    r = client.post("/api/action_lib/add", json=VALID_PAYLOAD)
    assert r.json()["code"] == 401
    assert db.sql_log == []


def test_add_action_rejects_invalid_json(client, db):
    db.push(("alice", None))                     # token 校验
    payload = dict(VALID_PAYLOAD, standard_pose="not-json")
    r = client.post("/api/action_lib/add", json=payload,
                    headers={"Authorization": "Bearer test-token"})
    assert r.json()["code"] == 400
    assert "JSON" in r.json()["message"]


def test_add_action_rejects_unsupported_joint(client, db):
    db.push(("alice", None))
    payload = dict(VALID_PAYLOAD, standard_pose=json.dumps({"joint_angles": {"tail_wag": [0, 1]}}))
    r = client.post("/api/action_lib/add", json=payload,
                    headers={"Authorization": "Bearer test-token"})
    assert r.json()["code"] == 400
    assert "不支持的关节" in r.json()["message"]


def test_add_action_success(client, db):
    db.push(("alice", None))                     # token 校验
    db.push()                                    # INSERT
    r = client.post("/api/action_lib/add", json=VALID_PAYLOAD,
                    headers={"Authorization": "Bearer test-token"})
    body = r.json()
    assert body["code"] == 200
    assert body["data"]["action_id"]
    assert any("INSERT INTO rehab_action_lib" in sql for sql in db.executed)


def test_delete_action_not_found(client, db):
    db.push(("alice", None))
    db.push()                                    # SELECT name -> 空
    r = client.delete("/api/action_lib/delete/missing",
                      headers={"Authorization": "Bearer test-token"})
    assert r.json()["code"] == 404
