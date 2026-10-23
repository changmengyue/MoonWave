"""用户注册 / 登录 / Token 鉴权。"""

import pymysql
import pytest

import zitaishibie as zt


def test_register_success(client, db):
    r = client.post("/api/user/register", json={"username": "alice", "password": "pw123"})
    assert r.status_code == 200
    assert r.json()["code"] == 200
    assert r.json()["data"]["username"] == "alice"
    assert any("INSERT INTO users" in sql for sql in db.executed)


@pytest.mark.parametrize("payload", [{"username": "", "password": "pw"}, {"username": "a", "password": ""}])
def test_register_rejects_empty_fields(client, db, payload):
    r = client.post("/api/user/register", json=payload)
    assert r.json()["code"] == 400
    assert db.sql_log == []          # 参数校验失败时不该碰数据库


def test_register_conflict_returns_400(client, db):
    db.fail_next = pymysql.err.IntegrityError(1062, "Duplicate entry 'alice' for key 'users.username'")
    r = client.post("/api/user/register", json={"username": "alice", "password": "pw123"})
    assert r.json()["code"] == 400
    assert "已存在" in r.json()["message"]


def test_login_unknown_user(client, db):
    db.push()                                    # SELECT password_hash -> 空结果
    r = client.post("/api/user/login", json={"username": "ghost", "password": "pw"})
    assert r.json()["code"] == 404


def test_login_wrong_password(client, db):
    db.push((zt.hash_password("correct"),))
    r = client.post("/api/user/login", json={"username": "alice", "password": "wrong"})
    assert r.json()["code"] == 401


def test_login_issues_token(client, db):
    db.push((zt.hash_password("correct"),))      # 校验密码
    db.push()                                    # REPLACE INTO user_tokens
    r = client.post("/api/user/login", json={"username": "alice", "password": "correct"})
    body = r.json()
    assert body["code"] == 200
    assert body["data"]["token"]
    assert any("REPLACE INTO user_tokens" in sql for sql in db.executed)


def test_user_info_requires_token(client, db):
    r = client.get("/api/user/info")
    assert r.json()["code"] == 401
    assert db.sql_log == []


def test_user_info_with_valid_token(client, db):
    db.push(("alice", None))
    r = client.get("/api/user/info", headers={"Authorization": "Bearer test-token"})
    assert r.json()["code"] == 200
    assert r.json()["data"]["username"] == "alice"


def test_user_info_rejects_expired_token(client, db):
    db.push(("alice", "2000-01-01 00:00:00"))    # 已过期
    db.push()                                    # DELETE FROM user_tokens
    r = client.get("/api/user/info", headers={"Authorization": "Bearer stale-token"})
    assert r.json()["code"] == 401


def test_user_info_rejects_non_bearer_header(client, db):
    response = client.get("/api/user/info", headers={"Authorization": "Basic dXNlcjpwdw=="})
    assert response.json()["code"] == 401


def test_profile_hides_avatar_when_null(client, db):
    db.push(("alice", None))                     # token
    db.push((None,))                             # SELECT avatar
    r = client.get("/api/user/profile", headers={"Authorization": "Bearer test-token"})
    assert r.json()["code"] == 200
    assert r.json()["data"]["avatar"] is None


def test_update_rejects_oversized_avatar(client, db):
    db.push(("alice", None))
    r = client.post(
        "/api/user/update",
        json={"avatar": "x" * 2_000_001},
        headers={"Authorization": "Bearer test-token"},
    )
    assert r.json()["code"] == 400
    assert "过大" in r.json()["message"]
