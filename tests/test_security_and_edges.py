"""边界与安全属性：白名单静态文件、已下线接口、CORS、时序接口入参。"""

import pytest

LANDMARKS = [[0.5, 0.5, 0.0]] * 33


@pytest.mark.parametrize(
    "path",
    ["/.git/config", "/zitaishibie.py", "/requirements.txt", "/talma_refs.json"],
)
def test_whitelist_blocks_project_files(client, path):
    """只有 index.html / login.html / logo1.jpg 能被当静态文件取走。"""
    r = client.get(path)
    assert r.status_code == 404
    # 不能把源码当文本吐出来
    assert "def " not in r.text and "import " not in r.text


@pytest.mark.parametrize("method", ["post", "get"])
def test_shutdown_endpoint_is_gone(client, method):
    """v1.0.0 的安全修复：移除无鉴权的 /api/shutdown。"""
    r = getattr(client, method)("/api/shutdown")
    assert r.status_code in (404, 405)


def test_cors_allows_configured_origin(client, db):
    db.push()
    r = client.get("/api/action_lib/list", headers={"Origin": "https://example.com"})
    assert r.headers.get("access-control-allow-origin") in ("*", "https://example.com")


def test_protocols_endpoint_returns_data(client):
    r = client.get("/api/rehab/protocols")
    assert r.json()["code"] == 200
    assert r.json()["data"] is not None


def test_alps_returns_404_before_any_frame(client):
    import zitaishibie as zt

    with zt.alps_seq_lock:
        zt.latest_alps_feature = None
    r = client.get("/api/pose/alps")
    assert r.json()["code"] == 404


def _frame(t: int):
    """造一帧「肘部逐渐弯曲」的 33 点，避免所有关节点重合导致特征退化。"""
    lm = [[0.5, 0.5, 0.0] for _ in range(33)]
    lm[11] = [0.5, 0.30, 0.0]                 # 左肩
    lm[13] = [0.5 + 0.02 * t, 0.45, 0.0]      # 左肘
    lm[15] = [0.5 + 0.05 * t, 0.60, 0.0]      # 左腕
    return lm


SEQUENCE = [_frame(i) for i in range(6)]


def test_sequence_score_requires_reference(client, db):
    r = client.post("/api/rehab/sequence_score", json={})
    assert r.json()["code"] == 400
    assert "参考序列" in r.json()["message"]


def test_sequence_score_accepts_reference(client, db):
    r = client.post("/api/rehab/sequence_score", json={"reference": SEQUENCE, "patient": SEQUENCE})
    body = r.json()
    assert body["code"] == 200, body
    assert 0.0 <= body["data"]["top_similarity"] <= 1.0
