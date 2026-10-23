"""pytest 公共装置。

两条约束决定了这里的写法：

1. **不依赖 MySQL**：`zitaishibie.get_db_connection()` 是本项目唯一的数据库入口，
   这里把它换成 `FakeConnection`，用「按顺序出结果」的方式模拟查询。
   好处是不用起数据库、不用清表，测试之间互不影响。
2. **不依赖 mediapipe / opencv**：这两个库加起来上百 MB，而接口测试根本不碰图像推理。
   在 import 被测模块之前把它们替换成替身，CI 因此只需要装十几 MB 的依赖。

被替换的只是「CV 引擎」和「数据库驱动」，FastAPI 路由、鉴权、校验、评分逻辑都是真的。
"""

import sys
from collections import deque
from unittest.mock import MagicMock

import pytest

# 必须在 import zitaishibie 之前完成替换
for _name in ("cv2", "mediapipe"):
    sys.modules.setdefault(_name, MagicMock())

import zitaishibie as zt  # noqa: E402


class FakeCursor:
    def __init__(self, conn):
        self.conn = conn
        self.rows = []

    def execute(self, sql, params=None):
        self.conn.sql_log.append((" ".join(str(sql).split()), params))
        if self.conn.fail_next is not None:
            exc, self.conn.fail_next = self.conn.fail_next, None
            raise exc
        self.rows = self.conn.pop_result()
        return 0

    def executemany(self, sql, seq_of_params):
        self.conn.sql_log.append((" ".join(str(sql).split()), seq_of_params))
        self.rows = []
        return 0

    def fetchone(self):
        return self.rows[0] if self.rows else None

    def fetchall(self):
        return list(self.rows)

    def close(self):
        pass


class FakeConnection:
    """按队列返回查询结果：每次 `execute` 消费一个结果集。"""

    def __init__(self):
        self._results = deque()
        self.sql_log = []
        self.fail_next = None
        self.closed = False
        self.autocommit = True

    def push(self, *rows):
        """压入下一次 `execute` 的结果集。"""
        self._results.append(list(rows))
        return self

    def pop_result(self):
        return self._results.popleft() if self._results else []

    def cursor(self):
        return FakeCursor(self)

    def commit(self):
        pass

    def rollback(self):
        pass

    def close(self):
        self.closed = True

    @property
    def executed(self):
        return [sql for sql, _ in self.sql_log]


@pytest.fixture
def db(monkeypatch):
    """把数据库连接换成 FakeConnection，并返回它供测试预置结果。"""
    conn = FakeConnection()
    monkeypatch.setattr(zt, "get_db_connection", lambda: conn)
    return conn


@pytest.fixture
def client(monkeypatch):
    """启动 TestClient；startup 里的建表与方案校验换成空操作。"""
    monkeypatch.setattr(zt, "init_db", lambda: None)
    monkeypatch.setattr(zt, "_validate_protocol_refs", lambda: None)
    from fastapi.testclient import TestClient

    with TestClient(zt.app) as c:
        yield c


@pytest.fixture
def auth_headers(db):
    """一条「有效 token」：token 查询返回 alice，之后的操作都认这个身份。"""
    db.push(("alice", None))
    return {"Authorization": "Bearer test-token"}
