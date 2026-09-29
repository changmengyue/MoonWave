# 部署说明（生产环境）

本文说明如何把 MoonWave 部署到一台 Linux 服务器（以 Ubuntu + nginx 为例）。文中**不包含任何真实 IP、域名或凭据**，请按你的环境替换。

## 架构

```
浏览器 ──HTTPS(443)──▶ nginx ──反代──▶ FastAPI (127.0.0.1:8000) ──▶ MySQL/MariaDB
                         │
                         └── /mediapipe/ 直接由 nginx 提供静态资产
```

- 后端监听 `127.0.0.1:8000`，**不直接对公网暴露**。
- nginx 负责 TLS、反向代理，以及 `/mediapipe/` 静态资产（MIME 要求见 [MEDIAPIPE_ASSETS.md](MEDIAPIPE_ASSETS.md)）。

## 1. 目录与依赖

```bash
# 放置代码
sudo mkdir -p /opt/moonwave
sudo cp -r <项目文件> /opt/moonwave/
cd /opt/moonwave

# 虚拟环境与依赖
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
```

> `mediapipe` 需 **0.10.x**。

## 2. 环境变量

后端全部通过环境变量配置（**不要把密码写进代码**）：

| 变量 | 说明 |
|---|---|
| `DB_HOST` / `DB_USER` / `DB_PASS` / `DB_NAME` / `DB_PORT` | 数据库连接（`DB_PASS` 必填） |
| `MEDIAPIPE_DISABLE_GPU` | 无 GPU / headless 环境设为 `1` |
| `NO_BROWSER` | 设为 `1`，避免启动时尝试打开浏览器 |

## 3. systemd 服务

`/etc/systemd/system/moonwave.service`：

```ini
[Unit]
Description=MoonWave rehab backend
After=network.target mysql.service mariadb.service

[Service]
Type=simple
WorkingDirectory=/opt/moonwave
Environment=DB_HOST=127.0.0.1
Environment=DB_USER=rehab_user
Environment=DB_PASS=在此填写数据库密码
Environment=DB_NAME=rehab_db
Environment=MEDIAPIPE_DISABLE_GPU=1
Environment=NO_BROWSER=1
ExecStart=/opt/moonwave/venv/bin/uvicorn zitaishibie:app --host 127.0.0.1 --port 8000
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now moonwave
```

> 也可以直接 `python zitaishibie.py` 运行（会监听 `127.0.0.1:8000`），但生产建议用上面的 systemd + uvicorn。

> ⚠️ systemd 单元里含有数据库密码，请设置文件权限（如 `chmod 600`）并确保它不会被提交到任何仓库。

## 4. nginx 反向代理

`/etc/nginx/sites-available/moonwave`：

```nginx
server {
    listen 80;
    server_name your-domain.example;
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.example;

    ssl_certificate     /etc/letsencrypt/live/your-domain.example/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your-domain.example/privkey.pem;

    # 前端姿态资产：由 nginx 直接提供，确保 MIME 正确
    location /mediapipe/ {
        alias /opt/moonwave/mediapipe/;
        types {
            application/javascript mjs;
            application/wasm       wasm;
        }
        default_type application/octet-stream;
    }

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        client_max_body_size 20m;
    }
}
```

```bash
sudo ln -s /etc/nginx/sites-available/moonwave /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
```

证书可用 certbot 获取（Let's Encrypt）。

## 5. 部署与更新流程

```bash
# 更新：覆盖应用文件后重启服务
sudo cp <新文件> /opt/moonwave/
sudo systemctl restart moonwave
```

## 6. 上线后自检

```bash
# 绕过 DNS，直接用 Host 头自测
curl -sk -H 'Host: your-domain.example' https://127.0.0.1/ -o /dev/null -w '%{http_code}\n'

# 确认源码/配置不会被直接下载（应返回 404）
curl -sk -o /dev/null -w '%{http_code}\n' https://your-domain.example/zitaishibie.py
```

## 7. 安全与合规清单

部署前请逐项确认（详见 [SECURITY.md](../SECURITY.md)）：

- [ ] 数据库密码通过环境变量提供，未硬编码、未入库；
- [ ] 后端只监听 `127.0.0.1`，未直接暴露到公网；
- [ ] 静态资源为白名单方式，源码/备份/配置不可被下载；
- [ ] 已启用 HTTPS，证书私钥权限正确；
- [ ] 已按当地法规处理个人健康信息的告知同意、最小化与留存期限；
- [ ] 已关注依赖的安全更新。
