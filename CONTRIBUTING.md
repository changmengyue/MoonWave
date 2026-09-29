# 贡献指南

感谢你愿意为 MoonWave 月波康复做贡献！本文说明如何提 Issue、提 PR，以及本项目的开发约定。

## 目录

- [行为准则](#行为准则)
- [报告问题（Issue）](#报告问题issue)
- [提交代码（Pull Request）](#提交代码pull-request)
- [开发环境](#开发环境)
- [代码规范](#代码规范)
- [分支与提交约定](#分支与提交约定)

## 行为准则

参与本项目即表示你同意遵守 [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)。

## 报告问题（Issue）

- 提 Issue 前请先搜索是否已有相同问题。
- Bug 请使用 [Bug 报告模板](.github/ISSUE_TEMPLATE/bug_report.md)，尽量包含：
  - 复现步骤、期望结果、实际结果；
  - 运行环境（操作系统、Python 版本、浏览器、`mediapipe` 版本）；
  - 相关日志或截图（**请先抹掉任何密钥、真实用户数据**）。
- 功能建议请使用 [功能请求模板](.github/ISSUE_TEMPLATE/feature_request.md)。

## 提交代码（Pull Request）

1. Fork 本仓库，从 `main` 切出功能分支：`feat/xxx`、`fix/xxx`。
2. 完成改动，本地自测（见下）。
3. 提交前确保没有引入密钥、真实数据或大体积无关文件。
4. 发起 PR 到 `main`，使用 [PR 模板](.github/PULL_REQUEST_TEMPLATE.md) 描述改动与验证方式。
5. 至少 1 位维护者 review 通过后合并；`main` 受保护，请不要直接推送。

## 开发环境

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

export DB_PASS='你的数据库密码'
export MEDIAPIPE_DISABLE_GPU=1    # 无 GPU / headless
export NO_BROWSER=1               # 不自动开浏览器

python zitaishibie.py
```

算法自测（不需要数据库与摄像头）：

```bash
python test_alps_lite.py
```

## 代码规范

- **保持改动最小**：只改与本次目标相关的代码，不要顺手重构无关部分。
- **不要提交敏感信息**：数据库密码、Token、密钥、真实用户数据、私有地址一律不入库（见 `.gitignore`）。
- **前端**：动态文本渲染优先用 `textContent` 或转义，避免 XSS；不要引入未声明来源的第三方库/字体/图片。
- **后端**：接口返回统一为 `{code, message, data}`；涉及阻塞操作注意不要卡住事件循环。
- **第三方资产**：新增依赖/素材时，请在 PR 中说明其来源与许可证。

## 分支与提交约定

- 分支：`main` 为稳定分支；功能用 `feat/*`，修复用 `fix/*`，文档用 `docs/*`。
- 提交信息建议采用约定式提交（Conventional Commits）：

```
feat: 新增 xxx 功能
fix: 修复 xxx 问题
docs: 更新 xxx 文档
chore: 杂项(构建/依赖等)
```

谢谢你的贡献！
