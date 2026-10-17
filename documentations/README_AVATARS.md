# 英雄立绘资源

React 前端用的立绘和图标由 `backend/recommender_service.py` 提供。

## 下载立绘

运行：

```powershell
.\.venv\Scripts\python.exe workflow_scripts\get_character_ids.py
```

脚本从 Smilegate 下载英雄立绘，并写入：

- `dataset/<hero_code>/c.png`
- `data/hero_details.csv`（`Hero`、`name` 以及其他元数据列）

`dataset` 是空的时候，网页上的英雄头像会缺失，需要先跑这个脚本。

## 在网页上显示立绘

React 通过 `GET /api/heroes` 读取英雄。每个英雄可能带有：

- `avatar_url`：对应 `GET /api/heroes/<hero_code>/avatar`，文件来自 `dataset/<hero_code>/c.png`
- `element_icon_url`：对应 `GET /api/icons/elements/<element>.png`，文件来自 `CharacterUI/elements/`
- `role_icon_url`：对应 `GET /api/icons/roles/<role>.png`，文件来自 `CharacterUI/roles/`

职业和属性图标来自 CeciliaBot 爬取。`workflow_scripts/get_hero_description.py` 会把结果写入 `data/hero_details.csv`。网页只用 `CharacterUI/elements/` 和 `CharacterUI/roles/`。
