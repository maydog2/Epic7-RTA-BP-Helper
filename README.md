# Epic Seven RTA BP Helper

第七史诗 RTA 的 BP 辅助工具。网页、Windows 和 Android 共用同一套推荐：预禁、先手、中盘选人和终禁。

网页版：[epic7-rta-bp-helper.vercel.app](https://epic7-rta-bp-helper.vercel.app)

## 推荐怎么算

| 阶段 | 做法 |
| --- | --- |
| 预禁 | 按先手方和阵营，用历史预禁统计 |
| 先手（第 1 手） | 用历史先手记录，不经过 Transformer |
| 中盘（第 2–10 手） | Transformer 给出概率，再由 reranker 按协同和应对调整 |
| 终禁 | 统计和模型混合排序 |

算法细节见 [documentations/DRAFT_RECOMMENDATION_LOGIC.md](documentations/DRAFT_RECOMMENDATION_LOGIC.md)。

## 三种使用方式

- **网页**：前端放在 Vercel，推理 API 放在 Render。本地前端开发时请求这套 API。
- **Windows**：PyInstaller 打包的离线程序，模型和数据在安装目录里。从 1.0.0 起可以用补丁更新，1.0.3 起用文件哈希清单同步。
- **Android**：WebView 加载同一套界面，选人推荐在手机上用 TFLite 计算。

## 本地运行网页

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-recommender.txt
$env:RECOMMENDER_RERANKER='true'
.\.venv\Scripts\python.exe -m backend.recommender_service
```

另开一个终端：

```powershell
cd frontend
npm install
npm run dev
```

服务会把端口写到 `recommender_service_port.txt`。启动、接口和重新训练见 [documentations/README_RECOMMENDER.md](documentations/README_RECOMMENDER.md)。

## 更新数据和模型

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-local.txt
.\.venv\Scripts\python.exe workflow_scripts\get_matches.py
.\.venv\Scripts\python.exe workflow_scripts\build_runtime_artifacts.py
.\.venv\Scripts\python.exe workflow_scripts\build_transformer_draft_dataset.py
.\.venv\Scripts\python.exe workflow_scripts\get_rec_transformer_model.py
```

爬虫、英雄表和统计脚本的参数见 [documentations/README_SCRAPERS.md](documentations/README_SCRAPERS.md)。立绘来源见 [documentations/README_AVATARS.md](documentations/README_AVATARS.md)。

## Windows 打包和补丁

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-local.txt
.\build_exe.ps1 -Version 1.0.0
```

成品在 `dist\E7_BP_Helper\`，不进入本仓库。补丁的生成和安装见 [documentations/README_WINDOWS_PATCH.md](documentations/README_WINDOWS_PATCH.md)。

## Android

用 Android Studio 打开 `E7_BP_Helper_Android/`。导出模型、统计和前端资源的命令见 [E7_BP_Helper_Android/README.md](E7_BP_Helper_Android/README.md)。打出来的 APK 不进入本仓库。

## 仓库里没有的内容

虚拟环境、`frontend/node_modules`、`dist`、Windows 完整包和补丁、Android 构建缓存与 APK、爬虫进度文件都留在本机。线上推理需要的模型权重、英雄表、reranker 统计和比赛原始记录在仓库里。
