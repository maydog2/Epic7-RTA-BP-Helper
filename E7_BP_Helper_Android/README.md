# E7 BP Helper — Android（阶段 3A）

用 WebView 离线加载现有的 React 界面。

## 当前接口

阶段 1 离线壳：

- `/api/status`
- `/api/heroes`
- `/api/heroes/{code}/avatar`
- `/api/icons/elements/{name}`
- `/api/icons/roles/{name}`

阶段 3B-1 本地选人推荐：

- `/api/recommend`：由 `TfliteRecommendEngine` 根据查询参数返回
- `/api/preban_recommend`：仍返回 HTTP 501

推荐接口：

- `/api/recommend`：本地 TFLite 选人推荐（查询参数）
- `/api/preban_recommend`：仍返回 HTTP 501

## 准备环境

- Android Studio（建议 Ladybug 或更新版本）
- JDK 17
- Node.js（用来构建 `frontend/dist`）
- Python 虚拟环境（用来导出阶段 1 和阶段 3 的资源）

## 从主仓库同步资源

在项目根目录 `E7_BP_Helper/` 运行：

```bash
cd frontend
npm install
npm run build
cd ..
python workflow_scripts/build_runtime_artifacts.py
python workflow_scripts/export_android_phase1_assets.py
python workflow_scripts/export_android_recommendation_artifacts.py
python workflow_scripts/export_android_tflite.py
python workflow_scripts/android_tflite_parity.py
python workflow_scripts/android_recommendation_parity.py
python workflow_scripts/export_android_phase3_assets.py
```

阶段 1 会把资源写到 `E7_BP_Helper_Android/app/src/main/assets/`：

```
assets/
  public/                           # frontend/dist
  data/hero_details.json
  dataset/{code}/c.png
  CharacterUI/elements/*.png
  CharacterUI/roles/*.png
```

`export_android_recommendation_artifacts.py` 把预禁、先手、reranker 和终禁数据导出成 `assets/data/` 里的 JSON：`preban_stats.json`、`first_pick_records.json`、`reranker_synergy_stats.json`、`reranker_response_stats.json`、`reranker_opening_response_stats.json`、`final_ban_stats.json`、`final_ban_config.json`。先运行 `build_runtime_artifacts.py`，这些 pickle 和 CSV 才存在。`--output` 可以改资源输出目录。

`android_recommendation_parity.py` 把这些资源和 Python 推荐结果对比，并写入 `data/transformer/android_recommendation_parity_report.json`。默认会重新导出 JSON；加上 `--skip-export` 则只检查已有文件。

阶段 3A 还会加入：

```
assets/data/
  rec_model_position_warfare.tflite
  transformer_encoder_metadata.json
  android_tflite_debug_case.json
  android_tflite_test_cases.json
```

英雄数据、立绘、图标、前端构建或 TFLite 产物有变化时，重新运行这些导出脚本。

只导出英雄和图片、不复制前端时：

```bash
python workflow_scripts/export_android_phase1_assets.py --skip-frontend
```

## 构建和安装

1. 用 Android Studio 打开 `E7_BP_Helper_Android/`。
2. 等 Gradle 同步（会下载 wrapper 和依赖）。
3. 连接设备或启动模拟器（API 24 及以上）。
4. 运行 `app` 配置。
5. 在 Logcat 里看 `TfliteEngineDebug`，确认阶段 3A 的本地推理和对照结果一致。

也可以在 Android 工程目录里执行：

```bash
./gradlew assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

Windows PowerShell 里用 `gradlew.bat`，不用 `./gradlew`。

## 怎么工作

- `MainActivity` 通过 `WebViewAssetLoader` 加载 `https://appassets.androidplatform.net/public/index.html`。
- `LocalApiHandler` 拦截 `/api/*`，从打包进去的资源返回 JSON 或 PNG。
- `/api/recommend` 解析和 React 前端相同的查询参数，返回 `{ phase, top_10_heroes, top_10_rates }`。
- `TfliteRecommendEngine` 加载导出的 TFLite 模型和编码器元数据，在本地做选人推理。
- `TfliteEngineDebugRunner` 启动时跑一条固定对照用例，并把前十名写到日志。
- 离线壳不需要网络。

## 还没做的阶段

- **阶段 3B-2**：`/api/preban_recommend`，以及先手、终禁路径。
- **之后**：移植 reranker。
