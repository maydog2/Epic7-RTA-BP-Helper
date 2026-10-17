# Epic Seven 数据爬虫

爬虫和英雄表工具在 `workflow_scripts/`。它们写入 `data/epic7_match_history_raw.jsonl`、`data/hero_details.csv` 和 `dataset/<hero_code>/c.png`。用到 Selenium 的脚本需要先安装 Chrome。

## 安装

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-recommender.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-local.txt
```

`requirements-local.txt` 在 `requirements-recommender.txt` 之外补上爬虫和 Windows 打包用的包：`beautifulsoup4`、`selenium`、`webdriver-manager`、`zhconv`、`pillow`、`pyinstaller`。

## 爬比赛

`get_matches.py` 读取 Stove 排行页，把 RTA 比赛追加到 `data/epic7_match_history_raw.jsonl`。预禁来自 `li.preban-hero`，选人来自 `li.pick-hero`。进度写在 `data/epic7_scraper_state.json` 的 `visited_players` 和 `in_progress_players` 里。这个进度文件留在正在爬数据的那台机器上。

```powershell
$env:E7_GET_MATCHES_RANK_SERVER='Korea'
$env:E7_GET_MATCHES_MAX_DAYS_ENABLED='1'
$env:E7_GET_MATCHES_MAX_DAYS='3'
$env:E7_GET_MATCHES_WORKERS='4'
$env:E7_GET_MATCHES_MAX_PLAYERS='70'
.\.venv\Scripts\python.exe workflow_scripts\get_matches.py
```

其他环境变量：`E7_SCRAPER_SMOKE=1` 做一次小规模试跑；`E7_GET_MATCHES_MAX_PER_PLAYER=N` 默认 100；`E7_GET_MATCHES_MAX_PLAYERS=N` 默认不限制人数；`E7_GET_MATCHES_RANK_SERVER` 默认 `All Servers`；`E7_GET_MATCHES_WORKERS` 默认 4。

## 英雄表

`hero_details_fields.py` 定义 `data/hero_details.csv` 的列：`Hero`、`name`、`name_zh`、`Role`、`Element`、`appearance_count`。`hero_details_io.py` 负责读写这份 CSV。其他脚本会引用它们，不要单独运行。

```powershell
.\.venv\Scripts\python.exe workflow_scripts\get_character_ids.py
.\.venv\Scripts\python.exe workflow_scripts\get_hero_description.py
.\.venv\Scripts\python.exe workflow_scripts\update_hero_names.py c2185
.\.venv\Scripts\python.exe workflow_scripts\update_hero_names.py c2185 c2186
.\.venv\Scripts\python.exe workflow_scripts\update_hero_names.py --heroes c2185,c2186
.\.venv\Scripts\python.exe workflow_scripts\update_hero_names.py --all
.\.venv\Scripts\python.exe workflow_scripts\update_hero_names.py c2185 --download-portrait
```

- `get_character_ids.py` 从 Smilegate 抓英雄代码、英文名和立绘，写入 `data/hero_details.csv` 和 `dataset/<hero_code>/c.png`。
- `get_hero_description.py` 从 CeciliaBot 抓职业和属性，写入同一份 CSV。`E7_HERO_DETAILS_FRESH=1` 不使用内存里已有的 CSV；`E7_HERO_DETAILS_LIMIT=N` 抓满 N 个新英雄后停止；`E7_LAST_HERO_DETAIL_NAME` 是结束标记名，默认 `Requiemroar`。
- `update_hero_names.py` 从 Stove 英雄录刷新英文名和简体中文名。可以传入英雄代码、`--heroes c2185,c2186`，或 `--all`。`--all` 配 `--zh-only` 只刷新 `name_zh`。`--download-portrait` 同时写入 `dataset/<hero_code>/c.png`。`--limit N` 限制 `--all` 的数量。

新英雄要先在 `data/hero_details.csv` 里有一行，下面的统计重建才会计入这个英雄。

## 用比赛记录生成统计

爬完数据后通常运行 `build_runtime_artifacts.py`。它重建预禁统计、先手选择记录、三张 reranker CSV、终禁统计和 `appearance_count`。它不登记新英雄，也不打开浏览器。

```powershell
.\.venv\Scripts\python.exe workflow_scripts\build_runtime_artifacts.py
.\.venv\Scripts\python.exe workflow_scripts\build_runtime_artifacts.py --skip-appearance
.\.venv\Scripts\python.exe workflow_scripts\build_runtime_artifacts.py --skip-reranker
```

下面两个脚本由 reranker 那一步调用。只想重写这两份 CSV 时再单独运行：

```powershell
.\.venv\Scripts\python.exe workflow_scripts\generate_response_stats.py
.\.venv\Scripts\python.exe workflow_scripts\generate_synergy_stats.py
```

- `generate_response_stats.py` 写入 `data/reranker/hero_counter_or_response_stats.csv` 和 `data/reranker/enemy_opening_response_stats.csv`。可选参数：`--input`、`--response-output`、`--opening-output`、`--print-top`。
- `generate_synergy_stats.py` 写入 `data/reranker/hero_synergy_stats.csv`。可选参数：`--input`、`--output`。
- `update_hero_details_appearance_stats.py` 根据 JSONL 写入 `appearance_count`。重建运行时数据时会调用它，除非加上 `--skip-appearance`。

用这份 JSONL 训练 Transformer 的步骤写在 [README_RECOMMENDER.md](./README_RECOMMENDER.md)。
