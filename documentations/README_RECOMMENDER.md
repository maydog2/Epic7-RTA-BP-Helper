# Epic Seven BP 推荐服务

推荐模型从 `E7-RTA-Helper` 迁到这个仓库。

## 文件

- `data/transformer/rec_model_position_warfare.weights.h5`：训练好的 Transformer 选人模型权重。
- `data/transformer/rec_variables.pkl`：Transformer 的编码器和元数据。
- `data/epic7_match_history_raw.jsonl`：爬下来的比赛原始记录，是训练、reranker 统计、预禁和先手推荐的数据来源。
- `data/hero_details.csv`：英雄代码、`name`、`name_zh`、`Role`、`Element`、`appearance_count`，供英雄选择器和 Transformer 元数据使用。
- `backend/recommender_service.py`：单独运行的 Flask 推理服务。
- `workflow_scripts/get_rec_transformer_model.py`：Transformer 训练脚本。
- `workflow_scripts/build_transformer_draft_dataset.py`：用比赛记录生成 Transformer 的 NPZ 数据集。
- `workflow_scripts/evaluate_draft_models.py`：离线评估 Transformer。
- `workflow_scripts/tf_requirements.txt`：固定版本的 TensorFlow 训练依赖（`tensorflow==2.17.0`、`tf_keras==2.17.0`）。线上服务用的是 `requirements-recommender.txt`。
- `workflow_scripts/generate_response_stats.py`：根据比赛 JSONL 写出 `data/reranker/` 下的两张应对统计 CSV。
- `workflow_scripts/generate_synergy_stats.py`：根据比赛 JSONL 写出 `data/reranker/hero_synergy_stats.csv`。
- `workflow_scripts/build_runtime_artifacts.py`：重建预禁、先手、reranker、终禁和 `appearance_count`。有新比赛后运行这一条；它会调用上面两个生成脚本。

## 运行

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-recommender.txt
# 本地爬虫和打包再装这一份：
.\.venv\Scripts\python.exe -m pip install -r requirements-local.txt
$env:RECOMMENDER_RERANKER='true'
.\.venv\Scripts\python.exe -m backend.recommender_service
```

服务会把选中的本地端口写到 `recommender_service_port.txt`。

## 刷新统计并重新训练

爬新比赛的步骤在 [README_SCRAPERS.md](./README_SCRAPERS.md)。`data/epic7_match_history_raw.jsonl` 更新后，先刷新推理用的统计，再重建数据集并训练 Transformer。构建数据集会同时重写 `data/transformer/rec_variables.pkl`。CPU 训练大约 30 分钟，产物是 `rec_model_position_warfare.weights.h5` 和 `training_report_position_warfare.json`。

```powershell
.\.venv\Scripts\python.exe workflow_scripts\build_runtime_artifacts.py
.\.venv\Scripts\python.exe workflow_scripts\build_transformer_draft_dataset.py
.\.venv\Scripts\python.exe workflow_scripts\get_rec_transformer_model.py
.\.venv\Scripts\python.exe workflow_scripts\evaluate_draft_models.py
```

`evaluate_draft_models.py` 可以不跑。它用验证集给训练好的权重打分，并把评估报告写到 `data/transformer/`。

如果要在单独环境里安装固定版本的训练依赖：

```powershell
.\.venv\Scripts\python.exe -m pip install -r workflow_scripts\tf_requirements.txt
```

## API

选人流程（预禁、先手、中途选人、终禁）见 [DRAFT_RECOMMENDATION_LOGIC.md](./DRAFT_RECOMMENDATION_LOGIC.md)。

先显式初始化：

```text
GET /init_recommender
```

获取推荐：

```text
GET /recommend?user_picks=c1117,c1134&enemy_picks=c2066&first_pick_team=My%20Team&warfare_rules=ANY
```

返回形状：

```json
{
  "top_10_heroes": ["c1117"],
  "top_10_rates": [3.2],
  "phase": "pick"
}
```

- `top_10_heroes`：英雄头给出的、当前还没被选走的下一手最高概率英雄。
- `top_10_rates`（可选）：顺序与 `top_10_heroes` 相同。每一项是该英雄 softmax 概率乘以 100，也就是模型估计的“下一手选到这个英雄”的百分比。概率是在全部英雄上计算的，所以这十个数字加起来通常不是 100%。
- `phase`：选人阶段是 `"pick"`；双方都锁了五人之后是 `"ban"`。终禁排序是英雄头的另一种用法。

空阵容的返回可能没有 `top_10_rates` 和 `phase`，英雄列表会用来自出场率的占位结果。
