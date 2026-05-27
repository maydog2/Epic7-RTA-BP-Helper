# Epic Seven BP Recommender

This folder contains the migrated recommendation model from `E7-RTA-Helper`.

## Files

- `data/transformer/rec_model_position_warfare.weights.h5`: trained Transformer draft model weights.
- `data/transformer/rec_variables.pkl`: Transformer encoders and metadata.
- `data/epic7_match_history_raw.jsonl`: raw scraped match history (source of truth for training, reranker stats, and preban/first-pick recommenders).
- `data/hero_details.csv`: hero code, `name`, `name_zh`, `Role`, `Element`, and `appearance_count` for the Hero Picker and Transformer metadata.
- `recommender_service.py`: standalone Flask service for Transformer inference.
- `workflow_scripts/get_rec_transformer_model.py`: Transformer training script.
- `workflow_scripts/build_transformer_draft_dataset.py`: builds Transformer NPZ datasets from raw match history.
- `workflow_scripts/evaluate_draft_models.py`: offline Transformer evaluation.
- `workflow_scripts/tf_requirements.txt`: original TensorFlow training requirements.

## Run

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-recommender.txt
$env:RECOMMENDER_RERANKER='true'
.\.venv\Scripts\python.exe recommender_service.py
```

The service writes its selected local port to `recommender_service_port.txt`.

## API

Initialize explicitly:

```text
GET /init_recommender
```

Get recommendations:

```text
GET /recommend?user_picks=c1117,c1134&enemy_picks=c2066&first_pick_team=My%20Team&warfare_rules=ANY
```

Response shape:

```json
{
  "top_10_heroes": ["c1117"],
  "top_10_rates": [3.2],
  "phase": "pick"
}
```

- `top_10_heroes`: highest–probability next locks (among heroes not already in the draft), from the hero head.
- `top_10_rates` (optional): same order as `top_10_heroes`. Each entry is **100 × the model softmax** for that hero — i.e. estimated **percent chance the next pick in the match is that hero** (trained as next–hero classification). These values are over the **full** hero list, so the ten numbers usually **do not** add up to 100%.
- `phase`: `"pick"` during draft; `"ban"` when both sides have five locks (ban ranking is a separate use of the hero head).

Empty-draft responses may omit `top_10_rates` / `phase` and use placeholder hero lists from pick-rate stats.
