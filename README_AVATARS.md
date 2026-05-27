# Hero Portrait Assets

Portrait and icon assets for the React frontend are served by `recommender_service.py`.

## Downloading Portraits

Run:

```powershell
.\.venv\Scripts\python.exe workflow_scripts\get_character_ids.py
```

This downloads hero portraits from Smilegate and writes:

- `dataset/<hero_code>/c.png`
- `data/hero_details.csv` (`Hero`, `name`, and other metadata columns)

If `dataset` is empty, hero avatars in the web UI will be missing until this script has been run.

## Displaying Portraits (Web UI)

The React app loads heroes from `GET /api/heroes`. Each hero may include:

- `avatar_url` → `GET /api/heroes/<hero_code>/avatar` (served from `dataset/<hero_code>/c.png`)
- `element_icon_url` → `GET /api/icons/elements/<element>.png` (from `CharacterUI/elements/`)
- `role_icon_url` → `GET /api/icons/roles/<role>.png` (from `CharacterUI/roles/`)

Role and element icons come from CeciliaBot scraping (`workflow_scripts/get_hero_description.py` populates `data/hero_details.csv`). Only `CharacterUI/elements/` and `CharacterUI/roles/` are used by the web app.
