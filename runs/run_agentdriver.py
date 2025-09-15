from pathlib import Path
import time
import json
import yaml

path = Path("configs/default.yaml")
with path.open("r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}
print(cfg)