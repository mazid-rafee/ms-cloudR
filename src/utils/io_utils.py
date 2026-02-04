import json
import os
from datetime import datetime


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_json(path, data):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def utc_timestamp():
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def map_seasons(value):
    mapping = {
        "spring": "ROIs1158_spring",
        "summer": "ROIs1868_summer",
        "fall": "ROIs1970_fall",
        "winter": "ROIs2017_winter",
    }
    seasons = []
    for item in value.split(","):
        key = item.strip()
        if not key:
            continue
        seasons.append(mapping.get(key, key))
    return seasons
