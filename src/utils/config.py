import json


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def apply_config(args, config, defaults=None):
    for key, value in config.items():
        if not hasattr(args, key):
            continue
        if defaults is not None and key in defaults:
            if getattr(args, key) != defaults[key]:
                continue
        setattr(args, key, value)
    return args
