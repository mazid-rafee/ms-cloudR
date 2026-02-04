import json


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def apply_config(args, config):
    for key, value in config.items():
        if hasattr(args, key):
            setattr(args, key, value)
    return args
