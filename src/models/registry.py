from .dbcr import DBCRNet


def get_model(name):
    name = name.lower()
    if name == "dbcr":
        return DBCRNet
    if name in {"diffcr", "uncrtaints"}:
        raise NotImplementedError(
            f"Model '{name}' is not implemented yet. Add it in src/models/registry.py."
        )
    raise ValueError(f"Unknown model: {name}")
