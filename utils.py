import numpy as np


def decode(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        # 👇️ alternatively use str()
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()


def normalize(embedding):
    items = []
    for item in embedding:
        items.append(decode(item))

    return items
