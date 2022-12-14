import os

from pathlib import Path
from typing import Set

def search_for_files(root: Path, keys: Set[str]):
    paths = {}
    for root, dirs, files in os.walk(root):
        for file in files:
            name = os.path.splitext(file)[0]
            if name in keys:
                paths[name] = os.path.join(root, file)
    return paths
