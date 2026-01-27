from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

__all__ = ['get_imglists']

def get_imglists(root, split='train', phase='train'):
    '''
    get all images path
    @param: 
        root : root path to dataset
        spilt: sub path to specific dataset folder
    '''

    root_path = Path(root)
    grain = root_path.name
    imgs, labels = [], []

    if split == 'train':
        split = phase

    project_root = Path(__file__).resolve().parents[2]
    datalist_dir = project_root / "runs" / "datalist"

    candidates = [
        datalist_dir / f"{grain}_{split}.txt",
        datalist_dir / "datalist" / f"{grain}_{split}.txt",
    ]
    split_file = next((p for p in candidates if p.exists()), None)
    if split_file is None:
        raise FileNotFoundError(
            f"Could not find split file for grain={grain!r} split={split!r}. "
            f"Tried: {', '.join(str(p) for p in candidates)}"
        )

    with split_file.open("r", encoding="utf-8") as f:
        for line in f.readlines():
            line = line.replace('\n','')
            im_path, label = line.split()
            im_path = str(root_path / im_path)
            imgs.append(im_path)
            labels.append(int(label))
    length = len(imgs)
    print(f'* {split} : {length}')
    files = pd.DataFrame({'filename': imgs, 'label': labels})
    return files
