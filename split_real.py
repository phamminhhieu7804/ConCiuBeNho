# split_real.py
import os, random, shutil

src = 'data/real/labeled'
splits = {
    'train': 0.8,
    'val':   0.1,
    'test':  0.1
}

files = os.listdir(src)
random.shuffle(files)
n = len(files)
cumsum = 0

for split, frac in splits.items():
    out_dir = f"data/real/{split}"
    os.makedirs(out_dir, exist_ok=True)
    count = int(frac * n)
    for fname in files[cumsum:cumsum+count]:
        shutil.copy(os.path.join(src, fname), os.path.join(out_dir, fname))
    cumsum += count

# Phần dư (n - sum) cho train
for fname in files[cumsum:]:
    shutil.copy(os.path.join(src, fname), os.path.join('data/real/train', fname))
