"""Flag frozen-livestream duplicate frames in people_zones.csv.

When the webcam stream freezes, the archiver keeps writing the same frame
under successive 15-min timestamps.  Both models then see pixel-identical
inputs, so consecutive rows carry identical segmentation pixel counts AND
identical people counts -- a combination that never happens by chance.

A frame is flagged ``frozen_dup = 1`` when it belongs to a run of 2+
consecutive frames with identical values in every key column.  The whole
run is dropped -- including the first frame -- because the freeze may have
begun before it, making its timestamp unreliable.

Usage:  python analysis/flag_frozen_frames.py   (run from repo root)
"""
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parent.parent / 'data'
KEY = ['px_beach', 'px_water',
       'people_beach', 'people_water', 'people_other', 'people_total']

ms = pd.read_csv(DATA / 'mask_stats_clean.csv')
pz = pd.read_csv(DATA / 'people_zones.csv')

d = (pz.merge(ms[['image', 'utc_ts', 'px_beach', 'px_water']], on='image',
              suffixes=('', '_ms'))
       .sort_values('utc_ts'))
runs = (d[KEY].ne(d[KEY].shift()).any(axis=1)).cumsum()
rl = d.groupby(runs).size()
bad_runs = set(rl[rl > 1].index)
d['frozen_dup'] = d.assign(r=runs).r.isin(bad_runs).astype(int)

pz['frozen_dup'] = pz['image'].isin(
    set(d.loc[d.frozen_dup == 1, 'image'])).astype(int)
pz.to_csv(DATA / 'people_zones.csv', index=False)

print(f'flagged {int(d.frozen_dup.sum())} frozen frames '
      f'in {len(bad_runs)} freeze events (longest {int(rl.max())} frames)')
print(d.loc[d.frozen_dup == 1, 'image'].iloc[:8].to_string(index=False))
