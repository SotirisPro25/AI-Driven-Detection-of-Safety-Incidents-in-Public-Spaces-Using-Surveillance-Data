#!/usr/bin/env python3
"""
build_split_list.py -- the published split list (data/splits/ieee_source_disjoint_split.csv) from the split
that kaggle_prepare.py wrote on Kaggle and the listing of the dataset as distributed by IEEE DataPort.

    python ieee_audit_tools.py listing --root <IEEE DataPort copy> --out ieee_listing_original.csv
    python build_split_list.py ieee_split_disjoint.csv ieee_listing_original.csv data/splits

Why a mapping is needed. In part of the Kaggle copy of the dataset the file names lost every character outside
ASCII and every '!', with the leading and trailing spaces then removed: "Трафик в стамбуле-splitted-130-135.mp4"
became "-splitted-130-135.mp4". The split was computed on those names (Section 6.2.3). This script gives every
row its original name. Where two original files collapsed to one Kaggle name, only one of them was in the copy,
and the row lists both candidates separated by " | ".

Outputs
    ieee_source_disjoint_split.csv     disjoint_partition, class, ieee_name, kaggle_name, distributed_partition,
                                       used_in_runs (0 for the ten files of the five names that occur in the same
                                       class in two supplied partitions, left out of every run)
    ieee_absent_from_kaggle_copy.csv   the files of the original release that the Kaggle copy does not hold
"""
import collections, csv, os, sys

KAGGLE_CSV, LISTING_CSV, OUT_DIR = sys.argv[1:4]
EXCLUDED = {'-splitted-130-135.mp4', '-splitted-250-255.mp4', '-splitted-270-275.mp4',
            '-splitted-290-295.mp4', '-splitted-390-395.mp4'}       # the EXCLUDED line of the training log


def kaggle_form(name):
    return ''.join(ch for ch in name if ord(ch) < 128 and ch != '!').strip()


R = list(csv.DictReader(open(KAGGLE_CSV, encoding='utf8')))
orig = [(r['partition'], r['cls'], r['name']) for r in csv.DictReader(open(LISTING_CSV, encoding='utf8'))]
exact = set(orig)
by_form = collections.defaultdict(list)
for k in orig:
    by_form[(k[0], k[1], kaggle_form(k[2]))].append(k[2])

rows, claimed = [], collections.Counter()
for r in R:
    key = (r['distributed_partition'], r['cls'], r['name'])
    cands = [r['name']] if key in exact else sorted(by_form[key])
    if not cands:
        sys.exit(f'no original file for {key}')
    for c in cands:
        claimed[(key[0], key[1], c)] += 1
    used = 0 if (r['cls'] == 'No Accident Normal Traffic' and r['name'] in EXCLUDED
                 and r['distributed_partition'] in ('val', 'test')) else 1
    rows.append([r['disjoint_partition'], r['cls'], ' | '.join(cands), r['name'], r['distributed_partition'], used])

os.makedirs(OUT_DIR, exist_ok=True)
with open(os.path.join(OUT_DIR, 'ieee_source_disjoint_split.csv'), 'w', newline='', encoding='utf8') as f:
    w = csv.writer(f)
    w.writerow(['disjoint_partition', 'class', 'ieee_name', 'kaggle_name', 'distributed_partition', 'used_in_runs'])
    w.writerows(rows)
with open(os.path.join(OUT_DIR, 'ieee_absent_from_kaggle_copy.csv'), 'w', newline='', encoding='utf8') as f:
    w = csv.writer(f)
    w.writerow(['distributed_partition', 'class', 'ieee_name', 'reason'])
    for p, c, n in sorted(k for k in orig if claimed[k] == 0):
        w.writerow([p, c, n, 'not in the Kaggle copy'])
    for r in rows:
        if ' | ' in r[2]:
            w.writerow([r[4], r[1], r[2], f'one of these two files is in the Kaggle copy under {r[3]}, the other is not'])

print('rows', len(rows), dict(collections.Counter(r[0] for r in rows)),
      '| used in runs', dict(collections.Counter(r[0] for r in rows if r[5])),
      '| supplied partition in runs', dict(collections.Counter(r[4] for r in rows if r[5])))
