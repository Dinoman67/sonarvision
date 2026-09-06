#!/usr/bin/env python3
"""
Build sonarvision_debris_honest_v1 — a single-class (unknown_debris) dataset
for an honest debris detector.

Sources (all h8-modal SSS, "similar features" by design):
  - h8.zip (core_model train+val): 633 debris + 2915 BG
  - v5 NOAA real debris (TGT-in-original_id): 511 debris
  - h8_unseen_test.zip: 48 debris (the honest test passes)

Split (PASS-DISJOINT, debris-only at test time):
  TEST  = all 48 h8_unseen_test debris + v5 real debris NOT in those 14 passes
  TRAIN = h8 train debris minus the 14 test passes + v5 train real debris minus those passes
  VAL   = h8 val debris minus the 14 test passes + v5 val real debris minus those passes
  BG    = h8 BG minus the 14 test passes + v5 NOAA BG minus those passes (for train/val)

The 14 test passes (E3_H11833, E4_TGT001..017) never appear in train/val.
This is the honest version of what the old core_model should have been tested on.
"""
import zipfile, io, csv, shutil, sys, yaml
from pathlib import Path
from collections import Counter

ROOT = Path('/home/ashish/sonar-vision')
DST = ROOT / 'datasets' / 'sonarvision_debris_honest_v1'

H8_ZIP = Path('/home/ashish/Downloads/h8.zip')
V5_ZIP = Path('/home/ashish/Downloads/sonarvision_multisource_v5.zip')
UNSEEN_ZIP = Path('/home/ashish/Downloads/h8_unseen_test.zip')


def pass_key(orig):
    parts = orig.split('_')
    return '_'.join(parts[:2]) if len(parts) >= 2 else orig


def unseen_pass_key(stem):
    # unseen manifest 'original_stem' like 'E3_H11833_TGT008_0003' -> 'E3_H11833'
    # or 'E4_TGT001_2_01' -> 'E4_TGT001'
    return pass_key(stem)


def load_metadata():
    with zipfile.ZipFile(H8_ZIP) as z:
        with z.open('h8/manifest.csv') as f:
            h8 = list(csv.DictReader(io.TextIOWrapper(f)))
    with zipfile.ZipFile(V5_ZIP) as z:
        with z.open('metadata.csv') as f:
            v5 = list(csv.DictReader(io.TextIOWrapper(f)))
    with zipfile.ZipFile(UNSEEN_ZIP) as z:
        with z.open('h8_unseen_test/test_manifest.csv') as f:
            unseen = list(csv.DictReader(io.TextIOWrapper(f)))
    # Build look-ups from the unseen manifest:
    #   original_stem -> pass_key
    #   UNSEEN_xxx.png image filename -> original_stem
    #   UNSEEN_xxx.png image filename -> pass_key
    unseen_stem_to_pk = {}
    unseen_stem_to_image = {}
    unseen_pk_of_image = {}
    for r in unseen:
        if r.get('has_debris') == 'True':
            stem = r['original_stem']
            unseen_stem_to_pk[stem] = unseen_pass_key(stem)
            img = r.get('image', '')
            if img:
                unseen_stem_to_image[img] = stem
                unseen_pk_of_image[img] = unseen_pass_key(stem)
    return h8, v5, unseen, unseen_stem_to_pk, unseen_stem_to_image, unseen_pk_of_image


def build_splits(h8, v5, unseen, unseen_stem_to_pk):
    h8_debris = [r for r in h8 if r['has_debris'] == '1']
    v5_real_debris = [r for r in v5
                       if r['source_dataset'] == 'NOAA'
                       and r['master_class'] in ('unknown', 'unknown_debris')
                       and 'TGT' in r['original_id']]
    unseen_debris = [r for r in unseen if r.get('has_debris') == 'True']
    # unseen manifest key is 'original_stem' (e.g. E3_H11833_TGT008_0003)
    test_passes = set(unseen_stem_to_pk[r['original_stem']] for r in unseen_debris)

    def h8_pk(row):
        return pass_key(row['stem'])
    def v5_pk(row):
        return pass_key(row['original_id'])

    forbidden = test_passes

    train_debris = [r for r in h8_debris
                    if r['split'] == 'train' and h8_pk(r) not in forbidden]
    train_debris += [r for r in v5_real_debris
                     if r['split'] == 'train' and v5_pk(r) not in forbidden]
    val_debris = [r for r in h8_debris
                  if r['split'] == 'val' and h8_pk(r) not in forbidden]
    val_debris += [r for r in v5_real_debris
                   if r['split'] == 'val' and v5_pk(r) not in forbidden]
    # Build the test set ONLY from unseen debris — never from v5's 'test' split,
    # because v5's labeled 'test' debris images are actually background (empty labels)
    # or belong to passes that are forbidden (shared with train/val). Those would either
    # be BG images mislabeled as test-debris, or leaks into the test set.
    test_debris = list(unseen_debris)
    # Ensure every test-debris row carries a non-empty pass_key by tagging it now.
    for r in test_debris:
        r['pass_key_override'] = unseen_stem_to_pk[r['original_stem']]
        r['original_id'] = r['original_stem']
    print(f'Test debris (unseen only): {len(test_debris)} images across '
          f'{len(set(r["pass_key_override"] for r in test_debris))} test passes')

    # The forbidden (test) passes are the 14 unseen debris passes. Any train/val row whose
    # pass is in this set must be dropped to keep the test fully pass-disjoint.
    forbidden = set(r['pass_key_override'] for r in test_debris)
    print(f'Forbidden (test) passes for filtering: {sorted(forbidden)}')


    train_bg = [r for r in h8
                if r['has_debris'] == '0' and r['split'] == 'train'
                and h8_pk(r) not in forbidden]
    train_bg += [r for r in v5
                 if r['source_dataset'] == 'NOAA' and 'TGT' not in r['original_id']
                 and r['split'] == 'train' and v5_pk(r) not in forbidden]
    val_bg = [r for r in h8
              if r['has_debris'] == '0' and r['split'] == 'val'
              and h8_pk(r) not in forbidden]
    val_bg += [r for r in v5
               if r['source_dataset'] == 'NOAA' and 'TGT' not in r['original_id']
               and r['split'] == 'val' and v5_pk(r) not in forbidden]
    return train_debris, val_debris, test_debris, train_bg, val_bg, test_passes


def extract_from_zips(z_h8, z_v5, z_unseen, rel, dst_path):
    for z in (z_h8, z_v5, z_unseen):
        try:
            with z.open(rel) as src:
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                with open(dst_path, 'wb') as dst:
                    shutil.copyfileobj(src, dst)
                return True
        except KeyError:
            continue
    return False


def image_rel(row, z_h8):
    src = row.get('source_dataset', '') or row.get('source', '')
    # h8 manifest uses 'source' column (f6/g7), NOT 'source_dataset'; treat as h8
    is_h8 = src in ('h8', 'f6', 'g7', '') and 'stem' in row
    if src == 'NOAA' or (row.get('source_dataset') == 'NOAA'):
        return ('v5', row['image_path'])
    if is_h8:
        stem = row['stem']
        for split in ('train', 'val', 'test'):
            rel = f'h8/images/{split}/{stem}.png'
            if rel in z_h8.namelist():
                return ('h8', rel)
        return None
    # unseen manifest: 'image' field (e.g. UNSEEN_0008.png)
    img = row.get('image', '')
    if img:
        rel = f'h8_unseen_test/images/test/{img}'
        return ('unseen', rel)
    return None


def image_exists_in_zips(row, z_h8, z_v5, z_unseen):
    rel = image_rel(row, z_h8)
    if not rel:
        return False
    key, rpath = rel
    z = {'h8': z_h8, 'v5': z_v5, 'unseen': z_unseen}[key]
    return rpath in z.namelist()


def label_rel(row, z_h8):
    src = row.get('source_dataset', '') or row.get('source', '')
    is_h8 = src in ('h8', 'f6', 'g7', '') and 'stem' in row
    if src == 'NOAA' or row.get('source_dataset') == 'NOAA':
        ipath = row['image_path']
        # ipath is like 'images/train/NOAA_000001.png'; label is 'labels/train/NOAA_000001.txt'
        parts = ipath.split('/')
        if len(parts) >= 2:
            return ('v5', 'labels/' + '/'.join(parts[1:]).replace('.png', '.txt'))
        return None
    if is_h8:
        stem = row['stem']
        for split in ('train', 'val', 'test'):
            rel = f'h8/labels/{split}/{stem}.txt'
            if rel in z_h8.namelist():
                return ('h8', rel)
        return None
    img = row.get('image', '')
    if img:
        return ('unseen', f'h8_unseen_test/labels/test/{Path(img).stem}.txt')
    return None


def image_filename(row):
    src = row.get('source_dataset', '') or row.get('source', '')
    if src == 'NOAA':
        return Path(row['image_path']).name
    # h8 manifest uses 'stem'
    if 'stem' in row:
        return f'{row["stem"]}.png'
    # unseen manifest uses 'image'
    img = row.get('image', '')
    if img:
        return img
    return f'{row.get("stem", row.get("image_id", "?"))}.png'


def write_split(z_h8, z_v5, z_unseen, split_name, debris_list, bg_list, DST, zips_map):
    print(f'Writing {split_name}: {len(debris_list)} debris + {len(bg_list)} BG')
    ok_debris = 0
    ok_bg = 0
    for row in debris_list:
        if not image_exists_in_zips(row, z_h8, z_v5, z_unseen):
            print(f'  SKIP debris image (not in zips): {image_filename(row)} src={row.get("source_dataset","?")} stem={row.get("stem","?")[:30]}')
            continue
        rel = image_rel(row, z_h8)
        if not rel:
            print(f'  SKIP debris image (no rel): {image_filename(row)}')
            continue
        key, rpath = rel
        z = zips_map[key]
        dst = DST / 'images' / split_name / Path(rpath).name
        if not dst.exists():
            if not extract_from_zips(z_h8, z_v5, z_unseen, rpath, dst):
                print(f'  FAIL image: {rpath} (stem={row.get("stem","?")})')
                continue
        ok_debris += 1
        lrel = label_rel(row, z_h8)
        if lrel:
            key2, lrpath = lrel
            z2 = zips_map[key2]
            dstl = DST / 'labels' / split_name / Path(lrpath).name
            if not dstl.exists():
                extract_from_zips(z_h8, z_v5, z_unseen, lrpath, dstl)
    for row in bg_list:
        if not image_exists_in_zips(row, z_h8, z_v5, z_unseen):
            continue
        rel = image_rel(row, z_h8)
        if not rel:
            continue
        key, rpath = rel
        z = zips_map[key]
        dst = DST / 'images' / split_name / Path(rpath).name
        if not dst.exists():
            if not extract_from_zips(z_h8, z_v5, z_unseen, rpath, dst):
                continue
        ok_bg += 1
        lrel = label_rel(row, z_h8)
        if lrel:
            key2, lrpath = lrel
            z2 = zips_map[key2]
            dstl = DST / 'labels' / split_name / Path(lrpath).name
            if not dstl.exists():
                extract_from_zips(z_h8, z_v5, z_unseen, lrpath, dstl)
        else:
            dstl = DST / 'labels' / split_name / Path(rpath).with_suffix('.txt').name
            if not dstl.exists():
                dstl.parent.mkdir(parents=True, exist_ok=True)
                dstl.write_text('')
    print(f'  written: {ok_debris} debris imgs, {ok_bg} BG imgs')
    return ok_debris, ok_bg


if __name__ == '__main__':
    if DST.exists():
        shutil.rmtree(DST)
    DST.mkdir(parents=True)

    h8, v5, unseen, unseen_stem_to_pk, unseen_stem_to_image, unseen_pk_of_image = load_metadata()
    print(f'unseen_stem_to_pk sample: {dict(list(unseen_stem_to_pk.items())[:5])}')
    print(f'unseen_pk_of_image keys: {list(unseen_pk_of_image.keys())[:5]}')
    train_debris, val_debris, test_debris, train_bg, val_bg, test_passes = build_splits(h8, v5, unseen, unseen_stem_to_pk)

    print(f'Train debris: {len(train_debris)}, train BG: {len(train_bg)}')
    print(f'Val debris:   {len(val_debris)}, val BG:   {len(val_bg)}')
    print(f'Test debris:  {len(test_debris)}')

    z_h8 = zipfile.ZipFile(H8_ZIP)
    z_v5 = zipfile.ZipFile(V5_ZIP)
    z_unseen = zipfile.ZipFile(UNSEEN_ZIP)
    zips_map = {'h8': z_h8, 'v5': z_v5, 'unseen': z_unseen}

    # Tag unseen-derived test rows with real pass_key BEFORE writing split and metadata.
    for row in train_debris + val_debris + test_debris + train_bg + val_bg:
        img = image_filename(row)
        if img in unseen_pk_of_image:
            row['pass_key_override'] = unseen_pk_of_image[img]
            row['original_id'] = unseen_stem_to_image.get(img, row.get('original_id', ''))
        elif 'stem' in row and row['stem'] in unseen_stem_to_pk:
            row['pass_key_override'] = unseen_stem_to_pk[row['stem']]

    write_split(z_h8, z_v5, z_unseen, 'train', train_debris, train_bg, DST, zips_map)
    write_split(z_h8, z_v5, z_unseen, 'val', val_debris, val_bg, DST, zips_map)
    write_split(z_h8, z_v5, z_unseen, 'test', test_debris, [], DST, zips_map)

    z_h8.close()
    z_v5.close()
    z_unseen.close()

    data_yaml = {
        'path': str(DST),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': 1,
        'names': {0: 'unknown_debris'},
    }
    with open(DST / 'dataset.yaml', 'w') as f:
        yaml.dump(data_yaml, f)

    meta_rows = []
    for split_name, debris_list, bg_list in [
        ('train', train_debris, train_bg),
        ('val', val_debris, val_bg),
        ('test', test_debris, []),
    ]:
        for row in debris_list:
            src = row.get('source_dataset', '') or row.get('source', '')
            meta_rows.append({
                'image_id': image_filename(row),
                'split': split_name,
                'is_debris': '1',
                'source': 'NOAA' if src == 'NOAA' or row.get('source_dataset') == 'NOAA' else 'h8',
                'original_id': row.get('original_id', row.get('stem', '')),
                'pass_key': row.get('pass_key_override', pass_key(row.get('original_id', row.get('stem', '')))),
            })
        for row in bg_list:
            src = row.get('source_dataset', '') or row.get('source', '')
            meta_rows.append({
                'image_id': image_filename(row),
                'split': split_name,
                'is_debris': '0',
                'source': 'NOAA' if src == 'NOAA' or row.get('source_dataset') == 'NOAA' else 'h8',
                'original_id': row.get('original_id', row.get('stem', '')),
                'pass_key': row.get('pass_key_override', pass_key(row.get('original_id', row.get('stem', '')))),
            })
    with open(DST / 'metadata.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['image_id', 'split', 'is_debris', 'source',
                                          'original_id', 'pass_key'])
        w.writeheader()
        w.writerows(meta_rows)

    train_imgs = list((DST / 'images/train').glob('*.png'))
    val_imgs = list((DST / 'images/val').glob('*.png'))
    test_imgs = list((DST / 'images/test').glob('*.png'))
    train_labels = list((DST / 'labels/train').glob('*.txt'))
    val_labels = list((DST / 'labels/val').glob('*.txt'))
    test_labels = list((DST / 'labels/test').glob('*.txt'))
    print(f'\nTrain: {len(train_imgs)} imgs, {len(train_labels)} labels')
    print(f'Val:   {len(val_imgs)} imgs, {len(val_labels)} labels')
    print(f'Test:  {len(test_imgs)} imgs, {len(test_labels)} labels')

    train_val_passes = set(r['pass_key'] for r in meta_rows if r['split'] in ('train', 'val'))
    test_passes_actual = set(r['pass_key'] for r in meta_rows if r['split'] == 'test')
    overlap = train_val_passes & test_passes_actual
    print(f'\nPass disjointness: train/val passes={len(train_val_passes)}, '
          f'test passes={len(test_passes_actual)}, overlap={len(overlap)}')
    if test_passes_actual:
        print(f'Test passes: {sorted(test_passes_actual)}')
    if overlap:
        print(f'OVERLAP passes (FAIL): {sorted(overlap)}')
    # Also report unusable test images (no valid pass_key) and why
    bad = [r for r in meta_rows if r['split']=='test' and not r['pass_key']]
    print(f'Test images with empty pass_key (unusable for pass-level strictness): {len(bad)}')
    if bad:
        print('  samples:', [r['image_id'] for r in bad[:10]])

    total_mb = sum(p.stat().st_size for p in DST.rglob('*') if p.is_file()) / 1e6
    print(f'\nTotal size: {total_mb:.1f} MB')
    print(f'Files: dataset.yaml, metadata.csv at {DST}')
