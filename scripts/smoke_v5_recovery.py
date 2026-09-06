#!/usr/bin/env python3
"""
Smoke test for scripts/colab_train_v5_drive.py Cells 4-6 recovery machinery.

Runs the REAL colab cell code (imported by extracting the cell blocks), but:
  - locally, with tiny epochs (2 per stage), batch 8, on the v5 dataset
  - SIMULATES a session death after Stage 1 (deletes the local weights dir)
  - verifies Cell 5 recovers s1_best.pt from the Drive backup and continues

Proves: build ESI -> Stage1 train -> Drive backup -> death -> recovery ->
Stage2 continuation (0 missing/0 unexpected) -> Stage2 backup -> Cell 6 winner eval.
"""
import ast, re, sys, os, shutil, textwrap
from pathlib import Path

ROOT = Path('/home/ashish/sonar-vision')
SCRIPT = ROOT / 'scripts/colab_train_v5_drive.py'
DS = ROOT / 'datasets/sonarvision_multisource_v5'
SMOKE = ROOT / 'mock_runs/smoke_v5_recovery'
DRIVE = SMOKE / 'drive'                      # simulated MyDrive
OUT = str(SMOKE / 'results')                 # simulated /content/results
os.makedirs(DRIVE, exist_ok=True)

# ---------------------------------------------------------------- extract cells
src = SCRIPT.read_text()
markers = [(m.start(), m.group(0)) for m in re.finditer(r'^# ={10,}\n# CELL (\d+)', src, re.M)]
blocks = {}
for i, (pos, _) in enumerate(markers):
    end = markers[i + 1][0] if i + 1 < len(markers) else len(src)
    blocks[int(re.search(r'CELL (\d+)', _).group(1))] = src[pos:end]
for need in (2, 3, 4, 5, 6):
    assert need in blocks, f'cell {need} missing'

# ---------------------------------------------------------------- preamble (replaces Cell 1)
preamble = f"""
import torch, torch.nn as nn, os, yaml, glob
from ultralytics import YOLO
from ultralytics.nn.modules.block import C2f
from ultralytics.models.yolo.detect.train import DetectionTrainer

DATASET_DIR = '{DS}'
DATA = f'{{DATASET_DIR}}/dataset.yaml'
cfg = yaml.safe_load(open(DATA))
cfg['path'] = DATASET_DIR
OUT = '{OUT}'
OUT_DRIVE = '{DRIVE}'
os.makedirs(OUT_DRIVE, exist_ok=True)

# tiny-epoch overrides (2 per stage) — structure identical, just short
STAGE1 = dict(epochs=2, patience=2, lr0=0.01, lrf=0.01, warmup_epochs=2)
STAGE2 = dict(epochs=2, patience=2, lr0=0.005, lrf=0.01, warmup_epochs=3,
              freeze=10, batch=8)
IMGSZ = 256; BATCH1 = 8; CLS_PW = 0.5
TRAIN_CONFIG = {{
    'mosaic': 0.0, 'mixup': 0.0, 'fliplr': 0.0, 'flipud': 0.0, 'degrees': 0.0,
    'translate': 0.05, 'scale': 0.2, 'project': OUT, 'exist_ok': True,
    'plots': True, 'verbose': True, 'workers': 0,   # 0 = no forkserver respawn
}}
"""
# Cell 4/5 hardcode f'{STAGE1["epochs"]}' etc — fine since we redefine above.
# But cells also re-define STAGE1/STAGE2 in Cell 3 — skip Cell 3 (we already
# defined the config). Run preamble (defs) + Cell 2 (SE classes) then cells 4-6.
cell3_src = blocks[3]
# drop the STAGE/IMGSZ/BATCH reassignments from cell 3 by only keeping prints
# -> simpler: run cell 2, then execute a REDUCED cell 4/5/6 with config injected.
# We execute cell 4/5/6 code verbatim but they reference STAGE1..TRAIN_CONFIG &
# BATCH1 — provided by preamble. Cell 4 does `**STAGE1, **TRAIN_CONFIG` — good.
exec_code = preamble + '\n' + blocks[2]
# Cell 2 ends with a print of 'ready'. It also has no epochs refs. Safe.
# Execute cells 4,5,6 verbatim:
exec_code += '\n' + blocks[4] + '\n' + blocks[5] + '\n' + blocks[6]
# remove any leftover 'from google.colab' or drive refs inside (shouldn't exist in 4-6)
ns = {'__name__': '__main__'}
# need pandas & Path & shutil available inside cells: imported per-cell by the
# colab script itself (Cell 6 imports pandas/pathlib/shutil). good.

import sys as _sys
_CLASS_NAMES = ['SEBlock', 'C2fWithSE']
def _register_classes():
    """Register exec'd classes into the REAL __main__ module so torch.save /
    torch.load can resolve C2fWithSE by reference (pickle requires
    getattr(sys.modules['__main__'], name) is cls). Colab cells naturally run
    in __main__; exec'd code does not, so mirror it here."""
    _main = _sys.modules['__main__']
    for _n in _CLASS_NAMES:
        if _n in ns:
            setattr(_main, _n, ns[_n])

# ---------------------------------------------------------------- SIMULATE SESSION DEATH
# Before running, we pre-create a fake 'BEST1 missing' scenario is impossible
# before stage 1 trains, so we do it AFTER stage 1: but exec runs 4+5+6 in one
# go. To simulate death between 4 and 5, split execution at cell boundaries.
def run(cell_src, label):
    print(f'\n{"#" * 60}\nEXECUTING {label}\n{"#" * 60}', flush=True)
    exec(cell_src, ns)
    _register_classes()

# stage 1: exec CELL 2 (class defs) FIRST so _register_classes() runs
# BEFORE training — torch.save inside Cell 4 pickles C2fWithSE and needs it
# in real __main__ at save time, not after the exec returns.
def _stage(extra_cell, label):
    exec(preamble + '\n' + blocks[2], ns)
    _register_classes()
    exec(extra_cell, ns)
    _register_classes()
    print(f'\n{"#" * 60}\nEXECUTING {label}\n{"#" * 60}', flush=True)

_stage(blocks[4], 'CELL 4 = STAGE 1 + DRIVE BACKUP')
best1 = f'{OUT}/model_esi_v5_s1/weights/best.pt'
assert os.path.exists(best1), 'Stage 1 best.pt missing'
assert os.path.exists(f'{DRIVE}/s1_best.pt'), 'Stage-1 Drive backup missing!'
print('\n✓ Stage 1 produced best.pt AND s1_best.pt Drive backup')

# ---- simulate Colab session death: wipe ALL local results -------------------
print('\n💀 SIMULATING SESSION DEATH — wiping /content/results ...')
shutil.rmtree(OUT, ignore_errors=True)
assert not os.path.exists(best1), 'wipe failed'
print('✓ local weights gone; Drive backup is the only survivor')

# ---- recovery: fresh session, exec CELL 2 then CELL 5 only -------------
ns['BEST1'] = f'{OUT}/model_esi_v5_s1/weights/best.pt'
_stage(blocks[5], 'CELL 5 = RECOVERY + STAGE 2')
assert os.path.exists(best1), 'recovery did not restore best.pt'
print('\n✓ Cell 5 recovered Stage-1 weights from Drive and ran Stage 2')
assert os.path.exists(f'{DRIVE}/s2_best.pt'), 'Stage-2 Drive backup missing!'
print('✓ s2_best.pt backup created')

# ---- Cell 6 winner eval ------------------------------------------------------
# NOTE: exec-harness quirk — ultralytics .val() spawns dataloader workers that
# re-import this script as __main__ and re-run training. Inject workers=0 into
# every val() call in the CELL 6 source (real Colab needs no such patch).
cell6_src = re.sub(r'val\(data=DATA, imgsz=IMGSZ, conf=([0-9.]+), split=([a-z]+), verbose=False\)',
                   r'val(data=DATA, imgsz=IMGSZ, conf=\1, split=\2, verbose=False, workers=0)',
                   blocks[6])
cell6_src = re.sub(r'val\(data=DATA, imgsz=IMGSZ, split=([a-z]+), verbose=False\)',
                   r'val(data=DATA, imgsz=IMGSZ, split=\1, verbose=False, workers=0)',
                   cell6_src)
# winner-eval form: split=split (variable) — inject workers=0 there too
cell6_src = re.sub(r'val\(data=DATA, imgsz=IMGSZ, conf=([0-9.]+), split=split, verbose=False\)',
                   r'val(data=DATA, imgsz=IMGSZ, conf=\1, split=split, verbose=False, workers=0)',
                   cell6_src)
# quoted split ('val' / 'test') — the candidate loop uses split='val'
cell6_src = re.sub("val\\(data=DATA, imgsz=IMGSZ, conf=([0-9.]+), split='([a-z]+)', verbose=False\\)",
                   "val(data=DATA, imgsz=IMGSZ, conf=\\1, split='\\2', verbose=False, workers=0)",
                   cell6_src)
ns['BEST1'] = best1
ns['BEST2'] = f'{OUT}/model_esi_v5_s2/weights/best.pt'
run(cell6_src, 'CELL 6 = WINNER EVAL (val+test per-class)')
print('\n' + '=' * 60)
print('SMOKE PASSED — full recovery chain works end to end')
print('=' * 60)
