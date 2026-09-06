# Debris Feature-Learning Report (Sep 5, 2026)

Question asked: *"learn the model's features and look if it improves our model"*

## Method

Confidence-sweep feature probe (`scripts/diag_debris_features.py`): for each model,
debris recall/AP measured at conf 0.001 → 0.25 on two test sets:

- **v5 NOAA test** — whole passes genuinely unseen by the v5 model's train
- **h8_unseen_test** — frames from passes SEEN by both models' training

Logic: if debris fails only because detections are low-confidence, recall should
rise as the threshold drops to the floor (0.001). If recall stays ~0 even at the
floor, the feature itself is absent for those inputs — no threshold recovers it.

## Results

### v5 4-class model (10-epoch mock)

| conf | v5 NOAA test (UNSEEN passes) | h8_unseen_test (SEEN passes) |
|---|---|---|
| 0.001 | AP 0.0025, **R 0.010** | AP 0.355, **R 0.387** |
| 0.005 | AP 0.0016, R 0.091* | AP 0.352, R 0.387 |
| 0.25  | AP 0.0005, R 0.010 | AP 0.252, R 0.323 |

\* single extra TP at the floor; unstable, not a trend.

**Verdict: the debris feature is ABSENT on unseen passes, not merely weak.**
At the confidence floor (0.001) recall is 1% on unseen passes vs 39% on seen
passes with the same weights. The model has memorized pass-specific seabed
texture — it does not carry a transferable debris feature across surveys.

### Core_model (fp32 export, single-class, trained on ALL h8 passes)

| conf | v5 NOAA test (SEEN for Core) | h8_unseen_test (SEEN for Core) |
|---|---|---|
| 0.001 | AP 0.924, R 0.787 | AP 0.882, R 0.984 |
| 0.25  | AP 0.870, R 0.787 | AP 0.826, R 0.919 |

**Verdict: Core's debris feature is crisp.** Recall is essentially flat from
conf 0.001 → 0.25, i.e. every true positive fires above conf 0.25 — a confident,
well-learned feature. But Core has never been measured on a genuinely unseen
pass: every evaluation set it faces is built from passes in its own training.

## Feature lessons — what this tells us about improving v5

1. **Not a threshold/calibration problem.** Dropping conf to 0.001 does not
   recover unseen-pass debris (1% recall). No conf-sweep lever exists.
2. **Not a resolution problem** (256 vs 512, both training and inference,
   measured earlier: debris ~0 on the honest test at every resolution).
3. **Not an epoch-count question by itself** — full-length runs capped at
   0.06–0.13 val AP50 even on the leaky (pass-seen) test.
4. **Core's success recipe is: single-class + ALL curated passes + full epochs
   + crisp confidences.** Its skill is real but was only ever graded on seen
   passes. The one honest proxy we have (v5 at 1% recall on unseen passes)
   suggests pass-texture memorization is the ceiling for both — Core simply had
   no unseen test to reveal it.

## Data inventory: no genuinely new labeled survey exists on disk

Every labeled folder traces back to the same two surveys (E3/H11833 + E4):

| Folder | What it really is | New passes? |
|---|---|---|
| e3 / e3_enhanced / e3_fixed | E3 re-crops (QA iterations) | none |
| e3_v2 | E3 re-crops re-centered on sonar returns | none (same TGT runs) |
| e4 | E4 survey | none (h8 already has E4) |
| e5 | E4 + synthetic SSS noise (`generate_e5_noisy.py`) | none (synthetic E4) |
| f6 / f6g7 | E3+E4 mixes | none |
| g7 | 7,305 frames, 0 debris positives (background) | no labels |
| h8_test | 500 pure-background frames | no labels |

The "23 not-in-h8" positives in e5/f6 are isolated missing frames of E4 runs
already present in h8 — near-duplicates, not new passes.

## What actually moves debris (in order)

1. **A genuinely new labeled survey** (new capture geometry / location) — the
   only input that can teach pass-invariant debris features. No such labels
   exist on disk; raw TIFFs of other lines exist but are unlabeled (labeling
   project, hours of manual work).
2. **Re-train debris single-class** (Core's recipe: all curated h8 frames, full
   epochs, no cls_pw dilution) and report the seen-pass capability honestly —
   maximizes what this data can give, but does not create unseen-pass skill.
3. Everything else (conf, resolution, epochs, distillation from Core's ONNX) is
   measured dead for unseen-pass debris: Core's weights can't be ported (no
   .pt; only ONNX), and distilling them would leak seen-pass context into the
   honest test anyway.
