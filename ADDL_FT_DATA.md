# Additional Fine-Tuning Data Plan

## Current State

| Dataset | Role | Split logic | Images (approx) |
|---------|------|-------------|-----------------|
| WCS Camera Traps | FT train+val+test | Random 80/10/10 | ~2,550 |
| COCO 2017 (person/vehicle) | FT train+val+test | Random 80/10/10 | ~2,000 |
| Caltech Camera Traps (CCT20) | OOD eval only | N/A | ~500 |

**Problems with this setup:**
1. Random splitting ignores location correlation — the same physical camera can have examples in both train and val, so the model can memorize backgrounds.
2. Only one OOD eval set (CCT20). That's a thin signal for generalization.
3. WCS is geographically biased toward Africa/megafauna. There are no North American or island-fauna examples in training.
4. No "empty" images — the model has never been trained on a frame with no animals. This can lead to over-eager detections in deployment.

---

## Core Strategy: Location-Based Train/Eval Split

All LILA camera trap datasets use the **COCO Camera Traps** JSON format. Every image object has a `location` integer field identifying the physical camera site. Multiple images from the same location are visually correlated: same background, lighting pattern, and species set.

**The rule:** no `location` ID appears in both train and eval for any dataset. This applies to:
- WCS (currently ignoring `location`)
- CCT (currently all in OOD eval, not split by location at all)
- Every new dataset added

For datasets with a `sub_location` field (WCS, Island Conservation), split at `location` level; within a location, all sub-locations stay together.

### Location-split implementation sketch
```python
def location_based_split(records, location_key="location",
                         train_frac=0.80, seed=42):
    """Split records by location so no location appears in both train and eval."""
    locations = sorted({r[location_key] for r in records})
    rng = random.Random(seed)
    rng.shuffle(locations)
    n_train_locs = int(len(locations) * train_frac)
    train_locs = set(locations[:n_train_locs])
    train = [r for r in records if r[location_key] in train_locs]
    eval_ = [r for r in records if r[location_key] not in train_locs]
    return train, eval_
```

This is a drop-in replacement for the current `_split_records` inside `dataset_builder.py`, and needs to be wired into each downloader that supplies a `location` field in its records.

---

## Dataset Candidates

### Already integrated

#### WCS Camera Traps (training) — upgrade: add location
- **Location field:** `location` (int) + `sub_location` (int) in the annotation JSON
- **Action:** Parse `location` from image metadata in `wcs_downloader.py` and propagate it in the record dict. Switch `dataset_builder.py` to use location-based split for WCS records.
- **Empty images:** WCS has a category `0 = empty`. Add a `--wcs-max-empty N` flag (suggest 300–500). Empty records produce a blank `.txt` label file (YOLO background image convention).

#### Caltech Camera Traps / CCT20 (currently OOD eval) — upgrade: promote to train+eval
- **Location field:** `location` (int, range 1–140, representing 140 camera sites in SW USA)
- **Action:** Run a location-based split on CCT. Put ~80% of locations in training, ~20% in a dedicated `cct_val` split. Keep the existing `cct_ood` structure for the held-out locations; update `cct_downloader.py` to produce both a training batch and an eval batch.
- **Justification:** CCT covers SW US species (coyote, deer, rabbit, squirrel, skunk) not in WCS. Adding it to training broadens species coverage while the held-out locations still provide a meaningful OOD signal.
- **Size:** ~2,000 train images + ~500 held-out eval images (from the current 500-image sample; scale up to ~3,000 total if budget allows).

---

### New training datasets

#### Snapshot Serengeti (S1–S11) — Africa, distinct from WCS
- **Source:** https://lila.science/datasets/snapshot-serengeti
- **Bboxes:** ~150,000 ground-truth bboxes across ~78,000 images (COCO Camera Traps format, added by the SnapshotSafari annotation effort)
- **Geography/species:** Serengeti National Park, Tanzania. Lions, elephants, zebra, wildebeest, buffalo, giraffe, hyena — flagship African megafauna at high image quality, shot from fixed baited stations.
- **Complementarity to WCS:** WCS covers diverse geographies; Serengeti is a single biome but with far higher image quality and many species not well-represented in WCS.
- **Location field:** `location` (int, identifying camera trap sites within the park)
- **Annotation URL pattern:**
  ```
  https://lilawildlife.blob.core.windows.net/lila-wildlife/
      snapshotserengeti/SnapshotSerengeti_S1-11_v2.1.json.zip
  ```
  (or individual per-season ZIPs — verify current URL at the LILA page)
- **Suggested sample:** 1,500–2,000 animal images, location-based split ~80/20 for train/eval.
- **Empty images:** Serengeti has a substantial empty category. Include ~300 empty images.

#### Island Conservation Camera Traps — island ecosystems
- **Source:** https://lila.science/datasets/island-conservation-camera-traps
- **Bboxes:** ~65,000 bboxes across ~50,000 images (COCO Camera Traps format)
- **Geography/species:** 7 islands in 6 countries. Unique island fauna: feral cats, rats, rabbits, native birds and reptiles at some sites. Completely different background textures (tropical/subtropical islands) from WCS or CCT.
- **Location field:** `location` (int) + `sub_location`
- **Value:** Adds habitat diversity (dense island vegetation, different lighting) and species diversity (feral/invasive species).
- **Suggested sample:** 500–1,000 animal images, location-based split.

#### Idaho Camera Traps — ~~North American, distinct from CCT~~ **DROPPED**
- **Source:** https://lila.science/datasets/idaho-camera-traps
- **Status: Not usable.** Despite the LILA page claiming ~375k bboxes, the annotation JSON (`idaho-camera-traps.json`, 574 MB) contains **zero bbox fields** across all 1.55M annotations. It is a classification-only dataset. Verified 2026-03-14. `idaho_downloader.py` has been deleted.

---

### Second OOD eval dataset (currently missing)

Idaho was the planned source for a second OOD set. With Idaho dropped, we remain at one OOD eval set (CCT held-out locations). A future replacement could be ENA24 (Eastern North America) or Wellington Camera Traps (NZ), both of which have confirmed bbox annotations on LILA.

---

## Empty Images Strategy

**Why:** The current FT dataset contains only images with at least one annotation. A model that has never seen a clean background during fine-tuning can overfit to "there's always something here." In deployment (e.g., running MegaDetector over thousands of frames at a busy road or empty field), this causes false positives.

**How:** LILA datasets mark "empty" or "blank" images in two ways:
1. No annotations for that image at all (images with no entries in `annotations`).
2. An explicit `empty` category annotation (WCS uses category 0 = empty).

For each dataset, sample a budget of empty images and write them with a **blank label file** (an empty `.txt` file). Ultralytics YOLO treats these as background images and uses them in classification loss but not localization loss.

**Targets:**

| Source | Empty budget | Notes |
|--------|-------------|-------|
| WCS | 300–500 | Already downloadable — just skip the category filter |
| CCT | 200–300 | Most CCT "empty" images have zero annotations |
| Serengeti | 200–300 | Large empty fraction |
| Idaho | 200–300 | |
| **Total** | **~1,000–1,400** | Aim for ~15–20% of total FT images |

---

## Revised Dataset Budget

After changes, target FT dataset composition:

| Source | Role | Train imgs | Eval imgs | Geography |
|--------|------|-----------|-----------|-----------|
| WCS | animal+vehicle train | ~2,000 animal + ~52 vehicle | ~500 (held-out locs) | Global/Africa |
| COCO 2017 | person+vehicle train | ~1,500 person + ~500 vehicle | ~250 (random) | General |
| CCT (train split) | animal train | ~1,500 | — | SW USA |
| CCT (held-out locs) | OOD eval #1 | — | ~500 | SW USA |
| Snapshot Serengeti | animal train | ~1,500 | ~375 (held-out locs) | Tanzania |
| Island Conservation | animal train | ~800 | ~200 (held-out locs) | Islands worldwide |
| Empty images | train only | ~1,200 | — | Various |
| **Total** | | **~9,000** | **~1,825** | |

This is a ~3× increase in training images. OOD eval currently uses CCT held-out locations only (one set); a second OOD set (e.g. ENA24 or Wellington Camera Traps) could be added later.

---

## Implementation Plan

### Step 1: Upgrade WCS downloader
- Parse `location` from `images[].location` in the WCS JSON
- Add `location` key to each record dict
- Add `--wcs-max-empty N` flag and sampling logic

### Step 2: Upgrade CCT downloader
- Add `--mode [ood|train|both]` flag (default `ood` to preserve current behavior)
- Parse `location` from `images[].location`
- In `both` mode: location-based split → write train records + eval records separately
- Include empty images from unannotated CCT images

### Step 3: New downloader: Snapshot Serengeti
- New file: `PytorchWildlife_Export/dataset/serengeti_downloader.py`
- Mirrors the wcs_downloader pattern (annotation ZIP → parse → sample → download)
- Parse `location`; location-based split
- Category map: Serengeti uses fine-grained species → all map to `animal` (class 0)

### Step 4: New downloader: Island Conservation
- New file: `PytorchWildlife_Export/dataset/island_conservation_downloader.py`
- Same pattern; location-based split (location field is a string like `"dominicanrepublic_camara09"`)

### Step 5: Update `dataset_builder.py`
- Accept records from all sources with a `location` field
- Implement `location_based_split()` (see sketch above)
- Sources without location (COCO) fall back to random split
- Pass `empty=True` flag in records for images with no annotations
- Print summary including per-source breakdown and empty image count

### Step 6: Update `eval.py` and Makefile
- Update `eval_results_ood.csv` to track which OOD set was used (add `dataset` column or separate files)

### Step 7: Verified annotation URLs
LILA URLs are documented in the project memory (`lila_urls.md`). Key bbox-annotated sources:
- Serengeti: `snapshotserengeti-v-2-0/SnapshotSerengetiBboxes_20190903.json.zip` (10 MB, 77,496 images)
- Island Conservation: `storage.googleapis.com/public-datasets-lila/islandconservationcameratraps/island_conservation_camera_traps_1.02.zip`

---

## Open Questions for Your Review

1. **CCT to training:** Are you comfortable moving CCT from pure OOD to a train+eval split? The held-out CCT locations still serve as an OOD signal, but the training CCT locations mean it's no longer fully out-of-distribution.

2. **Serengeti vs. WCS animal diversity:** WCS already has African animals. Does adding Serengeti risk over-representing African megafauna in training? Could consider Island Conservation as the primary new training source instead, for maximum habitat diversity.

3. **Scale:** The ~9,000 train image target is ~3× current. Is this the right scale-up, or would you prefer a more conservative +50%? Fine-tuning on more images takes longer and consumes more disk.

4. **COCO person/vehicle split:** Currently COCO is random-split. COCO images don't have a `location` field. This is acceptable (street photos don't have camera-correlation issues), but worth confirming you're OK with that.
