# Concept bottleneck experiments

This folder reproduces the table of heads of a concept bottleneck model (CUB, SUN Attribute and Derm7pt; macro-F1 over
three seeds), starting from the images. A concept bottleneck model reads an image, predicts a vector of human-defined
concepts with a detector, and predicts the class from the concepts with a head. FLARE is one of the heads.

```
images --prepare--> manifest + oracle concepts --train--> detectors --export--> predicted concepts
                                                              \--crossfit--> out-of-fold concepts
concepts --run_heads--> rows of the table
```

## Data to download

Nothing is downloaded by the code and no image is copied: the manifests keep paths relative to the folders below.

| Dataset | What | Where from | Folder to pass to `--root` |
|---|---|---|---|
| CUB-200-2011 (per-image concepts) | `CUB_200_2011.tgz` (1.1 GB) | https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1 | the extracted `CUB_200_2011/` (with `images/`, `images.txt`, `attributes/image_attribute_labels.txt`, ...). `attributes.txt` is read from `attributes/`, from the folder itself or from its parent; without it the concepts get generic names |
| SUN Attribute | `SUNAttributeDB.tar.gz` (annotations) and `SUNAttributeDB_Images.tar.gz` (images, 1.7 GB) | https://cs.brown.edu/~gmpatter/Attributes/SUNAttributeDB.tar.gz and https://cs.brown.edu/~gmpatter/Attributes/SUNAttributeDB_Images.tar.gz (page: https://cs.brown.edu/~gmpatter/sunattributes.html) | a folder holding `SUNAttributeDB/` (the `.mat` files) and `images/` (the extracted images, `images/a/abbey/...`) |
| Derm7pt | the release of the 7-point criteria database (CC BY-NC-ND 4.0: do not redistribute the images) | request it at https://derm.cs.sfu.ca (code for it: https://github.com/jeremykawahara/derm7pt) | the release folder, with `meta/meta.csv`, `meta/{train,valid,test}_indexes.csv` and `images/` |

## Steps

Every step resumes when it is run again. Replace `cub` by `sun` or `derm7pt`.

```bash
pip install -e ".[cbm]"                        # torchvision and pillow; SUN also needs scipy (a dependency of FLARE)
python -m baselines.fetch_external             # the authors' code of RRL, RL-Net, DR-Net and NeuRules

python -m experiments.cbm.pipeline prepare  cub --root /data/CUB_200_2011 --work work/cub
python -m experiments.cbm.pipeline train    --work work/cub --seeds 0 1 2      # one detector per seed
python -m experiments.cbm.pipeline export   --work work/cub --seeds 0 1 2      # predicted_seed<k>_<split>.npz
python -m experiments.cbm.pipeline crossfit --work work/cub --seeds 0 1 2      # crossfit_seed<k>_<split>.npz

python -m experiments.cbm.run_heads --dataset cub --export work/cub/export --regime independent --out results/cbm
python -m experiments.cbm.run_heads --dataset cub --export work/cub/export --regime sequential  --out results/cbm
```

`train` is the expensive step and needs a GPU: Inception v3 on 299 x 299 images for 100 epochs (CUB), 30 epochs (SUN) or
up to 40 epochs (Derm7pt), three times; `crossfit` trains five more detectors per seed on four fifths of the training
images. `--limit N --epochs 1 --batch_size 4` (a few images per split) checks that everything runs. The detector
recipes are in `detector.py`; their weights start from the ImageNet weights that torchvision downloads on first use.

## What the table contains

Each cell is the mean and standard deviation over head seeds 33, 34 and 35. Head seed 33 + k is paired with the detector
of seed k, and every head is scored on the test images' concepts as that detector predicts them:

| Column | Training concepts of the head | Run with |
|---|---|---|
| GT | the annotated concepts (`oracle`) | `--regime independent` |
| Pred. | the cross-fitted concepts (`crossfit_seed<k>`) | `--regime sequential` |

| Row | Head | Settings |
|---|---|---|
| FLARE | `flare_head.py` | FLARE-simple: 8 models of 100 + 20 rules, 2 slots, each concept a variable with the conditions TRUE (truth c) and FALSE (truth 1 - c); the other values are those of FLARE |
| Logistic regression | scikit-learn | class-balanced, 3000 iterations, on the concepts as they are |
| LogicCBM | `baselines/cbm_heads.py` | 250 neurons (or classes + 50), as in the paper's code |
| RRL, RL-Net | the authors' code | on concepts cut at 0.5; width / rules = max(250, classes + 50) |
| DCR | `baselines/cbm_heads.py` | temperature 100, 10 or 1 chosen on the validation macro-F1 |
| NeuRules | the authors' code | rules = max(250, classes + 50), on the concepts as they are |
| GradTree | `baselines/gradtree.py` | on concepts cut at 0.5; depth max(6, ceil(log2 classes) + 1) |
| DR-Net | the authors' code | on concepts cut at 0.5; only with at most 20 classes (Derm7pt) |

`run_heads` writes `<out>/<dataset>_<regime>.csv` with one row per head and seed (`f1`, `acc`, seconds and, for FLARE,
its rule counts) and FLARE's printed rules in `<out>/rules/`.

## Splits

| Dataset | Concepts | Classes | Train / validation / test |
|---|---|---|---|
| CUB, per image | 112 | 200 | 4,794 / 1,200 / 5,794 (the official split; 20% of each class's training images, seed 0, validate) |
| SUN Attribute | 102 | 717 | 11,472 / 1,434 / 1,434 (per class, seed 0; a concept is present with at least two of three votes) |
| Derm7pt | 12 | 5 | 413 / 203 / 395 (the authors' indexes) |

## Notes

* GPU detectors are not bit-reproducible across machines: the head results move a little with the detectors, as the
  predicted columns of any such pipeline do. The heads themselves are deterministic on the CPU given the concept files.
* The two macro-F1 computations of the table differ slightly (`ConceptHead.macro_f1`): over the classes in the test
  labels for the logistic regression and the rule learners on cut concepts, over those in the labels or the predictions
  for the others. They only differ when a class is predicted but absent from the test labels.
