# AI-Driven Detection of Safety Incidents in Public Spaces: code and split lists

This repository holds the code and the split lists behind the experimental chapter of the MSc thesis

> S. Prountzos, *AI-Driven Detection of Safety Incidents in Public Spaces Using Surveillance Data*,
> Postgraduate Diploma Thesis, Interdisciplinary Interuniversity Programme "Techno-Economic Systems",
> National Technical University of Athens, 2026.

It covers two datasets:

- **IEEE DataPort.** The Traffic Accident Detection Video Dataset (doi:10.21227/tjtg-nz28). The repository contains the leakage audit, the source-disjoint re-split and the eight-class clip classifiers.
- **CADP.** The CCTV accident dataset of Shah et al., AVSS 2018 (doi:10.1109/AVSS.2018.8639160). The repository contains the processing, the motion-based temporal baseline, the motion clip classifier and the Faster R-CNN detector with contextual region pooling.

Section, table, figure and image numbers below refer to the final thesis.

**No video, frame or annotation data is distributed here.** Both datasets must be obtained from their providers. See [Data and rights](#data-and-rights).

## Layout and licenses

| Folder | Content | License |
|---|---|---|
| `cadp_detector/` | CADP annotation reading, clip construction, accident windows and the detector (training, evaluation, timing). Contains code ported from the CADP repository (see `NOTICE.md`). | GPL-3.0 (`cadp_detector/LICENSE`) |
| `cadp_motion/` | CADP frame extraction and video-level split, motion signals, training-free temporal baseline, motion clip classifier | MIT (`cadp_motion/LICENSE`) |
| `ieee_dataport/` | IEEE DataPort audit, frame-overlap check, re-split, normalization, Kaggle preparation and training, clip classifier, split-list builder, Image 2.2 | MIT (`ieee_dataport/LICENSE`) |
| `data/splits/` | the IEEE DataPort source-disjoint split, the frame-overlap check of the supplied partition and the CADP video-level split | CC BY 4.0 (`data/splits/LICENSE`) |

The GPL-3.0 code is kept in its own folder so that the MIT folders stay independent of it. No file in `cadp_motion/` or `ieee_dataport/` imports a file of `cadp_detector/`.

## Scripts and what they produce

### IEEE DataPort (`ieee_dataport/`)

| Script | Thesis |
|---|---|
| `ieee_audit_tools.py listing` | census, clip families and source leakage of the audit, and the clip pairs whose frame ranges may intersect across partitions (Section 2.3.4) |
| `verify_frame_overlaps.py` | the clip pairs of Finding 17: candidates from the file names, each confirmed or rejected on the decoded frames (Section 2.3.4, Image 2.2) |
| `ieee_audit_tools.py probe`, `metadata-check` | container facts of the probed sample and the classifier on container facts alone (Section 2.3.4) |
| `ieee_audit_tools.py disjoint` | the source-disjoint split (Section 6.2.3) |
| `ieee_audit_tools.py normalize` | re-encoding of every clip to 5.0 s, 25 fps, 1280×720, H.264, no audio (Section 6.2.3) |
| `kaggle_prepare.py` | all of the above on Kaggle (CPU session), plus the frame cache for the GPU runs. `SHARD=k/n` splits the work over n sessions |
| `train_clip_classifier.py` | one run of the 2D baseline (`r2d`, ResNet-50) or the spatio-temporal network (`r2p1d`, R(2+1)D-18). Configuration in Table 6.5 |
| `kaggle_train.py` | the 18 runs of Section 6.3.2 on a Tesla T4 (two models, the supplied partition, the source-disjoint split and the clips as distributed, three seeds). Results in Table 6.7, Figures 6.4 and 6.5, and the classifier rows of Table 6.8 |
| `kaggle_all.py` | preparation and training in one GPU session |
| `make_kaggle_cells.py` | the one-line notebook cells that unpack these scripts on Kaggle |
| `parse_kaggle_log.py` | rebuilds the result files from the `RESULT_*` lines of a Kaggle log and checks them |
| `build_split_list.py` | the published split list, from the split written on Kaggle and a listing of the original copy |
| `make_image_2_2.py` | Image 2.2: the same source frames in a training clip and a test clip, faces and number plates blurred |

### CADP (`cadp_motion/` and `cadp_detector/`)

| Script | Thesis |
|---|---|
| `cadp_motion/process_cadp.py` | frame extraction, append-only identifiers and the video-level partition (Section 6.2.1, Table 6.1) |
| `cadp_detector/process_cadp_stage1.py` | accident clips and normal-footage clips from the stage-1 segments, accident windows, the statistics of Sections 6.1 and 6.2.1 |
| `cadp_detector/annotations.py` | reading of the stage-2 VATIC annotation (Table 6.2, Figure 6.1) |
| `cadp_detector/make_accident_windows.py` | ground-truth accident windows (Section 6.3.3) |
| `cadp_motion/extract_full.py`, `motion_features.py` | full-length frames at 10 fps and the four motion signals (Section 6.3.4) |
| `cadp_motion/temporal_baseline.py`, `temporal_references.py` | training-free temporal baseline and the always-on and random references (Table 6.9) |
| `cadp_motion/clip_classifier_motion.py --grid` | accident versus normal-footage clip classifier on motion statistics (Table 6.10). Without `--grid`, the earlier two-candidate configuration |
| `cadp_detector/train_cadp_tf2.py`, `contextual_roi_pooling.py`, `losses_tf2.py` | the detector of Sections 4.1.6 and 4.1.7 and Table 4.1, configuration in Table 6.4 |
| `cadp_detector/evaluate_cadp.py` | the inference path of the detector (detection and temporal metrics were not reported in the thesis) |
| `cadp_detector/bench_detector.py` | cost of the detector for n = 1, 2, 3 context windows on a Tesla T4 (Section 6.4.1, Tables 6.6 and 6.8, Figure 6.3) |
| `cadp_detector/run_cadp_pipeline.sh` | the CADP preprocessing chain with the scripts of the CADP repository |

## Environment

- **Clip classifiers.** Kaggle notebook with an NVIDIA Tesla T4 (16 GB), PyTorch 2.11.0+cu128 and torchvision 0.26.0+cu128. Mixed precision, batch 8 with two accumulation steps (effective batch 16), at most 30 epochs with early stopping on the validation macro-F1.
- **Detector timing.** TensorFlow 2.20.0 on the same GPU type.
- **CPU steps** (CADP processing, motion signals, baseline, motion classifier). A two-core Intel Xeon at 2.8 GHz, with Python 3.13, NumPy 2.5, OpenCV 5.0, scikit-learn 1.9, SciPy 1.18 and FFmpeg 6.1.

## Reproducing the experiments

### IEEE DataPort

1. **Get the dataset.** Download it from IEEE DataPort. The partitions are `train/`, `val/` and `test/` with eight class folders inside. A doubled partition level (`train/train/...`) and the class folder spellings of the release are accepted.
2. **Audit it.** `python ieee_dataport/ieee_audit_tools.py listing --root <copy> --out ieee_listing.csv` prints the census and the leakage figures of Section 2.3.4. `python ieee_dataport/verify_frame_overlaps.py --root <copy>` then checks the candidate pairs on the frames and prints the counts of Finding 17 (about four minutes on two CPU cores, OpenCV needed).
3. **Run the experiments on Kaggle.**
   - Add the dataset as notebook input.
   - Run `kaggle_prepare.py` in a CPU session. Its output (`audit_outputs/`, `cache/`) becomes the input of a GPU session that runs `kaggle_train.py`.
   - `SEEDS=0,1,2` selects the seeds. `make_kaggle_cells.py` builds the notebook cells.
   - Each run writes `results/results_<model>_<split>[_raw]_seed<seed>.json`, with accuracy, macro-F1, per-class recall, the confusion matrix, every test prediction, the training history and the inference time per clip.
4. **Use the published split instead of recomputing it** (see the next section). `ieee_audit_tools.py disjoint` reproduces it exactly from the file names of the Kaggle copy, but not from the original names. To build the disjoint trees of an original copy from the list, run the snippet below from the repository root, then normalize as in `kaggle_prepare.py`:

```python
import csv, os, sys
sys.path.insert(0, "ieee_dataport")
from ieee_audit_tools import enumerate_tree, link_or_copy
root = "<copy as distributed>"
path = {(r["partition"], r["cls"], r["name"]): r["path"] for r in enumerate_tree(root)}
for r in csv.DictReader(open("data/splits/ieee_source_disjoint_split.csv", encoding="utf8")):
    if r["used_in_runs"] != "1":
        continue
    for name in r["ieee_name"].split(" | "):
        dst = os.path.join(root, r["disjoint_partition"] + "_disjoint", r["class"])
        os.makedirs(dst, exist_ok=True)
        link_or_copy(path[(r["distributed_partition"], r["class"], name)], os.path.join(dst, name))
```

### CADP

1. **Get the data.**
   - The videos: follow the instructions of the CADP repository (https://github.com/ankitshah009/CarCrash_forecasting_and_detection). Save each one as `<youtube_id>.mp4`.
   - The stage-1 temporal annotation (`annotations_<timestamp>_stage1.json`) and the stage-2 VATIC files: from the same release.
2. **Restore the thesis partition.** Copy `data/splits/cadp/id_mapping.json` and `data/splits/cadp/split.json` into the CADP root. `process_cadp.py` keeps existing assignments, so the 90/20/19 video partition of the thesis is restored. Without these files, a new partition is drawn.
3. **Run the CPU chain:**

```bash
python cadp_motion/process_cadp.py <videos> CADP
python cadp_detector/process_cadp_stage1.py --stage1 annotations_stage1.json --cadp_root CADP --videos <videos>
python cadp_motion/extract_full.py --cadp_root CADP --videos <videos> --splits val test
python cadp_motion/motion_features.py --frames_root CADP/clips --out CADP/motion/clips.npz
python cadp_motion/motion_features.py --frames_root CADP/full_frames --out CADP/motion/full.npz
python cadp_motion/temporal_baseline.py --features CADP/motion/full.npz --windows CADP/accident_windows_full.json \
       --split CADP/split.json --out results/temporal_baseline.json
python cadp_motion/temporal_references.py --features CADP/motion/full.npz --windows CADP/accident_windows_full.json \
       --split CADP/split.json --out results/references.json
python cadp_motion/clip_classifier_motion.py --features CADP/motion/clips.npz --manifest CADP/clips_manifest.csv \
       --out results/clip_classifier_motion.json --grid
```

4. **Time the detector.** Run `bench_detector.py` as one notebook cell on a Tesla T4. It prints one `DETBENCH` line per setting and form of execution.

## The split lists

### IEEE DataPort: `data/splits/ieee_source_disjoint_split.csv`

One row per clip of the copy the experiments ran on (5,681 rows), with these columns:

| Column | Meaning |
|---|---|
| `disjoint_partition` | partition in the source-disjoint split: `train` 3,977, `val` 852, `test` 852 |
| `class` | one of the eight classes of the thesis |
| `ieee_name` | the file name in the dataset as distributed by IEEE DataPort |
| `kaggle_name` | the file name in the Kaggle copy, on which the split was computed |
| `distributed_partition` | the partition the clip has in the release |
| `used_in_runs` | 0 for the ten clips left out of every run, otherwise 1 |

With `used_in_runs = 1`:
- the source-disjoint split holds 3,972, 848 and 851 clips,
- the supplied partition holds 3,912, 1,047 and 712 clips.

These are the counts of Section 6.2.3 and Table 6.7.

**Why the two name columns differ.** In part of the Kaggle copy, the file names lost every character outside ASCII and every `!`, and leading and trailing spaces were then removed. For example, `Трафик в стамбуле-splitted-130-135.mp4` became `-splitted-130-135.mp4`. The split was computed on those names, so `kaggle_name` is what the code saw. `ieee_name` gives the original.

**The 9 files of the release absent from the Kaggle copy** (Section 6.2.3) are listed in `ieee_absent_from_kaggle_copy.csv`:
- two clips are simply absent,
- in seven cases, two original test clips of No Accident Normal Traffic collapsed to one Kaggle name. Only one of the two was in the copy, and names alone cannot tell which. For these rows, `ieee_name` lists both candidates, separated by ` | `. One of the seven rows is also among the ten excluded clips. The other six are in the training partition of the split, so the snippet above adds six clips there.

**The ten excluded clips.** They carry five Kaggle names that occur in the same class in the validation and the test partition. The frame cache identifies a clip by class and name, so these clips were left out of every run.

**What the name changes did to the grouping.** The split groups clips by source title. Because names lost characters in some parts of the Kaggle copy and not in others, one source could appear under two titles and be grouped as two sources. When titles are compared on their letters and digits only, the split shares three titles across partitions, 53 clips in all (0.9% of the dataset), as Section 6.2.3 reports:
- one title with and without `!`, between training and validation (19 clips),
- one with `Tráfico` and `Trfico`, between validation and test (10 clips),
- the empty title of names reduced to punctuation, between training and test (24 clips).

**Provenance and checks.**
- `ieee_split_disjoint_kaggle.csv` is the file written by `kaggle_prepare.py` in the Kaggle preparation session, unchanged. Its class counts per partition match `results/disjoint_counts.json` of that session.
- `ieee_audit_tools.py disjoint` (seed 0), run on the Kaggle names, reproduces it row for row.
- `build_split_list.py` derives the published list from it.

### IEEE DataPort: `data/splits/ieee_shared_frames_check.csv`

The 390 candidate pairs of `verify_frame_overlaps.py` on the supplied partition, one row per pair, with the best lag, the frames in common, the mean squared difference of the aligned grey frames, the motion residual and the verdict (`shares_frames`). 78 pairs share frames: 39 between training and validation, 31 between training and testing and 8 between validation and testing. They involve 36 validation clips and 27 test clips that repeat footage of a training clip, and five source videos. No frames are shared by all three partitions.

**Frame numbers count from the start of the segment.** Long sources were first cut into segments of about ten minutes, named `-splitted-A-B` (seconds), and the frame numbers that follow count from the start of that segment. Three sources were cut into overlapping segments (for example 0-600 and 254-854), so the same event was cut from both under different frame numbers. Before rev8f the audit compared frame numbers of different segments as if they were source frames. It reported 176 pairs (84, 73 and 19 per pair of partitions), of which 38 share frames, and it missed the 40 pairs that come from overlapping segments. `ieee_audit_tools.frame_overlaps` now keeps the segment, and the count of the thesis is the pixel check. The `overlap_pairs` figures in the `AUDIT` lines of `kaggle_train.py` (83, 72 and 21 on the Kaggle copy) were computed the old way. They are kept as logged and are not used in the thesis.

### CADP: `data/splits/cadp/`

- `cadp_video_split.csv`: the video-level partition of the 129 processed videos (90 training, 20 validation, 19 test), as identifier, YouTube identifier and partition.
- `id_mapping.json` and `split.json`: the same partition in the form that `process_cadp.py` reads.

Clips and frames inherit the partition of their video.

## Data and rights

- **IEEE DataPort.** The dataset states no license. Its clips are cut from videos uploaded online by third parties, and channel watermarks are visible in most of them.
  - Frames shown in the thesis (Images 2.1 and 2.2) are reproduced for scholarly commentary, with faces and number plates blurred.
  - Anyone republishing frames should clear the rights for that use and blur faces and number plates. `make_image_2_2.py` does this with boxes placed by hand and checked on enlarged crops.
- **CADP.** The code and annotations of the CADP repository are GPL-3.0. Use of the data is restricted to non-commercial purposes. The videos are YouTube videos and are not redistributed here.
- **This repository** contains file names, partition assignments, video identifiers and code only.
