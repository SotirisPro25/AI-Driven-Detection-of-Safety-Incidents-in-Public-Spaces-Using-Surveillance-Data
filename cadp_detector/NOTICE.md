# Notice

The files in this folder contain code ported from the CADP repository
(CarCrash_forecasting_and_detection, A. P. Shah, J.-B. Lamare, T. Nguyen-Anh and A. Hauptmann,
AVSS 2018, https://github.com/ankitshah009/CarCrash_forecasting_and_detection), which is
distributed under the GNU General Public License, version 3. This folder is therefore distributed
under the same license (see LICENSE). The ports were rewritten for TensorFlow 2 and Python 3 by
Sotirios Prountzos in 2026.

| File | What comes from the CADP repository |
|---|---|
| `annotations.py` | port of `analysis/annotations/utils.py` (`read_vatic`, `find_boundary`, `extract_frames`, `cross_validation_folds`) |
| `losses_tf2.py` | the four loss functions of `analysis/keras_frcnn/losses.py` |
| `train_cadp_tf2.py` | anchor-target assignment after `data_generators.calc_rpn`, proposal decoding and suppression after `roi_helpers.rpn_to_roi` and `non_max_suppression_fast`, head sampling after `train_cadp.py` |
| `evaluate_cadp.py` | inference path and AP protocol after `measure_map.py`, `apply_regr` after `roi_helpers.apply_regr` |
| `contextual_roi_pooling.py` | the window geometry of `improved_frcnn/layers.py` (`ContextualROIPoolConvLayer`), re-implemented with the clamping in effect |
| `bench_detector.py` | the same layer, proposal decoding and `apply_regr`, for the timing of the inference path |
| `accident_dataset_toolkit.py` | the segment-boundary rule of `find_boundary` |
| `process_cadp_stage1.py` | the clip construction of `extract_frames` |

`make_accident_windows.py` and `run_cadp_pipeline.sh` contain no ported code. They are kept here
because they import or call the files above. `run_cadp_pipeline.sh` also calls scripts of the CADP
repository (`generate_csv.py`, `csv_to_pascalvoc.py`, `create_crowd_far.py`), which are not
included and must be taken from that repository.
