#!/usr/bin/env bash
# ============================================================================
# run_cadp_pipeline.sh
# Wires the CADP processing chain of the release for the processed videos.
# Native frames of every accident segment are in clips_native/{clip:06d}/{k}.jpg
# (process_cadp_stage1.py --native, the layout of extract_frames of the release),
# the video-level split is in split.json and, by clip number, in split_clips.json.
#
# YOU SUPPLY: the stage-2 VATIC .txt files of the release, one per clip, named by
# the release clip number as distributed (e.g. 1407.txt for the frames of 001407/),
# placed in annotations_vatic/.
# cadp_clip_index.csv maps every clip number to its video and stage-1 segment.
# ============================================================================
set -e
CADP_ROOT="$(cd "$(dirname "$0")" && pwd)"
export CADP_IMAGE_HOME="$CADP_ROOT/clips_native/"
export CADP_MASK_HOME="$CADP_ROOT/masked_frames/"
ANNO_DIR="$CADP_ROOT/annotations_vatic"
CSV_DIR="$CADP_ROOT/csv"
VOC_DIR="$CADP_ROOT/voc"

echo "== CADP pipeline =="
echo "image home: $CADP_IMAGE_HOME"
echo "anno  dir : $ANNO_DIR"

# 0. sanity: annotations present?
n_anno=$(ls "$ANNO_DIR"/*.txt 2>/dev/null | wc -l)
if [ "$n_anno" -eq 0 ]; then
  echo "!! No VATIC .txt files in $ANNO_DIR. Drop the release's stage-2 files there first (named by clip number)."
  echo "   See cadp_clip_index.csv for the clip number -> video -> stage-1 segment mapping."
  exit 1
fi
echo "found $n_anno annotation files"

# 1. Mask crowd/far regions  (create_crowd_far.py)
echo "[1/4] masking CrowdRegion/FarRegion ..."
MASK_SCRIPT=analysis/cover_crowd_far.py; [ -f analysis/create_crowd_far.py ] && MASK_SCRIPT=analysis/create_crowd_far.py
python "$MASK_SCRIPT" --anno_dir "$ANNO_DIR" || echo "   (skip if the mask script is unavailable)"

# 2. VATIC -> flat CSV  (generate_csv.py)  -> simple_label format
echo "[2/4] VATIC -> CSV ..."
python analysis/generate_csv.py --anno_dir "$ANNO_DIR" \
       --csv_output "$CSV_DIR/cadp_all.csv" --use_mask

# 3. CSV -> PASCAL VOC XML  (csv_to_pascalvoc.py)
echo "[3/4] CSV -> PASCAL VOC ..."
python analysis/csv_to_pascalvoc.py --csv "$CSV_DIR/cadp_all.csv" \
       --output_xml_dir "$VOC_DIR/Annotations" \
       --listfile "$VOC_DIR/voc_listfile.txt"

# 4. Ground-truth accident windows + conditions template (for evaluate_cadp.py and Table 6.9)
echo "[4/5] accident windows from the Colliding spans ..."
python make_accident_windows.py --cadp_root "$CADP_ROOT"

# 5. Train / evaluate  -- requires the TF2 environment and a GPU (see README.md)
echo "[5/5] train + evaluate (needs TensorFlow 2 and a GPU) ..."
echo "  for n in 1 2 3; do for s in 0 1 2; do"
echo "    python train_cadp_tf2.py --cadp_root $CADP_ROOT --n_contexts \$n --seed \$s --epochs 30 --patience 5 \\"
echo "           --out checkpoints_n\${n}_s\${s} --history results/history_n\${n}_seed\${s}.json"
echo "    python evaluate_cadp.py --cadp_root $CADP_ROOT --split test --n_contexts \$n --seed \$s \\"
echo "           --weights checkpoints_n\${n}_s\${s} --conditions $CADP_ROOT/conditions.csv --out results"
echo "  done; done"
echo ""
echo "Split lists are in voc/ImageSets/Main/{train,val,test,trainval}.txt"
echo "Done (steps 1-4). Step 5 is printed, not executed, since it needs the GPU/TF env."
