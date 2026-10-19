"""
annotations.py -- the annotation utilities of the CADP repository
(analysis/annotations/utils.py of CarCrash_forecasting_and_detection), ported without the
glog, pandas and matplotlib dependencies. The functions keep the semantics of the release
line for line, so that the stage-2 VATIC files and the stage-1 JSON are read exactly as the
authors' own generate_csv.py and create_crowd_far.py read them.

VATIC line format (whitespace separated):
    trackId xmin ymin xmax ymax frame lost occluded generated "label" ["attribute"]

What the release relies on (do not "fix"):
  - read_vatic stores the box as [ymin, xmin, ymax, xmax]: downstream code unpacks
    `y1, x1, y2, x2 = frame["box"]`.
  - visible = (lost == 0). Only the eleventh token is read as the attribute, so a frame carries
    at most one attribute ("Colliding" marks the collision frames of a track).
  - find_boundary works on the "Separator" tracks, not on the Colliding attribute: with at
    least two separators the clip span runs from the earliest to the latest FIRST VISIBLE frame
    of a separator; with fewer than two, [0, -1] means the whole clip.
  - extract_frames enumerates the segments of the stage-1 JSON in file order of the videos and
    in start order within a video; the k-th segment overall becomes directory {k:06d}, which is
    the name of its stage-2 VATIC file. A segment's start and end are the first two of its
    sorted keyframe timestamps multiplied by the video frame rate; a segment with fewer than two
    keyframes is skipped.
  - cross_validation_folds draws train and test sets from the list of CLIPS with
    train_test_split(test_size=0.33, random_state=42), so clips of one video can fall on both
    sides. The thesis keeps its video-level split instead (split.json, split_clips.json).
"""
import os
import numpy as np

# Canonical CADP object classes (from csv_to_pascalvoc.py CADP_CLASSES)
OBJECT_LABELS = {"Car", "Bus", "Person", "Two-wheeler", "Three-wheeler", "Others"}
# Special region or utility labels that are not objects
REGION_LABELS = {"FarRegion", "CrowdRegion", "Separator"}


def read_vatic(fpath):
    """utils.read_vatic of the release: {trackId: {"label": str, "frames": {frame: {...}}}}."""
    annotations = {}
    with open(fpath) as fp:
        for line in fp.readlines():
            parts = line.split(" ")
            if len(parts) < 10:
                continue
            frame = {
                "box": [int(parts[2]), int(parts[1]), int(parts[4]), int(parts[3])],   # [ymin, xmin, ymax, xmax]
                "attribute": "",
                "visible": (int(parts[6]) == 0),
            }
            if len(parts) > 10:
                frame["attribute"] = parts[10].strip().strip('"')
            tid = int(parts[0])
            label = parts[9].strip().strip('"')
            if tid not in annotations:
                annotations[tid] = {"frames": {int(parts[5]): frame}, "label": label}
            else:
                if label != annotations[tid]["label"]:
                    print("warning: an illegal track (label changes within track %d)" % tid)
                annotations[tid]["frames"][int(parts[5])] = frame
    return annotations


def is_unusual_track(track):
    """utils.is_unusual_track: a track is unusual when one of its frames carries 'Colliding'."""
    for fr in track["frames"].values():
        if fr["attribute"] == "Colliding":
            return True
    return False


def find_boundary(annotations):
    """utils.find_boundary: the clip span delimited by the Separator tracks, or [0, -1]."""
    separators = [tid for tid in annotations if annotations[tid]["label"] == "Separator"]
    if len(separators) > 2:
        print("info: more than two separators, only the earliest and the latest count")
    if len(separators) < 2:          # even with one separator the whole clip is counted
        return [0, -1]
    intervals = []
    for tid in separators:
        fidx = [fid for fid, fr in annotations[tid]["frames"].items() if fr["visible"]]
        intervals.append([min(fidx), max(fidx)])
    intervals = np.array(intervals)
    return [int(np.min(intervals[:, 0])), int(np.max(intervals[:, 0]))]


def colliding_window(annotations):
    """Not in the release: the span of frames on which any track carries 'Colliding', or None.
    The thesis uses it as the collision onset (Section 6.3.3); the release's training span is
    find_boundary."""
    frames = [fid for t in annotations.values() for fid, fr in t["frames"].items() if fr["attribute"] == "Colliding"]
    return [min(frames), max(frames)] if frames else None


def segment_endpoints_seconds(segment):
    """Start and end of a stage-1 segment in seconds, as extract_frames reads them, or None."""
    ts = sorted(float(k["frame"]) for k in segment.get("keyframes", []))
    if len(ts) < 2:
        return None
    return ts[0], ts[1]


def clip_numbering(json_data):
    """The clip number that extract_frames assigns to every segment of the stage-1 JSON:
    {(video_name, stage1_id): clip_number}, zero based."""
    index, counter = {}, 0
    for video_name, segments in json_data.items():
        segs = []
        for seg in segments:
            se = segment_endpoints_seconds(seg)
            if se is None:
                continue
            segs.append((se[0], se[1], seg.get("id")))
        segs.sort(key=lambda t: t[0])
        for s0, e0, sid in segs:
            index[(video_name, sid)] = counter
            counter += 1
    return index


def extract_frames(json_data, input_dir, output_dir):
    """utils.extract_frames of the release (OpenCV decode of every native frame)."""
    import cv2
    segment_counter = 0
    last_counter = {}
    if not os.path.exists(output_dir):
        raise ValueError("{} does not exist!".format(output_dir))
    for video_name in json_data:
        segments = json_data[video_name]
        cap = cv2.VideoCapture("{}/{}".format(input_dir, video_name))
        if not cap.isOpened():
            print("error: cannot open {}".format(video_name))
            continue
        fps = cap.get(cv2.CAP_PROP_FPS)
        endpoints = []
        for segment in segments:
            timestamps = sorted(int(t["frame"] * fps) for t in segment["keyframes"])
            if len(timestamps) < 2:
                print("error: illegal segment, fewer than two endpoints")
                continue
            endpoints.append([timestamps[0], timestamps[1] + 1])
        endpoints.sort(key=lambda k: k[0])
        if len(endpoints) == 0:
            cap.release()
            continue
        counter = 0
        segment_counter_start = segment_counter
        for i in range(len(endpoints)):
            os.makedirs("{}/{:06d}".format(output_dir, segment_counter))
            last_counter[segment_counter] = 0
            segment_counter += 1
        while True:
            if counter > np.max(np.array(endpoints)[:, 1]):
                break
            segs = [i + segment_counter_start for i, seg in enumerate(endpoints) if counter in range(seg[0], seg[1])]
            if len(segs) == 0:
                cap.grab()
            else:
                status, img = cap.read()
                if not status or img is None:
                    break
                for i in segs:
                    cv2.imwrite("{}/{:06d}/{}.jpg".format(output_dir, i, last_counter[i]), img)
                    last_counter[i] += 1
            counter += 1
        cap.release()
    return last_counter


def cross_validation_folds(idx, n_folds=4, test_size=0.33, random_state=42):
    """utils.cross_validation_folds of the release: clip-level train/test draws."""
    from sklearn.model_selection import train_test_split
    assert n_folds > 0
    folds = []
    x = np.zeros_like(idx)
    for _ in range(n_folds):
        x_train, x_test, y_train, y_test = train_test_split(x, idx, test_size=test_size, random_state=random_state)
        folds.append([y_train, y_test])
    return folds


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "sample_vatic.txt"
    ann = read_vatic(path)
    print("tracks:", sorted(ann.keys()))
    print("labels:", {tid: t["label"] for tid, t in ann.items()})
    print("separator boundary (training span of the release):", find_boundary(ann))
    print("colliding window (onset used by the thesis):", colliding_window(ann))
    print("unusual tracks:", [tid for tid, t in ann.items() if is_unusual_track(t)])
