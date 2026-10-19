#!/usr/bin/env python3
"""
make_image_2_2.py -- Image 2.2 of the thesis: the same source frames in a training clip and a test
clip of the IEEE DataPort Traffic Accident Video dataset as distributed.

    python make_image_2_2.py --train <train clip.mp4> --test <test clip.mp4> --out image_2_2.png

The two clips are cut from the same source video. Their file names carry the source frame range
(..._9616_10436.mp4 and ..._10287_10578-splitted-0-5.mp4), so frame k of the test clip is source frame
10287 + k and frame 671 + k of the training clip. The script checks that alignment on the pixels before
drawing anything: the mean squared difference at the stated offset must be far below the difference one
frame off.

Number plates and people are blurred. The boxes below were placed by hand on the 1280x720 frames and
checked on enlarged crops. They apply to the training and the test frame alike, because the two are the
same image.
"""
import argparse
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

TRAIN_F0, TEST_F0 = 9616, 10287          # first source frame of each clip, from the file names
OFFSET = TEST_F0 - TRAIN_F0              # 671
TEST_FRAMES = (60, 108, 144)             # frames of the test clip shown, one per column

# (x0, y0, x1, y1) on the 1280x720 frame: number plates first, then people
BLUR = {
    60: [(822, 434, 910, 456), (1056, 430, 1110, 452), (1182, 438, 1236, 458), (748, 378, 788, 402),
         (432, 364, 474, 438), (145, 362, 182, 420)],
    108: [(740, 468, 816, 498), (1188, 484, 1270, 516), (442, 406, 478, 430),
          (270, 364, 322, 458), (36, 362, 66, 412), (1060, 350, 1092, 416), (1010, 380, 1040, 420),
          (958, 352, 1002, 392), (1246, 350, 1276, 382)],
    144: [(664, 524, 776, 560), (302, 418, 342, 444), (50, 398, 82, 432),
          (158, 362, 216, 472), (1002, 376, 1036, 424), (1066, 354, 1100, 424)],
}

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def read_frames(path):
    cap, out = cv2.VideoCapture(path), []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        out.append(f)
    return out


def check_alignment(train, test):
    """mean squared difference (grey levels) at the stated offset and one frame either side"""
    def msd(off):
        ks = [k for k in range(len(test)) if 0 <= off + k < len(train)]
        g = lambda f: cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)
        return float(np.mean([np.mean((g(train[off + k]) - g(test[k])) ** 2) for k in ks])), len(ks)
    at, n = msd(OFFSET)
    near = min(msd(OFFSET - 1)[0], msd(OFFSET + 1)[0])
    assert at * 5 < near, f"no frame-exact match at offset {OFFSET}: {at:.1f} against {near:.1f}"
    return at, near, n


def blur_boxes(frame, boxes, pad=4):
    out = frame.copy()
    h, w = frame.shape[:2]
    for x0, y0, x1, y1 in boxes:
        x0, y0, x1, y1 = max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + pad), min(h, y1 + pad)
        roi = out[y0:y1, x0:x1]
        small = cv2.resize(roi, (max(1, (x1 - x0) // 12), max(1, (y1 - y0) // 12)), interpolation=cv2.INTER_AREA)
        coarse = cv2.resize(small, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
        out[y0:y1, x0:x1] = cv2.GaussianBlur(coarse, (0, 0), sigmaX=4)
    return out


def compose(rows, col_titles, labels, tile_w=760, gap=18, font_px=40, out="image_2_2.png"):
    tile_h = int(round(tile_w * 9 / 16))
    font = ImageFont.truetype(FONT, font_px)
    head_h, lab_h = int(font_px * 1.7), int(font_px * 1.6)
    W = 3 * tile_w + 2 * gap
    H = head_h + 2 * (tile_h + lab_h) + gap
    canvas = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(canvas)
    for c, t in enumerate(col_titles):
        x = c * (tile_w + gap)
        tw = d.textlength(t, font=font)
        d.text((x + (tile_w - tw) / 2, int(font_px * 0.25)), t, fill=(30, 30, 30), font=font)
    for r, row in enumerate(rows):
        y = head_h + r * (tile_h + lab_h + gap)
        for c, frame in enumerate(row):
            x = c * (tile_w + gap)
            rgb = cv2.cvtColor(cv2.resize(frame, (tile_w, tile_h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
            canvas.paste(Image.fromarray(rgb), (x, y))
            d.text((x, y + tile_h + int(font_px * 0.2)), labels[r][c], fill=(30, 30, 30), font=font)
    canvas.save(out, dpi=(400, 400))
    return canvas.size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--out", default="image_2_2.png")
    ap.add_argument("--check_dir", default="", help="write the blurred full frames here for inspection")
    a = ap.parse_args()
    train, test = read_frames(a.train), read_frames(a.test)
    at, near, n = check_alignment(train, test)
    print(f"frames: train {len(train)}, test {len(test)}; {n} test frames are training frames "
          f"(mean squared difference {at:.1f} at offset {OFFSET}, {near:.1f} one frame off)")
    top, bottom = [], []
    for k in TEST_FRAMES:
        top.append(blur_boxes(train[OFFSET + k], BLUR[k]))
        bottom.append(blur_boxes(test[k], BLUR[k]))
        if a.check_dir:
            cv2.imwrite(f"{a.check_dir}/blurred_test_{k:03d}.png", bottom[-1])
            cv2.imwrite(f"{a.check_dir}/blurred_train_{OFFSET + k:03d}.png", top[-1])
    titles = [f"Source frame {TEST_F0 + k:,}" for k in TEST_FRAMES]
    labels = [[f"Training clip, frame {OFFSET + k}" for k in TEST_FRAMES],
              [f"Test clip, frame {k}" for k in TEST_FRAMES]]
    print("image", compose([top, bottom], titles, labels, out=a.out), "->", a.out)


if __name__ == "__main__":
    main()
