from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
IMG_OK = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

SRC_X = ROOT / "inputs"
SRC_Y = ROOT / "outputs"
WEIGHTS = ROOT / "weights.json"


@dataclass
class Fixed:
    blur_k: int
    morph_k: int
    h_low: int
    s_low: int
    v_low: int
    h_high: int
    s_high: int
    v_high: int


def _mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _files(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    return [p for p in sorted(folder.iterdir()) if p.is_file() and p.suffix.lower() in IMG_OK]


def _odd_clip(x: int, lo: int, hi: int) -> int:
    x = int(np.clip(int(x), lo, hi))
    if x % 2 == 0:
        x += 1
    return int(min(x, hi))


def _hsv_wrap(hsv: np.ndarray, lo: Tuple[int, int, int], hi: Tuple[int, int, int]) -> np.ndarray:
    h0, s0, v0 = lo
    h1, s1, v1 = hi
    if h0 <= h1:
        return cv2.inRange(hsv, (h0, s0, v0), (h1, s1, v1))
    a = cv2.inRange(hsv, (h0, s0, v0), (179, s1, v1))
    b = cv2.inRange(hsv, (0, s0, v0), (h1, s1, v1))
    return cv2.bitwise_or(a, b)


def _prep_hsv(bgr: np.ndarray, blur_k: int) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    k = _odd_clip(blur_k, 3, 21)
    return cv2.GaussianBlur(hsv, (k, k), 0)


def _segment(hsv_blur: np.ndarray, f: Fixed) -> np.ndarray:
    hL, sL, vL = f.h_low, f.s_low, f.v_low
    hH, sH, vH = f.h_high, f.s_high, f.v_high

    if sL > sH:
        sL, sH = sH, sL
    if vL > vH:
        vL, vH = vH, vL

    mk = _odd_clip(f.morph_k, 3, 31)
    ker = np.ones((mk, mk), np.uint8)

    m255 = _hsv_wrap(hsv_blur, (hL, sL, vL), (hH, sH, vH))
    m255 = cv2.morphologyEx(m255, cv2.MORPH_OPEN, ker, iterations=1)
    m255 = cv2.morphologyEx(m255, cv2.MORPH_CLOSE, ker, iterations=1)
    return (m255 > 0).astype(np.uint8)


def _load_fixed() -> Fixed:
    if not WEIGHTS.exists():
        raise RuntimeError(f"weights.json not found: {WEIGHTS} (run train.py first)")
    d = json.loads(WEIGHTS.read_text(encoding="utf-8"))
    return Fixed(
        blur_k=int(d["blur_ksize"]),
        morph_k=int(d["morph_ksize"]),
        h_low=int(d["h_low"]),
        s_low=int(d["s_low"]),
        v_low=int(d["v_low"]),
        h_high=int(d["h_high"]),
        s_high=int(d["s_high"]),
        v_high=int(d["v_high"]),
    )


def main() -> None:
    f = _load_fixed()
    _mkdir(SRC_Y)

    imgs = _files(SRC_X)
    if not imgs:
        raise RuntimeError(f"no input images found in {SRC_X}")

    for ip in imgs:
        bgr = cv2.imread(str(ip), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        hsv = _prep_hsv(bgr, f.blur_k)
        m01 = _segment(hsv, f)
        cv2.imwrite(str(SRC_Y / f"{ip.stem}.png"), (m01 * 255).astype(np.uint8))

    print(f"Done. Masks saved in: {SRC_Y}")


if __name__ == "__main__":
    main()
