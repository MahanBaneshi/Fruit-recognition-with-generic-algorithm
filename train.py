from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
IMG_OK = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
MSK_OK = {".png", ".jpg", ".jpeg"}


def _pick_dir(base: Path, split: str, role: str) -> Path:
    if role == "x":
        names = ("inputs", "images", "Images", "image", "Image")
    elif role == "y":
        names = ("masks", "Masks", "mask", "Mask")
    else:
        names = (role,)
    for n in names:
        p = base / split / n
        if p.exists():
            return p
    return base / split / role


TRAIN_X = _pick_dir(ROOT, "train", "x")
TRAIN_Y = _pick_dir(ROOT, "train", "y")


@dataclass
class Settings:
    n_pop: int = 40
    n_gen: int = 60
    k_tourn: int = 3
    n_elite: int = 2
    p_cross: float = 0.85
    p_mut: float = 0.10
    sigma: float = 10.0
    train_ratio: float = 0.85
    seed: int = 42
    blur_k: int = 5
    morph_k: int = 5
    s_floor: int = 35


CFG = Settings()


def _files_with_ext(folder: Path, ok_ext: set[str]) -> List[Path]:
    if not folder.exists():
        return []
    items: List[Path] = []
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix.lower() in ok_ext:
            items.append(p)
    return items


def _imread_bgr(p: Path) -> np.ndarray:
    im = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if im is None:
        raise RuntimeError(f"cannot read image: {p}")
    return im


def _imread_mask01(p: Path) -> np.ndarray:
    m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise RuntimeError(f"cannot read mask: {p}")
    return (m > 0).astype(np.uint8)


def _mask_candidates(img_path: Path, masks_dir: Path) -> List[Path]:
    stem = img_path.stem
    cand: List[Path] = []
    for ext in MSK_OK:
        cand.extend(sorted(masks_dir.glob(f"{stem}_*{ext}")))
    for ext in MSK_OK:
        p = masks_dir / f"{stem}{ext}"
        if p.exists():
            cand.append(p)
    for ext in MSK_OK:
        p = masks_dir / f"gt{stem}{ext}"
        if p.exists():
            cand.append(p)

    uniq: List[Path] = []
    seen = set()
    for p in cand:
        if p.exists() and p not in seen:
            uniq.append(p)
            seen.add(p)
    return uniq


def _load_gt_union(img_path: Path, masks_dir: Path) -> np.ndarray:
    ms = _mask_candidates(img_path, masks_dir)
    if not ms:
        raise RuntimeError(f"mask missing for {img_path.name} in {masks_dir}")
    out: Optional[np.ndarray] = None
    for mp in ms:
        m01 = _imread_mask01(mp)
        out = m01 if out is None else np.maximum(out, m01)
    return out.astype(np.uint8)


def _odd_clip(x: int, lo: int, hi: int) -> int:
    x = int(np.clip(int(x), lo, hi))
    if x % 2 == 0:
        x += 1
    return int(min(x, hi))


def _hsv_range_wrap(hsv: np.ndarray, lo: Tuple[int, int, int], hi: Tuple[int, int, int]) -> np.ndarray:
    h0, s0, v0 = lo
    h1, s1, v1 = hi
    if h0 <= h1:
        return cv2.inRange(hsv, (h0, s0, v0), (h1, s1, v1))
    m1 = cv2.inRange(hsv, (h0, s0, v0), (179, s1, v1))
    m2 = cv2.inRange(hsv, (0, s0, v0), (h1, s1, v1))
    return cv2.bitwise_or(m1, m2)


def _prep_hsv(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    k = _odd_clip(CFG.blur_k, 3, 21)
    return cv2.GaussianBlur(hsv, (k, k), 0)


def _mask_from_genome(hsv_blur: np.ndarray, g: np.ndarray, ker: np.ndarray) -> np.ndarray:
    hL, sL, vL, hH, sH, vH = [int(x) for x in g]
    if sL > sH:
        sL, sH = sH, sL
    if vL > vH:
        vL, vH = vH, vL
    m255 = _hsv_range_wrap(hsv_blur, (hL, sL, vL), (hH, sH, vH))
    m255 = cv2.morphologyEx(m255, cv2.MORPH_OPEN, ker, iterations=1)
    m255 = cv2.morphologyEx(m255, cv2.MORPH_CLOSE, ker, iterations=1)
    return (m255 > 0).astype(np.uint8)


def _iou01(a01: np.ndarray, b01: np.ndarray) -> float:
    a = a01.astype(bool)
    b = b01.astype(bool)
    inter = float(np.logical_and(a, b).sum())
    uni = float(np.logical_or(a, b).sum())
    return 0.0 if uni <= 0 else inter / uni


def _min_iou_fitness(g: np.ndarray, hsvs: Sequence[np.ndarray], gts: Sequence[np.ndarray], ker: np.ndarray) -> float:
    worst = 1.0
    for hsv, gt in zip(hsvs, gts):
        pred = _mask_from_genome(hsv, g, ker)
        s = _iou01(pred, gt)
        if s < worst:
            worst = s
            if worst < 0.02:
                break
    return float(worst)


def _fix_genome(g: np.ndarray) -> np.ndarray:
    g = g.astype(np.int32).copy()
    g[0] = int(np.clip(g[0], 0, 179))
    g[3] = int(np.clip(g[3], 0, 179))
    g[1] = int(np.clip(g[1], CFG.s_floor, 255))
    g[4] = int(np.clip(g[4], 0, 255))
    if g[1] > g[4]:
        g[1], g[4] = g[4], g[1]
    g[1] = int(np.clip(g[1], CFG.s_floor, 255))
    g[2] = int(np.clip(g[2], 0, 255))
    g[5] = int(np.clip(g[5], 0, 255))
    if g[2] > g[5]:
        g[2], g[5] = g[5], g[2]
    return g


def _spawn(rng: np.random.Generator) -> np.ndarray:
    if rng.random() < 0.5:
        hL = int(rng.integers(150, 180))
        hH = int(rng.integers(0, 25))
    else:
        hL = int(rng.integers(0, 20))
        hH = int(rng.integers(0, 35))
    s1 = int(rng.integers(CFG.s_floor, 256))
    s2 = int(rng.integers(CFG.s_floor, 256))
    v1 = int(rng.integers(0, 256))
    v2 = int(rng.integers(0, 256))
    g = np.array([hL, min(s1, s2), min(v1, v2), hH, max(s1, s2), max(v1, v2)], dtype=np.int32)
    return _fix_genome(g)


def _select_tournament(rng: np.random.Generator, pop: List[np.ndarray], fit: List[float], k: int) -> np.ndarray:
    idx = rng.integers(0, len(pop), size=k)
    best = int(idx[0])
    for j in idx[1:]:
        j = int(j)
        if fit[j] > fit[best]:
            best = j
    return pop[best].copy()


def _one_point_cross(rng: np.random.Generator, a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    cut = int(rng.integers(1, 6))
    c1, c2 = a.copy(), b.copy()
    c1[cut:], c2[cut:] = b[cut:], a[cut:]
    return _fix_genome(c1), _fix_genome(c2)


def _jitter(rng: np.random.Generator, g: np.ndarray) -> np.ndarray:
    out = g.copy()
    for i in range(6):
        if rng.random() < CFG.p_mut:
            out[i] = int(out[i] + rng.normal(0.0, CFG.sigma))
    return _fix_genome(out)


def _evolve(hsv_tr: List[np.ndarray], gt_tr: List[np.ndarray], hsv_va: List[np.ndarray], gt_va: List[np.ndarray]) -> np.ndarray:
    rng = np.random.default_rng(CFG.seed)
    mk = _odd_clip(CFG.morph_k, 3, 31)
    ker = np.ones((mk, mk), np.uint8)

    pop = [_spawn(rng) for _ in range(CFG.n_pop)]
    best = pop[0].copy()
    best_val = -1e18

    for gen in range(CFG.n_gen):
        fit = [_min_iou_fitness(ind, hsv_tr, gt_tr, ker) for ind in pop]
        order = np.argsort(fit)[::-1]
        top = int(order[0])

        if fit[top] > best_val:
            best_val = float(fit[top])
            best = pop[top].copy()

        v = _min_iou_fitness(pop[top], hsv_va, gt_va, ker) if hsv_va else fit[top]
        print(f"Gen {gen+1:02d}/{CFG.n_gen} | train(minIoU)={fit[top]:.4f} | val(minIoU)={v:.4f}")

        new_pop = [pop[int(order[i])].copy() for i in range(min(CFG.n_elite, CFG.n_pop))]
        while len(new_pop) < CFG.n_pop:
            p1 = _select_tournament(rng, pop, fit, CFG.k_tourn)
            p2 = _select_tournament(rng, pop, fit, CFG.k_tourn)

            if rng.random() < CFG.p_cross:
                c1, c2 = _one_point_cross(rng, p1, p2)
            else:
                c1, c2 = p1.copy(), p2.copy()

            new_pop.append(_jitter(rng, c1))
            if len(new_pop) < CFG.n_pop:
                new_pop.append(_jitter(rng, c2))

        pop = new_pop

    return best


def _split_paths(paths: List[Path]) -> Tuple[List[Path], List[Path]]:
    idx = list(range(len(paths)))
    random.Random(CFG.seed).shuffle(idx)
    if len(idx) <= 3:
        a, b = idx, idx
    else:
        cut = int(round(len(idx) * CFG.train_ratio))
        a = idx[:cut]
        b = idx[cut:] or a
    return [paths[i] for i in a], [paths[i] for i in b]


def _cache_arrays(imgs: Iterable[Path]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    hsvs: List[np.ndarray] = []
    gts: List[np.ndarray] = []
    for p in imgs:
        bgr = _imread_bgr(p)
        hsvs.append(_prep_hsv(bgr))
        gts.append(_load_gt_union(p, TRAIN_Y).astype(np.uint8))
    return hsvs, gts


def main() -> None:
    train_imgs = _files_with_ext(TRAIN_X, IMG_OK)
    if not train_imgs:
        raise RuntimeError(f"no training images in {TRAIN_X}")

    tr_paths, va_paths = _split_paths(train_imgs)
    print(f"Train images: {len(tr_paths)} | Val images: {len(va_paths)}")
    print(f"TRAIN_INPUTS={TRAIN_X}")
    print(f"TRAIN_MASKS={TRAIN_Y}")

    hsv_tr, gt_tr = _cache_arrays(tr_paths)
    hsv_va, gt_va = _cache_arrays(va_paths)

    best = _evolve(hsv_tr, gt_tr, hsv_va, gt_va)

    out = {
        "h_low": int(best[0]),
        "s_low": int(best[1]),
        "v_low": int(best[2]),
        "h_high": int(best[3]),
        "s_high": int(best[4]),
        "v_high": int(best[5]),
        "blur_ksize": int(CFG.blur_k),
        "morph_ksize": int(CFG.morph_k),
        "s_low_floor": int(CFG.s_floor),
    }
    (ROOT / "weights.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("Saved weights.json")


if __name__ == "__main__":
    main()
