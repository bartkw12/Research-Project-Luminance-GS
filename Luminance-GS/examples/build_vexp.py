"""Build the varying-exposure LOM set (milestone M1, the test set for H2).

For every training view k of a LOM scene, draw a known exposure change u_k in stops and write
    I'_k = clip(round(2^{u_k} * I_k), 0, 255)          (8-bit, per pixel and channel)
Test views, high/ (the normal-light ground truth), the COLMAP model and the train/test split are
copied unchanged, so the new scene folder has exactly the LOM layout and the official trainer reads
it with --exp_name low and only --data_dir changed.

    python build_vexp.py --dry_run                     # print the gains, write nothing
    python build_vexp.py                               # build all 5 scenes into data/LOM_vexp/
    python build_vexp.py --verify                      # also load each scene with the trainer's own
                                                       # Parser/Dataset and run the checks (needs the
                                                       # luminancegs env: torch, pycolmap)

Run from Luminance-GS/examples. Reads data/LOM_full/ only; never writes there.

Outputs
  data/LOM_vexp/<scene>/{low,high,colmap_sparse,transforms_*.json,exposure_gains.json}   (git-ignored)
  results/vexp/dataset/gains.csv, gains.json            per-view u_k and checks (commit these)
  results/vexp/dataset/oracle.csv, oracle.json          --verify only: how well the histogram targets
                                                        alone recover u_k (commit these)
  results/vexp/dataset/preview_<scene>.jpg              originals (top) vs gained views (bottom), display gain x4

Choices (see Implementation_guides/M1_vexp_plan.md for the reasons)
  * u_k ~ uniform on [-1, 1] stops, stratified: N evenly spaced values in random order, so the mean
    is exactly 0 and the range is covered evenly (--sampling iid gives independent draws instead).
  * One fixed dataset seed for all scenes (per-scene stream from crc32(scene)), independent of the
    training seeds. Every training seed sees the same set.
  * Gained views are saved as JPEG, quality 100, no chroma subsampling, under their original names,
    because the loader matches files to the COLMAP image names (e.g. 10.JPG). The re-encoding error
    is measured and saved per view (reencode_rms_levels, reencode_psnr_db).
  * Rebuilding or re-verifying some scenes (--scenes ... --overwrite) merges them into the manifest;
    the other scenes' rows are kept (refused if they were built with another seed, range or sampling).
  * The exposure initialisation reported by --verify is log2(m_bar / m_k) with m_bar the geometric mean,
    so it has mean 0 like Rung 2's s_k.
"""

import argparse
import csv
import hashlib
import io
import json
import shutil
import zlib
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

EXAMPLES = Path(__file__).resolve().parent
DATA = EXAMPLES.parent / "data"
SRC_DEFAULT = DATA / "LOM_full"
DST_DEFAULT = DATA / "LOM_vexp"
MANIFEST_DEFAULT = EXAMPLES.parent / "results" / "vexp" / "dataset"

ALL_SCENES = ["buu", "chair", "sofa", "bike", "shrub"]
COPY_DIRS = ["high", "colmap_sparse"]
COPY_FILES = ["transforms_train.json", "transforms_test.json", "transforms_val.json"]
JPEG_KW = {"quality": 100, "subsampling": 0, "optimize": False}
DEFAULT_SEED = 701
PREVIEW_DISPLAY_GAIN = 4.0


# ------------------------------------------------------------------ gains
def train_view_names(scene_dir: Path):
    """Training image names in transforms_train.json order (= trainset index = image_id)."""
    frames = json.loads((scene_dir / "transforms_train.json").read_text())["frames"]
    return [Path(f["file_path"]).name for f in frames]


def test_view_names(scene_dir: Path):
    frames = json.loads((scene_dir / "transforms_test.json").read_text())["frames"]
    return [Path(f["file_path"]).name for f in frames]


def draw_gains(scene: str, n: int, seed: int, u_max: float, sampling: str) -> np.ndarray:
    rng = np.random.default_rng([seed, zlib.crc32(scene.encode())])
    if sampling == "stratified":
        grid = -u_max + u_max * (2 * np.arange(n) + 1) / n  # evenly spaced, mean exactly 0
        return grid[rng.permutation(n)]
    if sampling == "iid":
        return rng.uniform(-u_max, u_max, n)
    raise ValueError(sampling)


# ------------------------------------------------------------------ images
def apply_gain(img: np.ndarray, u: float):
    """8-bit RGB -> 8-bit RGB after a gain of 2^u, rounded and clipped; also the clipped fraction."""
    scaled = img.astype(np.float64) * (2.0 ** u)
    clip_frac = float((scaled > 255.0).mean())
    return np.clip(np.rint(scaled), 0, 255).astype(np.uint8), clip_frac


def jpeg_bytes(img: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", **JPEG_KW)
    return buf.getvalue()


def read_rgb(path: Path) -> np.ndarray:
    # same decoder the loader uses (imageio -> Pillow); [..., :3] as in datasets/colmap.py
    return np.asarray(Image.open(path))[..., :3]


def psnr_levels(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(((a.astype(np.float64) - b.astype(np.float64)) ** 2).mean())
    return float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


# ------------------------------------------------------------------ build
def build_scene(scene: str, src: Path, dst: Path, seed: int, u_max: float, sampling: str,
                dry_run: bool, overwrite: bool, preview_dir: Path = None):
    s_dir, d_dir = src / scene, dst / scene
    names = train_view_names(s_dir)
    tests = set(test_view_names(s_dir))
    bad = [n_ for n_ in names if Path(n_).suffix.lower() not in (".jpg", ".jpeg")]
    if bad:
        raise SystemExit(f"[build_vexp] {scene}: expected JPEG training images, got {bad[:3]}")
    assert not tests & set(names), f"{scene}: train and test views overlap"
    u = draw_gains(scene, len(names), seed, u_max, sampling)

    if dry_run:
        print(f"[build_vexp] {scene}: {len(names)} train views, u in [{u.min():+.3f}, {u.max():+.3f}], "
              f"mean {u.mean():+.4f}, sd {u.std(ddof=1):.3f}")
        for n_, u_ in zip(names, u):
            print(f"    {n_:>10s}  u = {u_:+.3f}  gain = {2 ** u_:.3f}")
        return []

    if d_dir.exists():
        if not overwrite:
            raise SystemExit(f"[build_vexp] {d_dir} exists; pass --overwrite to rebuild it")
        shutil.rmtree(d_dir)
    (d_dir / "low").mkdir(parents=True)
    for d in COPY_DIRS:
        shutil.copytree(s_dir / d, d_dir / d)
    for f in COPY_FILES:
        if (s_dir / f).exists():
            shutil.copy2(s_dir / f, d_dir / f)

    # every non-training file in low/ (the test views) is copied byte for byte: the loader needs the
    # files to exist, but validation reads high/, so they never reach training or scoring
    train_set = set(names)
    for p in sorted((s_dir / "low").iterdir()):
        if p.is_file() and not p.name.startswith(".") and p.name not in train_set:
            shutil.copy2(p, d_dir / "low" / p.name)

    rows, previews = [], []
    for k, (name, u_k) in enumerate(zip(names, u)):
        img = read_rgb(s_dir / "low" / name)
        gained, clip_frac = apply_gain(img, float(u_k))
        data = jpeg_bytes(gained)
        (d_dir / "low" / name).write_bytes(data)
        back = read_rgb(d_dir / "low" / name)
        m_in, m_out = float(img.mean()) / 255.0, float(back.mean()) / 255.0
        rows.append({
            "scene": scene, "view": name, "train_index": k,
            "u": float(u_k), "gain": float(2 ** u_k),
            "mean_in": m_in, "mean_out": m_out,
            "log2_ratio": float(np.log2(m_out / m_in)),
            "max_in": int(img.max()), "max_out": int(back.max()),
            "clip_frac": clip_frac,
            "reencode_rms_levels": float(np.sqrt(((back.astype(float) - gained) ** 2).mean())),
            "reencode_psnr_db": psnr_levels(back, gained),
        })
        previews.append((float(u_k), img, back))

    meta = {"scene": scene, "seed": seed, "u_max": u_max, "sampling": sampling, "jpeg": JPEG_KW,
            "source": str(s_dir), "n_train": len(names), "n_test": len(tests),
            "created": f"{datetime.now():%Y-%m-%d %H:%M:%S}", "script_sha256": script_hash()}
    (d_dir / "exposure_gains.json").write_text(json.dumps({"meta": meta, "views": rows}, indent=1))
    if preview_dir is not None:
        save_preview(previews, preview_dir / f"preview_{scene}.jpg")

    ur = np.array([r["u"] for r in rows])
    err = np.array([r["log2_ratio"] for r in rows]) - ur
    print(f"[build_vexp] {scene}: {len(rows)} train views gained, {len(tests)} test views copied; "
          f"u in [{ur.min():+.3f}, {ur.max():+.3f}]; |log2 ratio - u| max {np.abs(err).max():.3f}; "
          f"max clip {max(r['clip_frac'] for r in rows):.2e}; "
          f"re-encode PSNR min {min(r['reencode_psnr_db'] for r in rows):.1f} dB")
    return rows


def save_preview(previews, path: Path, n_show: int = 6, width: int = 250):
    """Top: original dark views; bottom: the same views after the gain. Same display gain for all."""
    previews = sorted(previews, key=lambda t: t[0])
    idx = np.unique(np.linspace(0, len(previews) - 1, min(n_show, len(previews))).round().astype(int))
    tiles = []
    for i in idx:
        u_k, a, b = previews[i]
        col = []
        for im in (a, b):
            disp = np.clip(im.astype(np.float64) * PREVIEW_DISPLAY_GAIN, 0, 255).astype(np.uint8)
            t = Image.fromarray(disp)
            t = t.resize((width, int(round(t.height * width / t.width))))
            col.append(t)
        tile = Image.new("RGB", (width, col[0].height * 2 + 18), "white")
        tile.paste(col[0], (0, 0))
        tile.paste(col[1], (0, col[0].height))
        ImageDraw.Draw(tile).text((4, col[0].height * 2 + 3), f"u = {u_k:+.2f} stops", fill="black")
        tiles.append(tile)
    sheet = Image.new("RGB", (sum(t.width for t in tiles) + 4 * (len(tiles) - 1), tiles[0].height), "white")
    x = 0
    for t in tiles:
        sheet.paste(t, (x, 0))
        x += t.width + 4
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=90)


def script_hash() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]


# ------------------------------------------------------------- exposure fit
def fit_exposure_form(curves: np.ndarray, weights: np.ndarray, iters: int = 10):
    """Fit curves[k] ~ F(min(1, 2^{s_k} x)) with one shared monotone F and one s_k per view, mean s_k = 0.
    Alternates a per-view grid search over s_k (wide on the first pass) with a weighted monotone refit of F.
    Same method as the proposal's B4 sign check, with a wider first search so 2-stop spreads are reachable.
    Returns s, the spread of the curves around their mean and the residual after the fit (both x255, weighted)."""
    X = curves.astype(np.float64)
    V, nb = X.shape
    x = np.arange(nb) / (nb - 1)
    W = weights.astype(np.float64)
    W = W / W.sum(1, keepdims=True)
    F = X.mean(0)
    s = np.zeros(V)
    for it in range(iters):
        half, npts = (3.0, 6001) if it == 0 else (0.5, 2001)
        for k in range(V):
            grid = np.linspace(s[k] - half, s[k] + half, npts)
            P = np.interp(np.minimum(1.0, 2.0 ** grid[:, None] * x[None]), x, F)
            s[k] = grid[np.argmin(((P - X[k]) ** 2 * W[k]).sum(1))]
        s -= s.mean()  # gauge: only relative exposure is identifiable
        num, den = np.zeros(nb), np.zeros(nb)
        for k in range(V):
            pos = np.minimum(1.0, 2.0 ** s[k] * x) * (nb - 1)
            i0 = np.clip(np.floor(pos).astype(int), 0, nb - 2)
            t = pos - i0
            np.add.at(num, i0, (1 - t) * W[k] * X[k])
            np.add.at(den, i0, (1 - t) * W[k])
            np.add.at(num, i0 + 1, t * W[k] * X[k])
            np.add.at(den, i0 + 1, t * W[k])
        g = den > 1e-9
        F = np.maximum.accumulate(np.interp(x, x[g], num[g] / den[g]))
    pred = np.stack([np.interp(np.minimum(1.0, 2.0 ** s[k] * x), x, F) for k in range(V)])
    spread = float(np.sqrt(((X - X.mean(0)) ** 2 * W).sum() / V) * 255)
    resid = float(np.sqrt(((X - pred) ** 2 * W).sum() / V) * 255)
    return s, spread, resid


def slope_r(y: np.ndarray, u: np.ndarray):
    """Least-squares slope of y on u and Pearson r."""
    return float(np.polyfit(u, y, 1)[0]), float(np.corrcoef(u, y)[0, 1])


# ------------------------------------------------------------------ verify
def verify_scene(scene: str, src: Path, dst: Path):
    """Load both versions of the scene with the trainer's own Parser/Dataset and check the new set.
    Then measure how well the histogram-equalisation targets alone recover u_k (the 'oracle')."""
    import contextlib
    import io as _io
    import sys

    import torch
    sys.path.insert(0, str(EXAMPLES))
    from datasets.colmap import Dataset, Parser
    from losses import HistogramPriorLoss

    def load(root):
        with contextlib.redirect_stdout(_io.StringIO()):
            p = Parser(data_dir=str(root / scene), exp_name="low", method="", factor=1, normalize=True, test_every=8)
            return p, Dataset(p, split="train"), Dataset(p, split="val")

    p0, tr0, va0 = load(src)
    p1, tr1, va1 = load(dst)
    gains = json.loads((dst / scene / "exposure_gains.json").read_text())["views"]
    u_by_name = {g["view"]: g["u"] for g in gains}

    assert len(tr0) == len(tr1) and len(va0) == len(va1), "split sizes differ"
    assert p0.image_names == p1.image_names, "image order differs"
    assert np.allclose(p0.camtoworlds, p1.camtoworlds), "cameras differ"
    for i in range(len(va0)):
        assert torch.equal(va0[i]["image"], va1[i]["image"]), f"validation view {i} differs"

    hist_loss = HistogramPriorLoss()
    names, u, m0, m1, cdfs, hists, cdfs0, hists0 = [], [], [], [], [], [], [], []
    for i in range(len(tr1)):
        name = p1.image_names[int(tr1.indices[i])]
        a = tr0[i]["image"][None] / 255.0  # [1, H, W, 3], undistorted exactly as in training
        b = tr1[i]["image"][None] / 255.0
        names.append(name)
        u.append(u_by_name[name])
        m0.append(float(a.mean()))
        m1.append(float(b.mean()))
        for img, c_acc, h_acc in ((b, cdfs, hists), (a, cdfs0, hists0)):
            c_acc.append(hist_loss.compute_histogram_equalization(img)[0].numpy())
            grey = torch.mean(torch.nn.functional.interpolate(img.permute(0, 3, 1, 2), scale_factor=0.25), dim=1)
            h_acc.append(torch.histc(grey.flatten(), bins=255, min=0.0, max=1.0).numpy())
    u, m0, m1 = map(np.array, (u, m0, m1))

    # the trainer must see the gained images: brightness ratio new/original should equal 2^u up to rounding
    ratio = np.log2(m1 / m0)
    ratio_err = ratio - u
    assert np.corrcoef(ratio, u)[0, 1] > 0.99 and np.abs(ratio_err).max() < 0.1, \
        f"{scene}: the loaded images do not carry the gains (max error {np.abs(ratio_err).max():.3f} stops)"
    if np.abs(ratio_err).max() > 0.05:
        print(f"[verify] {scene}: note, brightness ratio differs from 2^u by up to {np.abs(ratio_err).max():.3f} stops "
              f"(8-bit rounding of very dark views); check gains.csv")

    # the planned s_k initialisation log2(m_bar / m_k) with m_bar the GEOMETRIC mean over views, so the
    # init has mean 0, the same gauge as Rung 2's s_k (an arithmetic mean shifts it by about +0.12 stops)
    init1 = np.log2(m1).mean() - np.log2(m1)
    init0 = np.log2(m0).mean() - np.log2(m0)  # same on the original LOM views (natural spread)
    s_or, spread, resid = fit_exposure_form(np.stack(cdfs), np.stack(hists))
    s_or0, spread0, resid0 = fit_exposure_form(np.stack(cdfs0), np.stack(hists0))
    out = {"scene": scene, "n_train": len(u)}
    out["init_slope"], out["init_r"] = slope_r(init1, u)
    out["oracle_slope"], out["oracle_r"] = slope_r(s_or, u)
    out["oracle_explained"] = 1 - resid ** 2 / spread ** 2
    out["natural_sd_stops_lom"] = float(np.std(init0, ddof=1))
    out["natural_sd_oracle_lom"] = float(np.std(s_or0, ddof=1))
    out["u_sd"] = float(np.std(u, ddof=1))
    out["max_ratio_err_stops"] = float(np.abs(ratio_err).max())
    out["view_names"] = names
    out["s_oracle"] = s_or.tolist()
    out["s_init"] = init1.tolist()
    out["u"] = u.tolist()
    print(f"[verify] {scene}: loader OK, validation images identical, brightness ratio within "
          f"{out['max_ratio_err_stops']:.3f} stops of 2^u. Exposure recovery vs u: init slope "
          f"{out['init_slope']:+.2f} (r {out['init_r']:+.3f}), histogram oracle slope {out['oracle_slope']:+.2f} "
          f"(r {out['oracle_r']:+.3f}). Natural LOM spread sd {out['natural_sd_stops_lom']:.2f} stops; u sd {out['u_sd']:.2f}.")
    return out


# ------------------------------------------------------------------ main
def merge_rows(path_json: Path, rows, scenes, meta):
    """New rows plus the rows already in path_json for scenes not processed now, in ALL_SCENES order,
    so rebuilding or re-verifying some scenes does not drop the others from the manifest."""
    if not path_json.exists():
        return rows
    old = json.loads(path_json.read_text())
    keep = [r for r in old["rows"] if r["scene"] not in set(scenes)]
    if keep:
        for k in ("seed", "u_max", "sampling"):
            if old["meta"].get(k) != meta[k]:
                raise SystemExit(f"[build_vexp] {path_json} holds other scenes built with {k}={old['meta'].get(k)!r}, "
                                 f"now {meta[k]!r}; rebuild every scene or use another --manifest_dir")
    order = {s: i for i, s in enumerate(ALL_SCENES)}
    return sorted(keep + rows, key=lambda r: order.get(r["scene"], len(order)))  # stable: view order kept


def write_table(rows, path_csv: Path, path_json: Path, meta=None):
    path_csv.parent.mkdir(parents=True, exist_ok=True)
    flat = [{k: v for k, v in r.items() if not isinstance(v, list)} for r in rows]
    with open(path_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(flat[0].keys()))
        w.writeheader()
        w.writerows(flat)
    path_json.write_text(json.dumps({"meta": meta or {}, "rows": rows}, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", nargs="+", default=ALL_SCENES)
    ap.add_argument("--src", type=Path, default=SRC_DEFAULT)
    ap.add_argument("--dst", type=Path, default=DST_DEFAULT)
    ap.add_argument("--manifest_dir", type=Path, default=MANIFEST_DEFAULT)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED, help="dataset seed (not a training seed)")
    ap.add_argument("--u_max", type=float, default=1.0, help="u_k lies in [-u_max, u_max] stops")
    ap.add_argument("--sampling", choices=["stratified", "iid"], default="stratified")
    ap.add_argument("--overwrite", action="store_true", help="rebuild scenes that already exist in --dst")
    ap.add_argument("--verify", action="store_true", help="load with the trainer's Parser/Dataset and check (needs torch)")
    ap.add_argument("--verify_only", action="store_true", help="skip building; verify an existing --dst")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    src, dst = args.src.resolve(), args.dst.resolve()
    if src == dst or src in dst.parents:
        raise SystemExit("--dst must not be inside --src")
    # datasets/colmap.py finds a validation image by replacing every 'low' in its path with 'high'
    for scene in args.scenes:
        if "low" in str(dst / scene):
            raise SystemExit(f"--dst/<scene> path must not contain 'low' (the loader's path.replace would break): "
                             f"{dst / scene}")

    meta = {"seed": args.seed, "u_max": args.u_max, "sampling": args.sampling, "jpeg": JPEG_KW,
            "src": str(src), "dst": str(dst), "script_sha256": script_hash(),
            "created": f"{datetime.now():%Y-%m-%d %H:%M:%S}"}
    # refuse before writing anything if the manifest holds other scenes built with other settings
    if not args.dry_run and not args.verify_only:
        merge_rows(args.manifest_dir / "gains.json", [], args.scenes, meta)
    if args.verify or args.verify_only:
        merge_rows(args.manifest_dir / "oracle.json", [], args.scenes, meta)

    if not args.verify_only:
        rows = []
        for scene in args.scenes:
            rows += build_scene(scene, src, dst, args.seed, args.u_max, args.sampling,
                                args.dry_run, args.overwrite, preview_dir=args.manifest_dir)
        if args.dry_run:
            return
        rows = merge_rows(args.manifest_dir / "gains.json", rows, args.scenes, meta)
        write_table(rows, args.manifest_dir / "gains.csv", args.manifest_dir / "gains.json", meta)
        print(f"[build_vexp] wrote {dst} and {args.manifest_dir / 'gains.csv'}")

    if args.verify or args.verify_only:
        orc = [verify_scene(scene, src, dst) for scene in args.scenes]
        orc = merge_rows(args.manifest_dir / "oracle.json", orc, args.scenes, meta)
        write_table(orc, args.manifest_dir / "oracle.csv", args.manifest_dir / "oracle.json", meta)
        print(f"[build_vexp] wrote {args.manifest_dir / 'oracle.csv'}")


if __name__ == "__main__":
    main()
