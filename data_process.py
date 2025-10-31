# make_pairs_concat_seq_secret_tqdm_safe.py
from pathlib import Path
from PIL import Image, ImageOps
import numpy as np
import random, re
from tqdm import tqdm
import cv2  # ← 新增：OpenCV 作为解码回退

# ================== 只改这里 ==================
CONFIG = dict(
    mode="folder",              # "folder" 或 "noise"
    cover_dir=r"E:\Py_projects\Patent\Code\data\val_256",
    secret_dir=r"E:\Py_projects\Patent\Code\data\val_256",
    out_dir=r"E:\Py_projects\Patent\Code\data\test",
    # cover_dir=r"E:\Py_projects\Patent\Code\data\download_cocotrain\covers_mid",
    # secret_dir=r"E:\Py_projects\Patent\Code\data\download_cocotrain\secrets",
    # out_dir=r"E:\Py_projects\Patent\Code\data\im2no",
    count=10,                 # 实际生成数=两文件夹数量和该值的最小值
    size=256,                   # 单图边长；拼接后为 (2*size, size)
    left="secret",              # 左图："cover" 或 "secret"
    avoid_self=True,            # cover 与 secret 同路径时尽量避免自配
    fmt="png",                  # "png" 或 "jpg"
    quality=95,                 # JPG 质量
    subsampling=0,              # JPG 色度抽样：0=4:4:4, 1=4:2:2, 2=4:2:0
    zfill=3,                    # 编号零填充位数：3→001
    seed=0,                     # 随机种子（复现）
    noise_pool=("uniform","gaussian","saltpepper","speckle","poisson"),
    noise_weights=None,
)
# ==============================================

from PIL import Image
try:
    RESAMPLE_LANCZOS = Image.Resampling.LANCZOS
except AttributeError:
    RESAMPLE_LANCZOS = getattr(Image, "LANCZOS", Image.ANTIALIAS)

EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}
NAME_RE = re.compile(r"^(\d+)\.(png|jpg|jpeg)$", re.IGNORECASE)

def list_images(root: Path):
    root = Path(root)
    return sorted([p for p in root.rglob("*") if p.suffix.lower() in EXTS])

def to_rgb_square(im: Image.Image, size: int) -> Image.Image:
    if im.mode == "RGBA":
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im).convert("RGB")
    else:
        im = im.convert("RGB")
    return ImageOps.fit(im, (size, size), method=RESAMPLE_LANCZOS, centering=(0.5, 0.5))

# -------- 安全读取与筛选 --------
def open_image_safe(path: Path) -> Image.Image:
    """
    先用 Pillow 解码并强制 load；若因 iCCP 或其他 PNG/JPEG 异常失败，
    回退用 OpenCV 解码，再转回 PIL.Image('RGB')。
    """
    try:
        im = Image.open(path)
        im.load()  # 触发真实解码，若异常直接抛出
        if im.mode != "RGB":
            im = im.convert("RGB")
        return im
    except Exception as e:
        data = np.fromfile(str(path), dtype=np.uint8)  # 兼容中文路径
        bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if bgr is None:
            raise e
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb, "RGB")

def filter_usable_rgb(paths):
    """仅保留可读且最终为 RGB 的图片。"""
    ok = []
    for p in tqdm(paths, desc="Filter usable RGB", unit="img"):
        try:
            im = open_image_safe(p)
            im.close()
            ok.append(p)
        except Exception:
            continue
    return ok

# -------- 噪声生成（保留原功能）--------
def rand_noise_any(size: int, rng_py: random.Random, rng_np: np.random.Generator,
                   pool, weights) -> Image.Image:
    t = rng_py.choices(pool, weights=weights, k=1)[0]
    if t == "uniform":
        arr = rng_np.integers(0, 256, size=(size, size, 3), dtype=np.uint8)
    elif t == "gaussian":
        sigma = rng_py.uniform(25, 60)
        arr = rng_np.normal(127.5, sigma, size=(size, size, 3)).clip(0, 255).astype(np.uint8)
    elif t == "saltpepper":
        p = rng_py.uniform(0.01, 0.10); s_ratio = rng_py.uniform(0.3, 0.7)
        arr = np.full((size, size, 3), 127, np.uint8)
        m = rng_np.random((size, size))
        salt = m < (p * s_ratio); pepper = m > (1 - p * (1 - s_ratio))
        arr[salt] = 255; arr[pepper] = 0
    elif t == "speckle":
        alpha = rng_py.uniform(0.2, 0.6)
        base = rng_np.integers(60, 196, size=(size, size, 3)).astype(np.float32)
        n = rng_np.normal(0.0, 1.0, size=(size, size, 3)).astype(np.float32)
        arr = (base * (1.0 + alpha * n)).clip(0, 255).astype(np.uint8)
    elif t == "poisson":
        lam = rng_py.uniform(20, 180)
        arr = rng_np.poisson(lam=lam, size=(size, size, 3)).clip(0, 255).astype(np.uint8)
    else:
        arr = rng_np.integers(0, 256, size=(size, size, 3), dtype=np.uint8)
    return Image.fromarray(arr, "RGB")

def next_start_index(out_dir: Path) -> int:
    max_idx = 0
    if out_dir.exists():
        for p in out_dir.iterdir():
            m = NAME_RE.match(p.name)
            if m:
                try:
                    max_idx = max(max_idx, int(m.group(1)))
                except ValueError:
                    pass
    return max_idx + 1 if max_idx >= 1 else 1

def main(C):
    rng = random.Random(C["seed"])
    np_rng = np.random.default_rng(C["seed"])

    # 1) 读取并筛选可用 RGB
    secret_paths_raw = list_images(C["secret_dir"])
    cover_paths_raw  = list_images(C["cover_dir"]) if C["mode"] == "folder" else []

    if not secret_paths_raw:
        raise RuntimeError(f"秘密目录无图片: {C['secret_dir']}")
    if C["mode"] == "folder" and not cover_paths_raw:
        raise RuntimeError(f"载体目录无图片: {C['cover_dir']}")

    secret_paths = filter_usable_rgb(secret_paths_raw)
    cover_paths  = filter_usable_rgb(cover_paths_raw) if C["mode"] == "folder" else []

    if not secret_paths:
        raise RuntimeError(f"秘密目录无可用 RGB 图片: {C['secret_dir']}")
    if C["mode"] == "folder" and not cover_paths:
        raise RuntimeError(f"载体目录无可用 RGB 图片: {C['cover_dir']}")

    out_dir = Path(C["out_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    start_idx = next_start_index(out_dir)

    if C["mode"] == "folder":
        # 2) 构造不重复随机的 cover 池，secret 按顺序
        total = min(C["count"], len(secret_paths), len(cover_paths))
        cover_pool = list(cover_paths)
        rng.shuffle(cover_pool)

        pairs = []
        pool_idx = 0
        for i in range(total):
            spath = secret_paths[i]
            # 从 cover_pool 取一张不同路径的
            chosen = None
            for j in range(pool_idx, len(cover_pool)):
                cpath = cover_pool[j]
                if not (C["avoid_self"] and Path(cpath).resolve() == Path(spath).resolve()):
                    chosen = cpath
                    cover_pool[pool_idx], cover_pool[j] = cover_pool[j], cover_pool[pool_idx]
                    pool_idx += 1
                    break
            if chosen is None:
                if pool_idx >= len(cover_pool): break
                chosen = cover_pool[pool_idx]; pool_idx += 1
            pairs.append((spath, chosen))

        # 3) 生成与保存（安全打开；个别失败时补位）
        for k, (spath, cpath) in enumerate(tqdm(pairs, desc="Generating", unit="img")):
            # secret
            try:
                sim = open_image_safe(spath)
                try:
                    secret = to_rgb_square(sim, C["size"])
                finally:
                    sim.close()
            except Exception:
                # secret 极少数仍失败：顺延下一张 secret 顶替
                repl = None
                for t in secret_paths[len(pairs):]:
                    try:
                        tmp = open_image_safe(t); tmp.close()
                        repl = t; break
                    except Exception:
                        continue
                if repl is None:  # 没有替补就跳过
                    continue
                spath = repl
                sim = open_image_safe(spath)
                try:
                    secret = to_rgb_square(sim, C["size"])
                finally:
                    sim.close()

            # cover
            try:
                cim = open_image_safe(cpath)
                try:
                    cover = to_rgb_square(cim, C["size"])
                finally:
                    cim.close()
            except Exception:
                # 用剩余 cover 顶替
                replaced = False
                while pool_idx < len(cover_pool):
                    cpath2 = cover_pool[pool_idx]; pool_idx += 1
                    try:
                        cim2 = open_image_safe(cpath2)
                        try:
                            cover = to_rgb_square(cim2, C["size"])
                        finally:
                            cim2.close()
                        cpath = cpath2
                        replaced = True
                        break
                    except Exception:
                        continue
                if not replaced:
                    continue  # 无可替代则跳过该对

            left_img, right_img = (cover, secret) if C["left"] == "cover" else (secret, cover)

            out = Image.new("RGB", (2 * C["size"], C["size"]))
            out.paste(left_img,  (0, 0))
            out.paste(right_img, (C["size"], 0))

            idx = start_idx + k
            while True:
                fname = f"{str(idx).zfill(C['zfill'])}.{C['fmt']}"
                out_path = out_dir / fname
                if not out_path.exists():
                    break
                idx += 1

            if C["fmt"].lower() == "jpg":
                out.save(out_path, format="JPEG", quality=C["quality"], subsampling=C["subsampling"], optimize=True)
            else:
                out.save(out_path, format="PNG")

    else:
        total = min(C["count"], len(secret_paths))
        for k in tqdm(range(total), desc="Generating (noise)", unit="img"):
            spath = secret_paths[k]
            cover  = rand_noise_any(C["size"], rng, np_rng, C["noise_pool"], C["noise_weights"])
            sim = open_image_safe(spath)
            try:
                secret = to_rgb_square(sim, C["size"])
            finally:
                sim.close()

            left_img, right_img = (cover, secret) if C["left"] == "cover" else (secret, cover)

            out = Image.new("RGB", (2 * C["size"], C["size"]))
            out.paste(left_img,  (0, 0))
            out.paste(right_img, (C["size"], 0))

            idx = start_idx + k
            while True:
                fname = f"{str(idx).zfill(C['zfill'])}.{C['fmt']}"
                out_path = out_dir / fname
                if not out_path.exists():
                    break
                idx += 1

            if C["fmt"].lower() == "jpg":
                out.save(out_path, format="JPEG", quality=C["quality"], subsampling=C["subsampling"], optimize=True)
            else:
                out.save(out_path, format="PNG")

if __name__ == "__main__":
    main(CONFIG)
