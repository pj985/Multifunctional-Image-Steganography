# build_from_coco_train_10k_filter_first.py
# 依赖：pip install pillow numpy tqdm requests opencv-python

from pathlib import Path
import os, json, random, zipfile, requests
import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import cv2

# 限制底层并发，稳定性能
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
cv2.setNumThreads(0)

# ==================== 只编辑这里：CONFIG ====================
CONFIG = {
    # 输出与数量
    "OUT_ROOT": r"E:\Py_projects\Patent\Code\data\download_cocotrain",  # 输出根目录
    "SIZE": 256,                 # 导出尺寸（边长）
    "PER_BUCKET": 1000,          # 载体每档：low/mid/high 各多少
    "SECRETS_N": 1000,           # 秘密张数
    "SEED": 0,                   # 随机种子

    # 训练集子集（逐步下载，先 10000，不足自动补）
    "TRAIN_SUBSET_DIR": r"E:\coco_cache\train_subset_auto",
    "N_INIT": 10000,             # 初次下载张数
    "DL_BATCH": 1000,            # 不足时每次追加下载多少
    "DL_WORKERS": 16,            # 下载并发

    # 纹理分桶
    "TAIL_FRAC": 0.25,           # 极端抽样尾部比例（拉开 low/high）
    "SAVE_SHUFFLE": True,        # 保存前打乱顺序，避免按纹理排序
}
# ===========================================================

ANN_ZIP_URL   = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
COCO_TRAIN_URL = "http://images.cocodataset.org/train2017/{fname}"
IMG_EXTS = {".jpg", ".jpeg", ".JPG", ".JPEG"}

# Pillow 兼容
try:
    RESAMPLE_LANCZOS = Image.Resampling.LANCZOS
except AttributeError:
    from PIL import Image as _PIL
    RESAMPLE_LANCZOS = getattr(_PIL, "LANCZOS", Image.ANTIALIAS)

# ---------- 基础工具 ----------
def to_rgb_square(im, size):
    # 统一到 RGB + 等比居中裁切为 size×size
    if im.mode == "RGBA":
        bg = Image.new("RGBA", im.size, (255,255,255,255))
        im = Image.alpha_composite(bg, im).convert("RGB")
    else:
        im = im.convert("RGB")
    return ImageOps.fit(im, (size, size), method=RESAMPLE_LANCZOS, centering=(0.5,0.5))

def robust_z(x):
    x = np.asarray(x, dtype=np.float64)
    q1, med, q3 = np.percentile(x, 25), np.median(x), np.percentile(x, 75)
    iqr = max(q3 - q1, 1e-6)
    return (x - med) / iqr

# ---------- 注释与文件名 ----------
def ensure_annotations(ann_cache_dir):
    ann_cache = Path(ann_cache_dir); ann_cache.mkdir(parents=True, exist_ok=True)
    ann_zip = ann_cache / "annotations_trainval2017.zip"
    ann_json = ann_cache / "instances_train2017.json"
    if not ann_json.exists():
        if not ann_zip.exists():
            print("[COCO] 下载 annotations_trainval2017.zip")
            with requests.get(ANN_ZIP_URL, stream=True, timeout=60) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0))
                with open(ann_zip, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=ann_zip.name) as pbar:
                    for ch in r.iter_content(1024*1024):
                        if ch: f.write(ch); pbar.update(len(ch))
        print("[COCO] 解压 annotations …")
        with zipfile.ZipFile(ann_zip, "r") as zf:
            zf.extract("annotations/instances_train2017.json", ann_cache)
        src = ann_cache / "annotations" / "instances_train2017.json"
        src.replace(ann_json)
        try: (ann_cache / "annotations").rmdir()
        except Exception: pass
    return ann_json

def get_shuffled_filenames(ann_json, seed):
    imgs = json.loads(Path(ann_json).read_text(encoding="utf-8"))["images"]
    fns = [x["file_name"] for x in imgs]  # e.g. "000000581781.jpg"
    rnd = random.Random(seed); rnd.shuffle(fns)
    return fns

# ---------- 下载 ----------
def download_missing(out_dir, fn_list, workers=16):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    def _dl(fn):
        dst = out_dir / fn
        if dst.exists(): return True
        try:
            with requests.get(COCO_TRAIN_URL.format(fname=fn), timeout=60, stream=True) as r:
                r.raise_for_status()
                with open(dst, "wb") as f:
                    for ch in r.iter_content(1024*64):
                        if ch: f.write(ch)
            return True
        except Exception:
            return False
    need = [fn for fn in fn_list if not (out_dir / fn).exists()]
    ok = 0
    if need:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_dl, fn) for fn in need]
            for fut in tqdm(as_completed(futs), total=len(futs), desc="download"):
                if fut.result(): ok += 1
    return ok

# ---------- 筛选（杜绝问题文件，保证仅 RGB） ----------
def is_usable_rgb(path: Path) -> bool:
    """
    可打开 + 真正解码成功 + 模式为 RGB。
    Pillow 失败时不回退 OpenCV，直接判为不可用，以杜绝异常文件。
    """
    try:
        with Image.open(path) as im:
            im.load()              # 触发真实解码，暴露潜在错误
            return im.mode == "RGB"
    except Exception:
        return False

def filter_usable_rgb(paths):
    usable = []
    for p in tqdm(paths, desc="filter RGB & readable"):
        if is_usable_rgb(p):
            usable.append(p)
    return usable

# ---------- 快速纹理打分（在 256×256 上） ----------
def fast_texture_score_on_256(p):
    with Image.open(p) as im:
        # 这里已保证是 RGB 可读文件
        im = to_rgb_square(im, 256)
        g = np.asarray(im.convert("L"), dtype=np.uint8)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    G = float(cv2.mean(cv2.magnitude(gx, gy))[0])       # 梯度幅值均值
    L = float(cv2.Laplacian(g, cv2.CV_32F, ksize=3).var())  # Laplacian 方差
    return {"G": G, "L": L}

# ---------- 主流程 ----------
def build_pipeline(C):
    rng = random.Random(C["SEED"]); np.random.seed(C["SEED"])
    out_root = Path(C["OUT_ROOT"]); out_root.mkdir(parents=True, exist_ok=True)
    subset_dir = Path(C["TRAIN_SUBSET_DIR"]); subset_dir.mkdir(parents=True, exist_ok=True)

    # A. 准备文件名序列（固定乱序）
    ann_json = ensure_annotations(ann_cache_dir=subset_dir.parent / "ann")
    all_fns  = get_shuffled_filenames(ann_json, C["SEED"])
    cursor   = 0

    # B. 先下载 N_INIT
    init_pick = all_fns[cursor : cursor + C["N_INIT"]]; cursor += C["N_INIT"]
    download_missing(subset_dir, init_pick, workers=C["DL_WORKERS"])

    # C. 列出当前已下载
    def list_downloaded():
        return sorted([p for p in subset_dir.glob("*.jpg") if p.suffix in IMG_EXTS])

    downloaded = list_downloaded()

    # D. 筛选：仅可读且 RGB
    usable = filter_usable_rgb(downloaded)

    # E. 若筛后不足，继续“下载→筛选”直到可用数 ≥ 需要值
    need_total = 3 * C["PER_BUCKET"] + C["SECRETS_N"]  # 例如 4000
    while len(usable) < need_total and cursor < len(all_fns):
        add = min(C["DL_BATCH"], len(all_fns) - cursor)
        more_pick = all_fns[cursor : cursor + add]; cursor += add
        print(f"[TopUp] 追加下载 {add} 张（可用 {len(usable)}/{need_total}）")
        download_missing(subset_dir, more_pick, workers=C["DL_WORKERS"])
        # 只对“新下载”的做增量筛选
        new_files = [subset_dir / fn for fn in more_pick if (subset_dir / fn).exists()]
        usable += filter_usable_rgb(new_files)

    if len(usable) < need_total:
        raise RuntimeError(f"可用 RGB 图片不足以选 3×{C['PER_BUCKET']} + {C['SECRETS_N']}，当前 {len(usable)}")

    # F. 在 256×256 上打分
    print(f"[Score] 对 {len(usable)} 张计算纹理分数…")
    feats = []
    for p in tqdm(usable, desc="scoring"):
        s = fast_texture_score_on_256(p)
        feats.append((p, s["G"], s["L"]))

    G = np.array([x[1] for x in feats], dtype=np.float64)
    L = np.array([x[2] for x in feats], dtype=np.float64)
    T = 0.6*robust_z(G) + 0.4*robust_z(L)  # 快速组合分数：小→平滑，大→粗糙

    paths = [x[0] for x in feats]
    scored = list(zip(paths, T))
    scored.sort(key=lambda x: x[1])

    # G. 两端与中间各取 PER_BUCKET
    n = C["PER_BUCKET"]; k = len(scored)
    tail = max(int(C["TAIL_FRAC"] * k), n)
    mid_center = k // 2
    low_candidates  = scored[:tail]
    high_candidates = scored[-tail:]
    mid_candidates  = scored[max(0, mid_center - tail//2) : min(k, mid_center + tail//2)]

    rng.seed(C["SEED"])
    low_paths  = [p for p,_ in rng.sample(low_candidates,  n)]
    high_paths = [p for p,_ in rng.sample(high_candidates, n)]
    mid_paths  = [p for p,_ in rng.sample(mid_candidates,  n)]

    chosen_set = set(low_paths) | set(mid_paths) | set(high_paths)

    # H. secrets：从“剩余可用集合”随机取 SECRETS_N（与载体零重叠）
    remain = [p for p in paths if p not in chosen_set]
    if len(remain) < C["SECRETS_N"]:
        raise RuntimeError(f"剩余图片不足 secrets：{len(remain)} < {C['SECRETS_N']}")
    rng.shuffle(remain)
    secrets_paths = remain[:C["SECRETS_N"]]

    # I. 导出四个文件夹（全部 256×256 PNG）
    def export_group(name, group_paths):
        out_dir = out_root / name
        group = list(group_paths)
        if C["SAVE_SHUFFLE"]:
            rng.shuffle(group)
        print(f"[Save] {name}: {len(group)}")
        for i, p in enumerate(tqdm(group, desc=name), 1):
            with Image.open(p) as im_src:
                im = to_rgb_square(im_src, C["SIZE"])
            out_dir.mkdir(parents=True, exist_ok=True)
            im.save(out_dir / f"{i:04d}.png", format="PNG")

    export_group("covers_low",  low_paths)
    export_group("covers_mid",  mid_paths)
    export_group("covers_high", high_paths)
    export_group("secrets",     secrets_paths)

def main():
    C = CONFIG
    random.seed(C["SEED"]); np.random.seed(C["SEED"])
    build_pipeline(C)
    print("\n完成：", Path(C["OUT_ROOT"]))
    print(f"  covers_low/mid/high: 各 {C['PER_BUCKET']}；secrets: {C['SECRETS_N']}；尺寸 {C['SIZE']}×{C['SIZE']} PNG")
    print(f"  子集目录：{C['TRAIN_SUBSET_DIR']} | 初次下载 {C['N_INIT']}，不足则每次追加 {C['DL_BATCH']}")
if __name__ == "__main__":
    main()
