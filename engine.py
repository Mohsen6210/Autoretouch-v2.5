"""
Auto Retouch engine
-------------------
1) Face detection + landmarks (MediaPipe) -> face skin polygon, minus eyes / brows / lips
2) Person segmentation (MediaPipe Selfie Segmentation, pre-trained, Apache-2.0)
   + a skin colour model learned from the detected face -> neck / body / hands skin
3) Dodge & burn: smooth, bounded corrections (brightness + colour) that even out
   mid-scale unevenness while leaving the pore-level texture untouched.
"""
from dataclasses import dataclass, field

import os
import shutil
import tempfile

import cv2
import numpy as np
import mediapipe as mp


def _ascii_mediapipe_fix(force=False):
    """MediaPipe's C++ loader cannot open model files under folders with non-ASCII names
    (e.g. Persian). In that case copy the few models we need to an ASCII folder and
    point MediaPipe there."""
    from mediapipe.python import solution_base as sb
    mod = os.path.abspath(sb.__file__)
    root = os.path.dirname(os.path.dirname(os.path.dirname(mod)))
    if root.isascii() and not force:
        return None
    names = ("face_detection", "face_landmark", "selfie_segmentation", "iris_landmark")
    cands = [os.environ.get("PROGRAMDATA"), r"C:\Users\Public", tempfile.gettempdir(), "C:\\"]
    for base in cands:
        if not base or not base.isascii():
            continue
        dst = os.path.join(base, "AutoRetouch_mp")
        try:
            for n in names:
                src = os.path.join(root, "mediapipe", "modules", n)
                tgt = os.path.join(dst, "mediapipe", "modules", n)
                if os.path.isdir(src) and not os.path.isdir(tgt):
                    shutil.copytree(src, tgt, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            sb.__file__ = os.path.join(dst, "mediapipe", "python", "solution_base.py")
            return dst
        except Exception:
            continue
    return None


try:
    _ascii_mediapipe_fix()
except Exception:
    pass

_FD = mp.solutions.face_detection
_FM = mp.solutions.face_mesh
_SS = mp.solutions.selfie_segmentation

MODES = ("auto", "face", "half", "full")


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def resize_max(img, m, interp=cv2.INTER_AREA):
    h, w = img.shape[:2]
    s = m / max(h, w)
    if s >= 1:
        return img
    return cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=interp)


def gblur(a, sigma):
    """Gaussian blur; very large sigmas are done on a down-scaled copy (fast)."""
    if sigma < 6:
        return cv2.GaussianBlur(a, (0, 0), max(sigma, 0.1))
    f = max(2, int(sigma / 3))
    h, w = a.shape[:2]
    small = cv2.resize(a, (max(1, w // f), max(1, h // f)), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), sigma / f)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def nblur(x, w, sigma):
    """Normalised (mask-weighted) blur: average of x over the masked area only."""
    num = gblur(x * w, sigma)
    den = gblur(w, sigma)
    return num / (den + 1e-4)


def _hull(pts):
    return cv2.convexHull(np.asarray(pts, np.float32).reshape(-1, 1, 2)).astype(np.int32)


def _idx(conn):
    return sorted({i for c in conn for i in c})


@dataclass
class Face:
    box: tuple
    oval: np.ndarray
    excl: list
    face_px: float
    center: tuple
    mesh_ok: bool


@dataclass
class Analysis:
    shape: tuple
    bgr: np.ndarray = None
    faces: list = field(default_factory=list)
    person: np.ndarray = None
    lab_small: np.ndarray = None
    person_small: np.ndarray = None
    med_face_px: float = 0.0


# ----------------------------------------------------------------------------
# detection
# ----------------------------------------------------------------------------
def _iou(a, b):
    ax0, ay0, ax1, ay1 = a[0], a[1], a[0] + a[2], a[1] + a[3]
    bx0, by0, bx1, by1 = b[0], b[1], b[0] + b[2], b[1] + b[3]
    iw = max(0, min(ax1, bx1) - max(ax0, bx0))
    ih = max(0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih
    u = a[2] * a[3] + b[2] * b[3] - inter
    return inter / u if u > 0 else 0


def _nms(boxes, thr=0.35):
    boxes = sorted(boxes, key=lambda b: -b[4])
    keep = []
    for b in boxes:
        if all(_iou(b, k) < thr for k in keep):
            keep.append(b)
    return keep


def _detect_boxes(rgb):
    H, W = rgb.shape[:2]
    boxes = []

    def run(det, im, ox, oy):
        res = det.process(np.ascontiguousarray(im))
        h_, w_ = im.shape[:2]
        for d in (res.detections or []):
            bb = d.location_data.relative_bounding_box
            boxes.append((ox + bb.xmin * w_, oy + bb.ymin * h_, bb.width * w_, bb.height * h_, float(d.score[0])))

    with _FD.FaceDetection(model_selection=1, min_detection_confidence=0.4) as far, \
            _FD.FaceDetection(model_selection=0, min_detection_confidence=0.5) as near:
        run(far, rgb, 0, 0)
        run(near, rgb, 0, 0)
        boxes = _nms(boxes)
        if not boxes:  # small faces: look at overlapping tiles
            tw, th = int(W * 0.62), int(H * 0.62)
            for oy in (0, H - th):
                for ox in (0, W - tw):
                    run(far, rgb[oy:oy + th, ox:ox + tw], ox, oy)
            boxes = _nms(boxes)
    boxes = [b for b in boxes if b[2] > 12 and b[3] > 12 and b[4] > 0.3]
    return boxes[:12]


def _mesh_points(fm, rgb, box):
    x, y, w, h, _ = box
    cx, cy = x + w / 2, y + h / 2
    side = max(w, h) * 2.2
    S = 640
    sc = S / side
    M = np.array([[sc, 0, S / 2 - cx * sc], [0, sc, S / 2 - cy * sc]], np.float32)
    crop = cv2.warpAffine(rgb, M, (S, S), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    res = fm.process(crop)
    if not res.multi_face_landmarks:
        return None
    lm = res.multi_face_landmarks[0].landmark
    pts = np.array([[l.x * S, l.y * S] for l in lm], np.float32)
    pts[:, 0] = (pts[:, 0] - M[0, 2]) / sc
    pts[:, 1] = (pts[:, 1] - M[1, 2]) / sc
    mc = pts.mean(0)
    if abs(mc[0] - cx) > w * 0.6 or abs(mc[1] - cy) > h * 0.6:
        return None  # locked onto a different face
    return pts


def _ellipse_face(box):
    x, y, w, h, _ = box
    cx, cy = x + w / 2, y + h / 2
    t = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    oval = np.stack([cx + 0.52 * w * np.cos(t), cy + 0.58 * h * np.sin(t)], 1)
    excl = []
    for ex in (0.30, 0.70):  # rough eye / brow zones
        c = (x + ex * w, y + 0.40 * h)
        e = np.stack([c[0] + 0.15 * w * np.cos(t), c[1] + 0.07 * h * np.sin(t)], 1)
        excl.append(_hull(e))
        c2 = (x + ex * w, y + 0.31 * h)
        e2 = np.stack([c2[0] + 0.17 * w * np.cos(t), c2[1] + 0.05 * h * np.sin(t)], 1)
        excl.append(_hull(e2))
    c = (cx, y + 0.78 * h)
    lips = np.stack([c[0] + 0.22 * w * np.cos(t), c[1] + 0.08 * h * np.sin(t)], 1)
    excl.append(_hull(lips))
    return _hull(oval), excl


def analyze(bgr):
    """Detect faces + person once. Returns an Analysis (mode independent)."""
    H, W = bgr.shape[:2]
    an = Analysis(shape=(H, W), bgr=bgr)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    boxes = _detect_boxes(rgb)
    ov, le, re_ = _idx(_FM.FACEMESH_FACE_OVAL), _idx(_FM.FACEMESH_LEFT_EYE), _idx(_FM.FACEMESH_RIGHT_EYE)
    lb, rb, lips = _idx(_FM.FACEMESH_LEFT_EYEBROW), _idx(_FM.FACEMESH_RIGHT_EYEBROW), _idx(_FM.FACEMESH_LIPS)

    if boxes:
        with _FM.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True,
                          min_detection_confidence=0.3) as fm:
            for box in boxes:
                pts = _mesh_points(fm, rgb, box)
                if pts is not None:
                    oval = _hull(pts[ov])
                    excl = [_hull(pts[i]) for i in (le, re_, lb, rb, lips)]
                    ok = True
                else:
                    oval, excl = _ellipse_face(box)
                    ok = False
                bx, by, bw, bh = cv2.boundingRect(oval)
                an.faces.append(Face(box, oval, excl, float(bh), (bx + bw / 2, by + bh / 2), ok))
        an.med_face_px = float(np.median([f.face_px for f in an.faces]))

    try:
        with _SS.SelfieSegmentation(model_selection=0) as ss:
            an.person = ss.process(rgb).segmentation_mask.astype(np.float32)
    except Exception:
        an.person = None
    return an


# ----------------------------------------------------------------------------
# mask
# ----------------------------------------------------------------------------
def resolve_mode(an, mode):
    if mode != "auto":
        return mode
    if not an.faces:
        return "face"
    r = an.med_face_px / an.shape[0]
    if r > 0.45:
        return "face"
    if r > 0.15:
        return "half"
    return "full"


def build_mask(an, mode="auto"):
    """Skin mask (float32 0..1) at the analysis resolution + info dict."""
    H, W = an.shape
    used = resolve_mode(an, mode)
    info = {"mode": used, "faces": len(an.faces), "mesh": sum(f.mesh_ok for f in an.faces)}
    if not an.faces:
        return np.zeros((H, W), np.float32), info

    face_poly = np.zeros((H, W), np.uint8)
    excl = np.zeros((H, W), np.uint8)
    for f in an.faces:
        cv2.fillConvexPoly(face_poly, f.oval, 255)
        for e in f.excl:
            cv2.fillConvexPoly(excl, e, 255)
    fp = an.med_face_px
    k = max(1, int(fp * 0.014)) | 1
    excl = cv2.dilate(excl, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k * 2 + 1, k * 2 + 1)))
    ke = max(1, int(fp * 0.02))
    face_poly = cv2.erode(face_poly, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ke * 2 + 1, ke * 2 + 1)))

    ext = np.zeros((H, W), np.uint8)
    if used in ("half", "full"):
        ext = _skin_extension(an, face_poly, excl, used)

    mask = np.maximum(face_poly, ext)
    mask[excl > 0] = 0
    return mask.astype(np.float32) / 255.0, info


def _skin_extension(an, face_poly, excl, used):
    H, W = an.shape
    s = 1000.0 / max(H, W)
    s = min(1.0, s)
    sw, sh = max(1, int(W * s)), max(1, int(H * s))
    small = cv2.resize(an.bgr, (sw, sh), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(small.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)
    fpoly = cv2.resize(face_poly, (sw, sh), interpolation=cv2.INTER_NEAREST)
    ex = cv2.resize(excl, (sw, sh), interpolation=cv2.INTER_NEAREST)

    if an.person is not None:
        person = cv2.resize(an.person, (sw, sh), interpolation=cv2.INTER_LINEAR)
        kk = max(3, int(0.03 * max(sw, sh))) | 1
        person_ok = cv2.dilate((person > 0.35).astype(np.uint8), np.ones((kk, kk), np.uint8)) > 0
    else:
        person_ok = np.ones((sh, sw), bool)

    sel = (cv2.erode(fpoly, np.ones((5, 5), np.uint8)) > 0) & (ex == 0)
    if sel.sum() < 150:
        return np.zeros((H, W), np.uint8)

    ab = lab[..., 1:3]
    Ls = lab[..., 0][sel]
    a_b = ab[sel]
    mu = a_b.mean(0)
    cov = np.cov(a_b.T) + np.eye(2, dtype=np.float32) * 1.0
    inv = np.linalg.inv(cov).astype(np.float32)
    d = ab - mu.astype(np.float32)
    m2 = d[..., 0] ** 2 * inv[0, 0] + 2 * d[..., 0] * d[..., 1] * inv[0, 1] + d[..., 1] ** 2 * inv[1, 1]
    lmin = np.percentile(Ls, 3) - 24
    lmax = np.percentile(Ls, 99.7) + 12
    skin = (m2 < 9.0) & (lab[..., 0] > lmin) & (lab[..., 0] < lmax) & person_ok
    skin = skin.astype(np.uint8) * 255
    skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    skin = cv2.morphologyEx(skin, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    fp_s = an.med_face_px * s
    seed = cv2.dilate(fpoly, np.ones((int(fp_s * 0.2) | 1, int(fp_s * 0.2) | 1), np.uint8)) > 0
    # hair / head zone: never extend the skin colour model over the top half of the head
    hair = np.zeros((sh, sw), np.uint8)
    for f in an.faces:
        cx, cy = f.center[0] * s, f.center[1] * s
        fps = f.face_px * s
        x0, x1 = int(cx - 1.1 * fps), int(cx + 1.1 * fps)
        y1 = int(cy + 0.12 * fps)
        hair[max(0, 0):max(0, y1), max(0, x0):max(0, x1)] = 255
    skin[hair > 0] = 0
    n, labels, stats, _ = cv2.connectedComponentsWithStats(skin, connectivity=8)
    out = np.zeros((sh, sw), np.uint8)
    min_area = (fp_s * 0.12) ** 2
    if used == "half":
        ys = np.arange(sh)[:, None]
        cy = np.mean([f.center[1] for f in an.faces]) * s
        vert = (ys < cy + 2.4 * fp_s)
    for i in range(1, n):
        comp = labels == i
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            continue
        if used == "half":
            if (comp & seed).any():
                out[comp & vert] = 255
        else:
            out[comp] = 255
    out = cv2.resize(out, (W, H), interpolation=cv2.INTER_LINEAR)
    return (out > 127).astype(np.uint8) * 255


def feather(mask, face_px, amount=20):
    """Soft mask edge. amount 0..100 (20 = default); sigma grows with face size."""
    s = max(0.6, face_px * float(amount) / 2000.0)
    return np.clip(cv2.GaussianBlur(mask, (0, 0), s), 0, 1)


# ----------------------------------------------------------------------------
# retouch
# ----------------------------------------------------------------------------
def guided(I, p, r, eps):
    """Edge-preserving smoothing of p guided by I (guided filter, O(1) box filters)."""
    k = (2 * r + 1, 2 * r + 1)

    def bf(x):
        return cv2.boxFilter(x, -1, k, borderType=cv2.BORDER_REPLICATE)

    mI, mp = bf(I), bf(p)
    cov = bf(I * p) - mI * mp
    var = bf(I * I) - mI * mI
    a = cov / (var + eps)
    b = mp - a * mI
    return bf(a) * I + bf(b)


def compute_delta(lab, m, face_px):
    """
    Frequency-separation dodge & burn.
      tone T  = edge-preserving smooth of L  (keeps real edges: nostrils, lids, hairline)
      texture = L - T                          (pores; NEVER touched here)
    The correction pulls T toward the local skin average at two scales (large shading and
    blemish / dark-circle scale). Large deviations (real facial structure) are left alone
    by a soft knee, and dark structures (hair, stubble, deep shadows) are protected.
    Returns smooth additive corrections (dL, da, db) - they never blur the image.
    """
    L, a, b = cv2.split(lab)
    w = np.clip(m, 0, 1).astype(np.float32)
    r = max(2, int(face_px / 80))
    T = guided(L, L, r, 20.0)

    wsum = float(w.sum()) + 1e-6
    muT = float((T * w).sum()) / wsum
    prot = 1.0 - np.clip((muT - T - 10.0) / 10.0, 0, 1)  # protect clearly darker structure
    wp = (w * prot).astype(np.float32)

    sb = max(5.0, face_px / 7.0)
    sm = max(3.0, face_px / 18.0)
    d = 0.5 * (nblur(T, wp, sb) - T) + 0.5 * (nblur(T, wp, sm) - T)
    c = 9.0
    dL = d * np.exp(-0.5 * (d / c) ** 2)
    dL = np.where(dL < 0, dL * 0.8, dL)  # lighten a little more than darken

    out = [dL]
    cc = 3.5
    for ch in (a, b):
        Tc = guided(L, ch, r * 2, 20.0)
        dc = nblur(Tc, wp, sb * 0.8) - Tc
        out.append(0.8 * dc * np.exp(-0.5 * (dc / cc) ** 2))
    return tuple((x * w).astype(np.float32) if False else x.astype(np.float32) for x in out)


def apply_maps(bgr_u8, maps, blend, strength, warmth, texture=1.0, s_tex=1.0,
               db=None, db_limit=False, contrast=0.0, sharpen=0.0, s_sharp=1.0, limit=False):
    """
    maps: float32 HxWx4 = (mask, dL, da, db) at the SAME size as bgr_u8.
    texture: 1.0 = keep skin texture, >1 boost, <1 soften (inside the mask).
    db: manual dodge/burn map in L units (same size), added to L (smooth, texture untouched).
    contrast: -1..1 (S-curve on L). sharpen: >=0 (unsharp mask on L, radius s_sharp).
    limit: restrict sharpen/contrast to the skin mask.
    """
    m = maps[..., 0]
    wgt = m * float(blend)
    lab = cv2.cvtColor(bgr_u8.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)
    L = lab[..., 0].copy()
    corr = strength * maps[..., 1]
    if abs(texture - 1.0) > 1e-3:
        hp = L - cv2.GaussianBlur(L, (0, 0), max(0.5, s_tex))
        corr = corr + (texture - 1.0) * hp
    L1 = L + wgt * corr
    act = wgt > 1e-3
    if db is not None:
        dbv = db * (m if db_limit else 1.0)
        L1 = L1 + dbv
        act = act | (np.abs(dbv) > 1e-3)
    gw = m if limit else 1.0
    if contrast != 0.0:
        x = np.clip(L1 / 100.0, 0.0, 1.0)
        sm = x * x * (3.0 - 2.0 * x)
        L1 = L1 + ((x + contrast * (sm - x)) * 100.0 - L1) * gw
    if sharpen > 0.0:
        hp = L1 - cv2.GaussianBlur(L1, (0, 0), max(0.5, s_sharp))
        L1 = L1 + sharpen * hp * gw
    if contrast != 0.0 or sharpen > 0.0:
        act = (act | (m > 1e-3)) if limit else np.ones_like(act)
    lab[..., 0] = L1
    lab[..., 1] += wgt * (strength * maps[..., 2] + warmth * 0.04)
    lab[..., 2] += wgt * (strength * maps[..., 3] + warmth * 0.16)
    new = np.clip(cv2.cvtColor(lab, cv2.COLOR_LAB2BGR), 0, 1)
    new = (new * 255.0 + 0.5).astype(np.uint8)
    return np.where(act[..., None], new, bgr_u8)


def render_full(bgr, maps_p, blend, strength, warmth, texture=1.0, s_tex_p=1.0, progress=None,
                db_p=None, db_limit=False, contrast=0.0, sharpen=0.0, s_sharp_p=1.0, limit=False):
    """Apply the preview-resolution maps to the full-resolution image, strip by strip.
    Texture / sharpen high-pass filters are computed at FULL resolution (with row padding)."""
    H, W = bgr.shape[:2]
    Hp, Wp = maps_p.shape[:2]
    k = H / Hp
    s_tex, s_sh = s_tex_p * k, s_sharp_p * k
    pad = int(3 * max(s_tex, s_sh)) + 2
    out = np.empty_like(bgr)
    step = 512
    xs = ((np.arange(W, dtype=np.float32) + 0.5) * (Wp / W) - 0.5)
    for y0 in range(0, H, step):
        y1 = min(H, y0 + step)
        a0, a1 = max(0, y0 - pad), min(H, y1 + pad)
        ys = ((np.arange(a0, a1, dtype=np.float32) + 0.5) * (Hp / H) - 0.5)
        mx, my = np.meshgrid(xs, ys)
        up = cv2.remap(maps_p, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        dbu = None
        if db_p is not None:
            dbu = cv2.remap(db_p, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        res = apply_maps(bgr[a0:a1], up, blend, strength, warmth, texture, s_tex,
                         db=dbu, db_limit=db_limit, contrast=contrast, sharpen=sharpen,
                         s_sharp=s_sh, limit=limit)
        out[y0:y1] = res[y0 - a0:y0 - a0 + (y1 - y0)]
        if progress:
            progress(y1 / H)
    return out
