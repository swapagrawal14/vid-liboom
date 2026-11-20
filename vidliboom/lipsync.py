"""Lip-sync and blinks for AI-art PNG characters. No external models.

Three pieces:

* ``mouth_track(audio)``  -> per-video-frame mouth openness (0..1) and a coarse
  viseme ("closed", "small", "open", "wide", "o") from RMS energy, spectral
  centroid and zero-crossing rate, with attack/release smoothing.
* ``blink_schedule(duration, seed)`` -> per-frame eyelid closure (0..1):
  a blink every 2-6 s, 4 frames each, now and then a double blink.
* ``apply_face(img, name, pose, openness, viseme, blink)`` -> new RGBA image.
  Characters with a visible mouth (BABLU) get their drawn mouth covered with a
  skin patch and a cartoon mouth drawn on top. Masked characters (DIDI, UNCLE)
  get their face mask mesh-warped (puff / jaw stretch) instead. Eyelids are
  drawn for blinks on every character.

Landmarks live in assets/characters/faces.json, as fractions of the pose PNG
(x of width, y of height), so ``apply_face`` works on the PNG at any uniform
scale. Apply it before rotating or cropping the image.

Typical use::

    from vidliboom import lipsync, voice, characters
    audio = voice.read_wav(path)
    track = lipsync.mouth_track(audio)
    blinks = lipsync.blink_schedule(len(track) / 24, seed=1)
    for f in range(len(track)):
        im = lipsync.apply_face(characters.load("BABLU", "idle"), "BABLU", "idle",
                                track.openness[f], track.viseme[f], blinks[f])
"""
import json
import math
import os
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

SR = 44100
FPS = 24
FACES = "assets/characters/faces.json"
VISEMES = ("closed", "small", "open", "wide", "o")

OUTLINE = (50, 35, 30)
INTERIOR = (125, 28, 38)
TONGUE = (232, 112, 122)
TEETH = (255, 253, 245)

# ---------------------------------------------------------------- audio -> visemes


@dataclass
class MouthTrack:
    """Per-frame mouth animation. ``openness[i]`` is 0..1, ``viseme[i]`` one of VISEMES."""
    openness: np.ndarray
    viseme: list
    fps: int = FPS
    features: dict = field(default_factory=dict, repr=False)

    def __len__(self):
        return len(self.openness)

    def at(self, t):
        """(openness, viseme) at time t seconds; closed outside the track."""
        i = int(t * self.fps)
        if 0 <= i < len(self.openness):
            return float(self.openness[i]), self.viseme[i]
        return 0.0, "closed"


def _smooth_attack_release(x, attack=0.65, release=0.3):
    """One-pole follower: rises fast (attack) and falls slower (release)."""
    y = np.zeros_like(x)
    v = 0.0
    for i, s in enumerate(x):
        k = attack if s > v else release
        v += k * (s - v)
        y[i] = v
    return y


def _mode_filter(labels, k=1):
    out = list(labels)
    for i in range(len(labels)):
        win = labels[max(0, i - k): i + k + 1]
        best = max(set(win), key=win.count)
        if win.count(best) > len(win) // 2:
            out[i] = best
    return out


def mouth_track(audio, sr=SR, fps=FPS, attack=0.7, release=0.45):
    """Mouth openness + viseme per video frame from a mono float audio array.

    Heuristics (per frame, window = 2 hops, Hann):
      loudness: RMS in dB mapped from a gate (30 dB under the clip's 97th-pct
                peak, at least 10 dB over its noise floor) to that peak, ^1.5,
                plus a little extra on syllable onsets.
      centroid: spectral centroid 80 Hz-8 kHz, relative to the clip's median
                voiced centroid (so it adapts to the voice's pitch/timbre).
      zcr:      zero-crossing rate, high for fricatives (s, sh, f).
    Classification: quiet -> closed; fricative-ish (high zcr + bright) -> wide
    (teeth); dark/low centroid -> "o"; loud -> open; otherwise small.
    """
    a = np.asarray(audio, np.float32)
    if a.ndim > 1:
        a = a.mean(1)
    hop = sr / fps
    n = max(1, int(math.ceil(len(a) / hop)))
    win = int(1.5 * hop)
    pad = np.concatenate([np.zeros(win // 2, np.float32), a, np.zeros(win, np.float32)])
    w = np.hanning(win).astype(np.float32)
    freqs = np.fft.rfftfreq(win, 1 / sr)
    band = (freqs > 80) & (freqs < 8000)
    rms = np.zeros(n); cen = np.zeros(n); zcr = np.zeros(n)
    for i in range(n):
        s = int(i * hop)
        seg = pad[s: s + win]
        rms[i] = math.sqrt(float((seg * seg).mean()) + 1e-12)
        mag = np.abs(np.fft.rfft(seg * w))[band]
        cen[i] = float((freqs[band] * mag).sum() / (mag.sum() + 1e-9))
        zcr[i] = float(np.mean(np.abs(np.diff(np.signbit(seg).astype(np.int8)))))
    db = 20 * np.log10(rms + 1e-9)
    floor = max(np.percentile(db, 10), -70.0)
    peak = np.percentile(db, 97)
    gate = max(floor + 10, peak - 26, -50.0)
    level = np.clip((db - gate) / max(peak - gate, 6.0), 0, 1) ** 1.5
    voiced = level > 0.1
    ref_c = np.median(cen[voiced]) if voiced.any() else 1500.0
    ref_z = np.median(zcr[voiced]) if voiced.any() else 0.05
    rc = cen / (ref_c + 1e-9)
    rz = zcr / (ref_z + 1e-9)
    # syllable onsets: a rising envelope opens the jaw a little extra
    rise = np.clip(np.diff(level, prepend=0.0), 0, None)
    openness = np.clip(_smooth_attack_release(np.clip(level + 0.5 * rise, 0, 1), attack, release), 0, 1)

    labels = []
    for i in range(n):
        o = openness[i]
        if o < 0.08:
            labels.append("closed")
        elif rz[i] > 2.2 and rc[i] > 1.35:
            labels.append("wide")          # s / sh / f / ch: teeth showing
        elif rc[i] < 0.65 and o > 0.2:
            labels.append("o")             # dark vowels: o, u, w
        elif o > 0.5:
            labels.append("open")          # a, aa
        elif rc[i] > 1.2 and o > 0.2:
            labels.append("wide")          # e, ee, i
        else:
            labels.append("small")
    # smooth the shape choice (not open/closed) so it doesn't flicker
    sm = _mode_filter(labels, 1)
    labels = [a if a == "closed" or b == "closed" else b for a, b in zip(labels, sm)]
    for i in range(1, n - 1):  # hold each open shape >= 2 frames
        if "closed" not in (labels[i - 1], labels[i]) and labels[i] != labels[i - 1] and labels[i] != labels[i + 1]:
            labels[i] = labels[i - 1]
    return MouthTrack(openness.astype(np.float32), labels, fps,
                      {"level": level, "centroid": cen, "zcr": zcr})


def visemes(audio, sr=SR, fps=FPS):
    """Shorthand: (openness array, viseme list)."""
    t = mouth_track(audio, sr, fps)
    return t.openness, t.viseme


# ---------------------------------------------------------------- blinks

BLINK_SHAPE = (0.55, 1.0, 0.8, 0.3)  # 4 frames: closing, shut, opening, almost open


def blink_schedule(duration, seed=0, fps=FPS, min_gap=2.0, max_gap=6.0, double_p=0.18):
    """Per-frame eyelid closure 0..1 for ``duration`` seconds.

    A blink every 2-6 s (first one 0.4-2.5 s in), 4 frames long; with
    probability ``double_p`` a second blink follows two frames later."""
    rng = np.random.default_rng(seed)
    n = int(round(duration * fps))
    out = np.zeros(n, np.float32)
    t = rng.uniform(0.4, 2.5)
    while True:
        f = int(t * fps)
        if f >= n:
            break
        starts = [f]
        if rng.random() < double_p:
            starts.append(f + len(BLINK_SHAPE) + 2)
        for s in starts:
            for k, v in enumerate(BLINK_SHAPE):
                if 0 <= s + k < n:
                    out[s + k] = max(out[s + k], v)
        t += rng.uniform(min_gap, max_gap) + (0.3 if len(starts) > 1 else 0)
    return out


# ---------------------------------------------------------------- landmarks

_faces = None


def load_faces(path=FACES):
    global _faces
    if _faces is None or path != FACES:
        f = json.load(open(path)) if os.path.exists(path) else {}
        if path != FACES:
            return f
        _faces = f
    return _faces


def landmarks(name, pose):
    """(character entry, pose landmarks) or (None, None). Unknown poses fall back to idle."""
    c = load_faces().get(name)
    if not c:
        return None, None
    L = c["poses"].get(pose) or c["poses"].get("idle")
    return c, L


def has_face(name, pose="idle"):
    return landmarks(name, pose)[1] is not None


# ---------------------------------------------------------------- drawing helpers

SS = 4  # supersampling for anti-aliased drawing


def _down(layer):
    """Box-downsample an SS-x layer with premultiplied alpha (no halos or ringing)."""
    a = np.asarray(layer, np.float32)
    h, w = a.shape[0] // SS * SS, a.shape[1] // SS * SS
    a = a[:h, :w]
    al = a[..., 3:4] / 255.0
    pre = np.concatenate([a[..., :3] * al, a[..., 3:4]], -1)
    pre = pre.reshape(h // SS, SS, w // SS, SS, 4).mean((1, 3))
    al = pre[..., 3:4] / 255.0
    rgb = np.where(al > 0, pre[..., :3] / np.maximum(al, 1e-6), 0)
    return Image.fromarray(np.concatenate([np.clip(rgb, 0, 255), pre[..., 3:4]], -1).round().astype(np.uint8), "RGBA")


def _sample_color(arr, cx, cy, rx, ry, ref, tol=70, ring=(1.15, 1.8)):
    """Median colour of the pixels in an elliptical ring that resemble ``ref``."""
    H, W = arr.shape[:2]
    x0, x1 = int(max(0, cx - rx * ring[1])), int(min(W, cx + rx * ring[1] + 1))
    y0, y1 = int(max(0, cy - ry * ring[1])), int(min(H, cy + ry * ring[1] + 1))
    if x1 <= x0 or y1 <= y0:
        return tuple(ref)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    r = np.sqrt(((xx - cx) / max(rx, 1)) ** 2 + ((yy - cy) / max(ry, 1)) ** 2)
    px = arr[y0:y1, x0:x1]
    sel = (r > ring[0]) & (r < ring[1]) & (px[..., 3] > 200)
    rgb = px[..., :3][sel].astype(int)
    if len(rgb):
        close = rgb[np.abs(rgb - np.array(ref)).sum(1) < tol]
        if len(close) >= 6:
            return tuple(int(v) for v in np.median(close, 0))
    return tuple(ref)


def _stroke(d, pts, width, color):
    """Thick polyline with round caps/joints (PIL's wide lines leave gaps)."""
    pts = [(float(x), float(y)) for x, y in pts]
    width = max(1, int(round(width)))
    d.line(pts, fill=color, width=width)
    r = (width - 1) / 2
    for x, y in pts:
        d.ellipse((x - r, y - r, x + r, y + r), fill=color)


def _ellipse_mask(size, cx, cy, rx, ry, feather):
    m = Image.new("L", size, 0)
    ImageDraw.Draw(m).ellipse((cx - rx, cy - ry, cx + rx, cy + ry), fill=255)
    return m.filter(ImageFilter.GaussianBlur(feather)) if feather > 0 else m


def _mouth_polygon(w, h, top_bulge, smile, n=28):
    """Closed polygon (centred at 0,0) for a mouth opening w x h.
    top_bulge: 0 = flat top (D shape), 0.5 = ellipse. smile lifts the corners."""
    pts = []
    for i in range(n + 1):  # top edge, left -> right
        u = -1 + 2 * i / n
        top = (1 - u * u) ** (0.5 if top_bulge >= 0.4 else 1.0)
        pts.append((u * w / 2, -h * top_bulge * top - smile * h * u * u * 0.35))
    for i in range(1, n):  # bottom edge, right -> left
        u = 1 - 2 * i / n
        bot = (1 - u * u) ** (0.5 if top_bulge >= 0.4 else 0.75)
        pts.append((u * w / 2, h * (1 - top_bulge) * bot - smile * h * u * u * 0.35))
    return pts


def _draw_mouth(size, cx, cy, mw, openness, viseme, line):
    """RGBA layer (size) with the cartoon mouth, drawn at SS x and downsampled."""
    W, H = size
    pad = int(mw * 1.2)
    box = (int(cx - pad), int(cy - pad), int(cx + pad), int(cy + pad))
    bw, bh = (box[2] - box[0]) * SS, (box[3] - box[1]) * SS
    ox, oy = (cx - box[0]) * SS, (cy - box[1]) * SS
    lw = max(2, int(round(line * SS)))
    S = mw * SS
    layer = Image.new("RGBA", (bw, bh), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    o = float(np.clip(openness, 0, 1))

    if viseme == "closed" or o < 0.06:
        # gentle closed smile
        w = S * 0.62
        pts = [(ox + u * w / 2, oy - S * 0.05 + S * 0.09 * (1 - u * u)) for u in np.linspace(-1, 1, 24)]
        _stroke(d, pts, lw, OUTLINE + (255,))
    else:
        if viseme == "o":
            w, h, tb, sm = S * (0.38 + 0.08 * o), S * (0.30 + 0.35 * o), 0.5, 0.0
        elif viseme == "wide":
            w, h, tb, sm = S * (0.92 + 0.1 * o), S * (0.16 + 0.22 * o), 0.12, 0.25
        elif viseme == "small":
            w, h, tb, sm = S * (0.50 + 0.12 * o), S * (0.12 + 0.30 * o), 0.2, 0.15
        else:  # open
            w, h, tb, sm = S * (0.70 + 0.12 * o), S * (0.25 + 0.45 * o), 0.15, 0.2
        poly = [(ox + x, oy + y) for x, y in _mouth_polygon(w, h, tb, sm)]
        inner = Image.new("L", (bw, bh), 0)
        ImageDraw.Draw(inner).polygon(poly, fill=255)
        fill = Image.new("RGBA", (bw, bh), INTERIOR + (255,))
        fd = ImageDraw.Draw(fill)
        bottom = oy + h * (1 - tb)
        top = oy - h * tb
        if viseme in ("open", "small", "o") and h > S * 0.18:
            tw = w * (0.55 if viseme != "o" else 0.7)
            th = h * 0.42
            fd.ellipse((ox - tw / 2, bottom - th, ox + tw / 2, bottom + th * 0.6), fill=TONGUE + (255,))
        if viseme == "wide":
            fd.rectangle((ox - w / 2, top - S, ox + w / 2, top + h * 0.42), fill=TEETH + (255,))
            fd.ellipse((ox - w * 0.18, bottom - h * 0.35, ox + w * 0.18, bottom + h * 0.3), fill=TONGUE + (255,))
        elif viseme == "open" and o > 0.45:
            fd.rectangle((ox - w * 0.28, top - S, ox + w * 0.28, top + h * 0.16), fill=TEETH + (255,))
        layer.paste(fill, (0, 0), inner)
        d = ImageDraw.Draw(layer)
        # thinner outline on small openings so they don't read as a dark blob
        _stroke(d, poly + [poly[0]], max(2, min(lw, h * 0.22)), OUTLINE + (255,))
    layer = _down(layer)
    full = Image.new("RGBA", size, (0, 0, 0, 0))
    full.paste(layer, (box[0], box[1]))
    return full


def _eyelids(img, arr, c, L, blink, line):
    """Skin lids sliding down over each eye; ``blink`` 0 (open) .. 1 (shut)."""
    closed = L.get("eyes_closed", False)
    closed = list(closed) if isinstance(closed, (list, tuple)) else [bool(closed)] * len(L["eyes"])
    if blink <= 0.02 or all(closed):
        return img
    W, H = img.size
    skin = c.get("skin", (240, 210, 180))
    outlined = L.get("eye_style", c.get("eye_style", "dot")) == "outlined"
    out = img
    ew, eh = L["eye_w"], L["eye_h"]
    ew = list(ew) if isinstance(ew, (list, tuple)) else [ew] * len(L["eyes"])
    eh = list(eh) if isinstance(eh, (list, tuple)) else [eh] * len(L["eyes"])
    for (ex, ey), shut, e_w, e_h in zip(L["eyes"], closed, ew, eh):
        if shut:
            continue
        cx, cy = ex * W, ey * H
        pad = (1.04, 1.2) if outlined else (1.3, 1.4)  # dots: also hide their halo
        rx = e_w * W / 2 * pad[0] + 1.0
        ry = e_h * H / 2 * pad[1] + 1.0
        lid_col = _sample_color(arr, cx, cy, rx, ry, skin, tol=90, ring=(1.1, 1.9))
        x0, y0 = int(cx - rx - 3), int(cy - ry - 3)
        bw, bh = int(2 * rx + 7) * SS, int(2 * ry + 7) * SS
        lx, ly = (cx - x0) * SS, (cy - y0) * SS
        RX, RY = rx * SS, ry * SS
        # lid edge: a downward-bowed curve whose lowest point sinks with blink
        yy, xx = np.mgrid[0:bh, 0:bw].astype(np.float32)
        u = np.clip((xx - lx) / RX, -1, 1)
        edge = ly - RY + 2 * RY * blink * (0.55 + 0.45 * (1 - u * u))
        shut_y = 0.3 if outlined else 0.15
        if blink > 0.95:
            edge = ly + RY * shut_y * (1 - u * u)  # shut: a soft smile-shaped lash line
        inside = ((xx - lx) / RX) ** 2 + ((yy - ly) / RY) ** 2 <= 1
        lid = (inside & ((yy <= edge) | (blink > 0.95))).astype(np.uint8) * 255
        layer = Image.new("RGBA", (bw, bh), lid_col + (255,))
        # never paint lids outside the character's silhouette, and keep the dark
        # outlines near the rim (eye outline, face edge) visible around the lid
        src = Image.fromarray(arr).crop((x0, y0, x0 + bw // SS, y0 + bh // SS)).resize((bw, bh), Image.NEAREST)
        sa = np.asarray(src).astype(np.float32)
        lum = sa[..., :3] @ np.array([0.299, 0.587, 0.114], np.float32)
        # keep dark lines in the rim: the eye's own outline (outlined eyes) and
        # the face edge; once shut, only the upper lid outline stays
        rim = ((xx - lx) / RX) ** 2 + ((yy - ly) / RY) ** 2 > (0.62 if outlined else 0.8)
        protect = rim & (lum < 110)
        if blink > 0.95:
            protect &= yy < edge - 2 * SS
        # in the rim, only paint over lid-coloured (skin) pixels: never over
        # a neighbouring background, glass, hair or clothing
        skinlike = np.abs(sa[..., :3] - np.array(lid_col, np.float32)).sum(-1) < 45
        outer = ((xx - lx) / RX) ** 2 + ((yy - ly) / RY) ** 2 > 0.9
        other = ~skinlike & (lum >= 110)
        if outlined:  # eye whites reach the rim: those may be covered
            other &= sa[..., :3].min(-1) < 200
        protect |= outer & other
        keep = (sa[..., 3] > 0) & ~protect
        layer.putalpha(Image.fromarray(np.where(keep, lid, 0).astype(np.uint8)))
        d = ImageDraw.Draw(layer)
        xs = np.linspace(-1, 1, 30)
        pts = []
        for v in xs:
            px = lx + v * RX * 0.92
            ey_ = ly - RY + 2 * RY * blink * (0.55 + 0.45 * (1 - v * v)) if blink <= 0.95 else ly + RY * shut_y * (1 - v * v)
            # keep the lash line inside the eye ellipse
            lim = RY * math.sqrt(max(0.0, 1 - (v * 0.92) ** 2))
            pts.append((px, min(max(ey_, ly - lim), ly + lim)))
        lw = max(2, int(round(line * SS * (0.9 if blink < 0.95 else 1.1))))
        if blink >= 0.4:  # a lid only just starting to drop shows no lash line yet
            _stroke(d, pts, lw, OUTLINE + (255,))
        layer = _down(layer)
        out.alpha_composite(layer, (x0, y0))
    return out


def _warp_mask(img, L, openness, viseme, strength=1.0):
    """Mesh-warp the face-mask region: jaw stretch + puff while talking."""
    o = float(np.clip(openness, 0, 1)) * strength
    if o < 0.02:
        return img
    W, H = img.size
    cx, cy = L["mask"][0] * W, L["mask"][1] * H
    rx, ry = L["mask_w"] * W / 2 * 1.3, L["mask_h"] * H / 2 * 1.6
    x0, x1 = int(max(0, cx - rx - 2)), int(min(W, cx + rx + 3))
    y0, y1 = int(max(0, cy - ry - 2)), int(min(H, cy + ry + 3))
    arr = np.asarray(img, np.float32)[y0:y1, x0:x1]
    a = arr[..., 3:4] / 255.0
    pre = np.concatenate([arr[..., :3] * a, arr[..., 3:4]], -1)
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    nx, ny = (xx - cx) / rx, (yy - cy) / ry
    r2 = nx * nx + ny * ny
    fall = np.clip(1 - r2, 0, 1) ** 1.5
    sx = {"o": -0.05, "wide": 0.06, "open": 0.015, "small": 0.01}.get(viseme, 0.01)
    sy = {"o": 0.10, "wide": 0.04, "open": 0.13, "small": 0.07}.get(viseme, 0.05)
    # forward displacement (output point = source + disp); inverted approximately.
    # Horizontal: puff (+) or pucker (-) around the centre. Vertical: the jaw
    # drops, so points below the mask's top move down, more toward the chin.
    dx = (sx + 0.03) * o * nx * rx * fall
    dy = (sy * np.clip(ny + 0.6, 0, None) + 0.02 * ny) * o * ry * fall
    srcx = np.clip(xx - dx - x0, 0, x1 - x0 - 1)
    srcy = np.clip(yy - dy - y0, 0, y1 - y0 - 1)
    ix, iy = np.floor(srcx).astype(int), np.floor(srcy).astype(int)
    fx, fy = (srcx - ix)[..., None], (srcy - iy)[..., None]
    ix1, iy1 = np.minimum(ix + 1, x1 - x0 - 1), np.minimum(iy + 1, y1 - y0 - 1)
    s = (pre[iy, ix] * (1 - fx) * (1 - fy) + pre[iy, ix1] * fx * (1 - fy)
         + pre[iy1, ix] * (1 - fx) * fy + pre[iy1, ix1] * fx * fy)
    al = s[..., 3:4]
    rgb = np.where(al > 0, s[..., :3] / np.maximum(al, 1e-6) * 255.0, 0)
    patch = np.concatenate([np.clip(rgb, 0, 255), np.clip(al, 0, 255)], -1).astype(np.uint8)
    out = img.copy()
    out.paste(Image.fromarray(patch, "RGBA"), (x0, y0))
    return out


_cover_cache = {}


def _mouth_cover(arr, name, pose, c, L, line):
    """(skin colour, L mask) that paints out the pose's original mouth.

    The mask is the non-skin blob(s) touching the landmark mouth box, dilated
    and feathered, so nearby features (nose, cheeks, outline) survive. Cached
    per pose and image size."""
    H, W = arr.shape[:2]
    key = (name, pose, W, H)
    if key in _cover_cache:
        return _cover_cache[key]
    mx, my = L["mouth"][0] * W, L["mouth"][1] * H
    hw, hh = L["mouth_w"] * W / 2, L["mouth_h"] * H / 2
    skin = _sample_color(arr, mx, my, hw * 1.3 + line, hh * 1.4 + line, c.get("skin", (250, 225, 200)), tol=45)
    x0, x1 = int(max(0, mx - hw * 1.8 - 4)), int(min(W, mx + hw * 1.8 + 5))
    y0, y1 = int(max(0, my - hh * 2.2 - 4)), int(min(H, my + hh * 2.2 + 5))
    reg = arr[y0:y1, x0:x1].astype(int)
    diff = (np.abs(reg[..., :3] - np.array(skin)).sum(-1) > 40) & (reg[..., 3] > 128)
    # flood-fill from non-skin pixels inside the inner mouth box
    seed = np.zeros_like(diff)
    iy0, iy1 = int(my - hh - y0), int(my + hh - y0) + 1
    ix0, ix1 = int(mx - hw - x0), int(mx + hw - x0) + 1
    seed[max(0, iy0):iy1, max(0, ix0):ix1] = True
    blob = diff & seed
    while True:  # grow within the non-skin pixels (small region, few iterations)
        g = blob.copy()
        g[1:] |= blob[:-1]; g[:-1] |= blob[1:]; g[:, 1:] |= blob[:, :-1]; g[:, :-1] |= blob[:, 1:]
        g &= diff
        if (g == blob).all():
            break
        blob = g
    m = np.zeros((H, W), np.uint8)
    m[y0:y1, x0:x1] = blob * 255
    rad = max(1, int(round(line * 0.9)))
    mi = Image.fromarray(m).filter(ImageFilter.MaxFilter(2 * rad + 1)).filter(ImageFilter.GaussianBlur(max(0.8, line * 0.4)))
    mi = Image.fromarray(np.minimum(np.asarray(mi), arr[..., 3]))
    _cover_cache[key] = (skin, mi)
    return skin, mi


# ---------------------------------------------------------------- public drawing API


def apply_face(img_rgba, name, pose, openness=0.0, viseme="closed", blink=0.0):
    """Return a new RGBA image of the pose with the mouth/mask and eyelids animated.

    img_rgba: the pose PNG (any uniform scale, unrotated, uncropped).
    openness: 0..1; viseme: one of VISEMES; blink: 0 (open) .. 1 (shut).
    Characters/poses missing from faces.json are returned unchanged (copied)."""
    img = img_rgba.convert("RGBA").copy()
    c, L = landmarks(name, pose)
    if L is None:
        return img
    W, H = img.size
    eye_dx = abs(L["eyes"][1][0] - L["eyes"][0][0]) * W
    line = max(1.2, eye_dx * 0.045)
    arr = np.asarray(img)
    if c.get("type") == "mask":
        img = _warp_mask(img, L, openness, viseme, L.get("warp", 1.0))
    else:
        mx, my = L["mouth"][0] * W, L["mouth"][1] * H
        skin, m = _mouth_cover(arr, name, pose, c, L, line)
        img.paste(Image.new("RGBA", img.size, skin + (255,)), (0, 0), m)
        mw = max(eye_dx * 0.5, L["mouth_w"] * W * 0.7)
        img.alpha_composite(_draw_mouth(img.size, mx, my, mw, openness, viseme, line))
    return _eyelids(img, np.asarray(img), c, L, float(blink), line)


def talking_frames(name, pose, audio, seed=0, sr=SR, fps=FPS, loader=None):
    """Generator of RGBA frames for a whole line of dialogue (one per video frame)."""
    if loader is None:
        from . import characters
        loader = characters.load
    track = mouth_track(audio, sr, fps)
    blinks = blink_schedule(len(track) / fps, seed, fps)
    base = loader(name, pose)
    for i in range(len(track)):
        b = float(blinks[i]) if i < len(blinks) else 0.0
        yield apply_face(base, name, pose, track.openness[i], track.viseme[i], b)
