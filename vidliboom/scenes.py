"""Parallax procedural scene library.

Six illustrated, pastel, outlined backdrops that match the AI-art cartoon cast:

    park, metro, clinic, city, space, rooftop

Every scene is built once per (name, w, h, mood) into a handful of cached RGBA
layers (drawn 2x supersampled, then graded for the mood and given a light paper
grain). Per frame only the moving parts are touched: layers are shifted by
``camera_x * depth`` (they tile horizontally, so any camera value works),
clouds drift, trees / clothes / handles sway, kites bob, rain falls and puddles
reflect.

API
    SCENES                                   list of scene names
    MOODS                                    ("day", "sunset", "night", "rain")
    render_scene(name, w, h, t, mood="day", camera_x=0.0) -> HxWx3 uint8
    render(w, h, t, mood="day", bg="sky")    drop-in for backgrounds.render;
                                             unknown bg falls back to it.
"""
import math

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter

from . import backgrounds

__all__ = ["SCENES", "MOODS", "render_scene", "render", "clear_cache"]

SCENES = ["park", "metro", "clinic", "city", "space", "rooftop"]
MOODS = ("day", "sunset", "night", "rain")
OUT = (50, 35, 30)
SS = 2  # supersampling for cached layers

# --------------------------------------------------------------------------
# colour grading per mood
# --------------------------------------------------------------------------
GRADE = {
    "day": dict(mul=(1.0, 1.0, 1.0), add=(0, 0, 0), sat=1.0, haze=(0.83, 0.90, 0.97)),
    "sunset": dict(mul=(1.03, 0.82, 0.80), add=(0.05, 0.0, 0.03), sat=1.05, haze=(0.99, 0.70, 0.62)),
    "night": dict(mul=(0.42, 0.46, 0.66), add=(0.015, 0.02, 0.06), sat=0.85, haze=(0.17, 0.19, 0.36)),
    "rain": dict(mul=(0.74, 0.79, 0.85), add=(0.02, 0.03, 0.05), sat=0.62, haze=(0.62, 0.67, 0.73)),
}
SKY = {
    "day": [(0.0, (0.47, 0.71, 0.95)), (0.55, (0.70, 0.85, 0.98)), (1.0, (0.96, 0.95, 0.90))],
    "sunset": [(0.0, (0.36, 0.32, 0.60)), (0.4, (0.86, 0.50, 0.60)), (0.75, (1.0, 0.70, 0.50)),
               (1.0, (1.0, 0.86, 0.62))],
    "night": [(0.0, (0.03, 0.04, 0.14)), (0.6, (0.09, 0.11, 0.29)), (1.0, (0.24, 0.21, 0.42))],
    "rain": [(0.0, (0.42, 0.46, 0.53)), (1.0, (0.70, 0.73, 0.77))],
}
GLOW = (255, 214, 140)


def _mix(a, b, t):
    return tuple(int(round(x + (y - x) * t)) for x, y in zip(a, b))


def _grade_arr(rgb, mood, haze=0.0, interior=0.0):
    g = GRADE[mood]
    k = 1.0 - interior
    lum = (rgb @ np.array([0.3, 0.59, 0.11], np.float32))[..., None]
    sat = 1 + (g["sat"] - 1) * k
    rgb = lum + (rgb - lum) * sat
    mul = 1 + (np.array(g["mul"], np.float32) - 1) * k
    rgb = rgb * mul + np.array(g["add"], np.float32) * k
    if haze:
        rgb = rgb * (1 - haze) + np.array(g["haze"], np.float32) * haze
    return rgb


def _grain(shape, seed, amt=1.0):
    rng = np.random.default_rng(seed)
    h, w = shape
    fine = rng.normal(0, 0.016 * amt, (h, w)).astype(np.float32)
    cw, ch = max(2, w // 24), max(2, h // 24)
    coarse = Image.fromarray(((rng.random((ch, cw)) * 255)).astype(np.uint8)).resize((w, h), Image.BICUBIC)
    coarse = (np.asarray(coarse, np.float32) / 255 - 0.5) * 0.028 * amt
    return 1 + fine + coarse


# --------------------------------------------------------------------------
# drawing toolkit
# --------------------------------------------------------------------------
def _dshape(d, sh, f, fill=None, grow=0.0):
    """Draw shape ``sh`` (1x coords) through transform f, optionally grown."""
    k = sh[0]
    if k == "e":
        x0, y0, x1, y1 = sh[1]
        a, b = f(x0 - grow, y0 - grow), f(x1 + grow, y1 + grow)
        if b[0] > a[0] and b[1] > a[1]:
            d.ellipse((a[0], a[1], b[0], b[1]), fill=fill)
    elif k == "r":
        x0, y0, x1, y1 = sh[1]
        a, b = f(x0 - grow, y0 - grow), f(x1 + grow, y1 + grow)
        r = max(0, (sh[2] + grow) * SS)
        if b[0] > a[0] and b[1] > a[1]:
            d.rounded_rectangle((a[0], a[1], b[0], b[1]), radius=min(r, (b[0] - a[0]) / 2, (b[1] - a[1]) / 2), fill=fill)
    elif k == "p":
        pts = [f(x, y) for x, y in sh[1]]
        d.polygon(pts, fill=fill)
        if grow > 0:
            lw = max(1, int(2 * grow * SS))
            d.line(pts + [pts[0]], fill=fill, width=lw, joint="curve")
            rr = (lw - 1) / 2
            for x, y in (pts if len(pts) < 16 else []):
                d.ellipse((x - rr, y - rr, x + rr, y + rr), fill=fill)


def _bounds(shapes):
    xs, ys = [], []
    for sh in shapes:
        if sh[0] in ("e", "r"):
            xs += [sh[1][0], sh[1][2]]
            ys += [sh[1][1], sh[1][3]]
        else:
            xs += [p[0] for p in sh[1]]
            ys += [p[1] for p in sh[1]]
    return min(xs), min(ys), max(xs), max(ys)


def _shift(m, dx, dy):
    o = Image.new("L", m.size, 0)
    o.paste(m, (dx, dy))
    return o


def _paste(dst, src, x, y):
    """alpha_composite that tolerates negative / overflowing positions."""
    x, y = int(x), int(y)
    sw, sh = src.size
    x0, y0 = max(0, -x), max(0, -y)
    x1, y1 = min(sw, dst.width - x), min(sh, dst.height - y)
    if x1 <= x0 or y1 <= y0:
        return
    if (x0, y0, x1, y1) != (0, 0, sw, sh):
        src = src.crop((x0, y0, x1, y1))
    dst.alpha_composite(src, (x + x0, y + y0))


class Painter:
    """RGBA canvas drawn at SS x resolution; optionally tiles horizontally."""

    def __init__(self, w, h, u=1.0, wrap=True):
        self.w, self.h, self.u, self.wrap = int(w), int(h), u, wrap
        self.img = Image.new("RGBA", (self.w * SS, self.h * SS), (0, 0, 0, 0))
        self.d = ImageDraw.Draw(self.img)
        self.lm = None  # mask of light-emitting pixels (exempt from grading)
        self.lmd = None

    def _offs(self, x0, x1, wrap):
        if not (wrap and self.wrap):
            return [0]
        return [o for o in (-self.w, 0, self.w) if x0 + o < self.w and x1 + o > 0]

    def blobs(self, shapes, top, bot=None, ow=2.6, light=0.28, dark=0.22, rim=None,
              wrap=True, out=OUT, gy=None):
        """Union of shapes, gradient filled, one shared outline, soft cel shading."""
        bot = bot or top
        ow = ow * self.u
        x0, y0, x1, y1 = _bounds(shapes)
        gy0, gy1 = gy if gy else (y0, y1)
        pad = ow * 2 + 3
        if rim is None:
            rim = min(max(0.09 * min(x1 - x0, y1 - y0), 2.0), 16.0)
        for o in self._offs(x0 - pad, x1 + pad, wrap):
            bx0, by0 = int((x0 - pad + o) * SS), int((y0 - pad) * SS)
            bw = int((x1 - x0 + 2 * pad) * SS) + 2
            bh = int((y1 - y0 + 2 * pad) * SS) + 2
            if bx0 + bw < 0 or bx0 > self.img.width or by0 > self.img.height or by0 + bh < 0:
                continue

            def f(x, y, o=o, bx0=bx0, by0=by0):
                return ((x + o) * SS - bx0, y * SS - by0)

            inner = Image.new("L", (bw, bh), 0)
            di = ImageDraw.Draw(inner)
            for sh in shapes:
                _dshape(di, sh, f, 255)
            outer = None
            if ow > 0:
                outer = Image.new("L", (bw, bh), 0)
                do = ImageDraw.Draw(outer)
                for sh in shapes:
                    _dshape(do, sh, f, 255, grow=ow)
            # vertical gradient
            col = np.zeros((bh, 3), np.float32)
            yy = (np.arange(bh) + by0) / SS
            tt = np.clip((yy - gy0) / max(1e-3, gy1 - gy0), 0, 1)[:, None]
            col[:] = np.array(top, np.float32) * (1 - tt) + np.array(bot, np.float32) * tt
            colimg = Image.fromarray(col[:, None, :].astype(np.uint8)).resize((bw, bh), Image.NEAREST)
            k = int(rim * self.u * SS)
            if k >= 2 and (light or dark):
                base = np.asarray(colimg, np.float32)
                res = base.copy()
                if light:
                    hl = ImageChops.subtract(inner, _shift(inner, k, k)).filter(ImageFilter.GaussianBlur(k * 0.6))
                    a = np.asarray(hl, np.float32)[..., None] / 255 * light
                    res = res * (1 - a) + (base * 0.45 + 255 * 0.55) * a
                if dark:
                    dk = ImageChops.subtract(inner, _shift(inner, -k, -k)).filter(ImageFilter.GaussianBlur(k * 0.6))
                    a = np.asarray(dk, np.float32)[..., None] / 255 * dark
                    res = res * (1 - a) + (base * 0.62 + np.array((70, 40, 90), np.float32) * 0.38) * a
                colimg = Image.fromarray(np.clip(res, 0, 255).astype(np.uint8))
            src = Image.new("RGBA", (bw, bh), out + (0,))
            if outer is not None:
                src.putalpha(outer)
            fill = colimg.convert("RGBA")
            fill.putalpha(inner)
            src.alpha_composite(fill)
            _paste(self.img, src, bx0, by0)
            if self.lm is not None:
                _lpaste(self.lm, 0, outer if outer is not None else inner, bx0, by0)

    def fill(self, shapes, color, wrap=True):
        """Plain fill (replaces pixels; use alpha 0 to punch holes)."""
        x0, _, x1, _ = _bounds(shapes)
        if len(color) == 3:
            color = tuple(color) + (255,)
        for o in self._offs(x0, x1, wrap):
            for sh in shapes:
                _dshape(self.d, sh, lambda x, y, o=o: ((x + o) * SS, y * SS), color)
                if self.lmd is not None:
                    _dshape(self.lmd, sh, lambda x, y, o=o: ((x + o) * SS, y * SS), 0)

    def lit(self, shapes, color, alpha=255, blur=0.0, wrap=True):
        if self.lm is None:
            self.lm = Image.new("L", self.img.size, 0)
            self.lmd = ImageDraw.Draw(self.lm)
        self.soft(shapes, color, alpha, blur, wrap, lit=True)

    def soft(self, shapes, color, alpha=255, blur=0.0, wrap=True, lit=False):
        """Blended (alpha-composited) fill, optional blur: glows, shadows, sheens."""
        x0, y0, x1, y1 = _bounds(shapes)
        pad = blur * 3 + 2
        for o in self._offs(x0 - pad, x1 + pad, wrap):
            bx0, by0 = int((x0 - pad + o) * SS), int((y0 - pad) * SS)
            bw, bh = int((x1 - x0 + 2 * pad) * SS) + 2, int((y1 - y0 + 2 * pad) * SS) + 2
            m = Image.new("L", (bw, bh), 0)
            dm = ImageDraw.Draw(m)
            for sh in shapes:
                _dshape(dm, sh, lambda x, y, o=o: ((x + o) * SS - bx0, y * SS - by0), alpha)
            if blur:
                m = m.filter(ImageFilter.GaussianBlur(blur * SS))
            src = Image.new("RGBA", (bw, bh), tuple(color[:3]) + (0,))
            src.putalpha(m)
            _paste(self.img, src, bx0, by0)
            if lit:
                _lpaste(self.lm, 255, m, bx0, by0)

    def line(self, pts, color, width=2.5, wrap=True, closed=False):
        xs = [p[0] for p in pts]
        if len(color) == 3:
            color = tuple(color) + (255,)
        for o in self._offs(min(xs) - width, max(xs) + width, wrap):
            q = [((x + o) * SS, y * SS) for x, y in pts]
            if closed:
                q.append(q[0])
            self.d.line(q, fill=color, width=max(1, int(width * self.u * SS)), joint="curve")
            if self.lmd is not None:
                self.lmd.line(q, fill=0, width=max(1, int(width * self.u * SS)), joint="curve")

    def done(self, mood, haze=0.0, interior=0.0, seed=0, grain=1.0, emissive=None, crop=True):
        im = self.img.resize((self.w, self.h), Image.LANCZOS)
        a = np.asarray(im, np.float32) / 255
        rgb = _grade_arr(a[..., :3], mood, haze, interior)
        m = None
        if self.lm is not None:
            m = np.asarray(self.lm.resize((self.w, self.h), Image.BILINEAR), np.float32)[..., None] / 255
            rgb = rgb * (1 - m) + a[..., :3] * m
        if grain:
            rgb = rgb * _grain(rgb.shape[:2], seed, grain)[..., None]
        out = np.concatenate([np.clip(rgb, 0, 1), a[..., 3:]], -1)
        img = Image.fromarray((out * 255 + 0.5).astype(np.uint8), "RGBA")
        if m is not None:  # soft halo around lights
            g = np.concatenate([a[..., :3], np.clip(m * a[..., 3:] * 1.1, 0, 1)], -1)
            g = Image.fromarray((g * 255).astype(np.uint8), "RGBA").filter(ImageFilter.GaussianBlur(11 * self.u))
            img.alpha_composite(g)
        if isinstance(emissive, Painter):
            img.alpha_composite(emissive.glow())
        if crop:
            al = np.asarray(img)[..., 3]
            rows = np.nonzero(al.max(1))[0]
            y0 = int(rows[0]) if len(rows) else 0
            return img.crop((0, y0, img.width, img.height)), y0
        return img, 0

    def glow(self, radius=9, strength=1.6):
        im = self.img.resize((self.w, self.h), Image.LANCZOS)
        g = im.filter(ImageFilter.GaussianBlur(radius * self.u))
        r, gg, b, a = g.split()
        a = a.point(lambda v: min(255, int(v * strength)))
        g = Image.merge("RGBA", (r, gg, b, a))
        g.alpha_composite(im)
        return g


def _lpaste(dst, value, mask, x, y):
    """Paste a constant into L image dst through mask at (x, y), clipped."""
    x, y = int(x), int(y)
    mw, mh = mask.size
    x0, y0 = max(0, -x), max(0, -y)
    x1, y1 = min(mw, dst.width - x), min(mh, dst.height - y)
    if x1 <= x0 or y1 <= y0:
        return
    mk = mask.crop((x0, y0, x1, y1))
    if value:
        reg = dst.crop((x + x0, y + y0, x + x1, y + y1))
        dst.paste(ImageChops.lighter(reg, mk), (x + x0, y + y0))
    else:
        dst.paste(0, (x + x0, y + y0, x + x1, y + y1), mk)


class _Lit:
    def __init__(self, p):
        self.p = p

    def soft(self, shapes, color, alpha=255, blur=0.0, wrap=True):
        self.p.lit(shapes, color, alpha, blur, wrap)


def _periodic(pw, base, terms, x):
    return base + sum(a * math.sin(2 * math.pi * k * x / pw + ph) for a, k, ph in terms)


def _ridge(pw, base, terms, bottom, step=6, pad=40):
    pts = [(x, _periodic(pw, base, terms, x)) for x in range(-pad, pw + pad + step, step)]
    return pts + [(pw + pad, bottom), (-pad, bottom)]


def _wcrop(img, x, w):
    pw = img.width
    x = int(round(x)) % pw
    if x + w <= pw:
        return img.crop((x, 0, x + w, img.height))
    out = Image.new(img.mode, (w, img.height))
    out.paste(img.crop((x, 0, pw, img.height)), (0, 0))
    rest = w - (pw - x)
    while rest > 0:
        seg = min(rest, pw)
        out.paste(img.crop((0, 0, seg, img.height)), (w - rest, 0))
        rest -= seg
    return out


def _place(fr, spr, lx, ly, off, pw, w):
    sx = (lx - off) % pw
    for x in (sx, sx - pw):
        if -spr.width < x < w:
            _paste(fr, spr, int(round(x)), int(round(ly)))


def _sway(img, a):
    """Shear sprite about its bottom edge by slope ``a``."""
    h = img.height
    return img.transform(img.size, Image.AFFINE, (1, a, -a * h, 0, 1, 0), resample=Image.BILINEAR)


def _swing(img, deg, pivot):
    return img.rotate(deg, resample=Image.BILINEAR, center=pivot)


def _sprite(draw_fn, sw, sh, u, mood, haze=0.0, interior=0.0, seed=0, emissive_fn=None):
    p = Painter(sw, sh, u, wrap=False)
    draw_fn(p)
    em = None
    if emissive_fn is not None:
        em = Painter(sw, sh, u, wrap=False)
        if not emissive_fn(em):
            em = None
    img, _ = p.done(mood, haze, interior, seed, emissive=em, crop=False)
    return img


def _cloud(p, cx, cy, s, rng, top=(255, 255, 255), bot=(222, 230, 246), ow=2.2):
    shapes = [("e", (cx - 0.95 * s, cy - 0.15 * s, cx + 0.95 * s, cy + 0.38 * s))]
    for i in range(int(rng.integers(3, 6))):
        r = s * rng.uniform(0.28, 0.5)
        x = cx + rng.uniform(-0.6, 0.6) * s
        y = cy - rng.uniform(0.05, 0.3) * s
        shapes.append(("e", (x - r, y - r, x + r, y + r)))
    p.blobs(shapes, top, bot, ow=ow, light=0.35, dark=0.18, wrap=False)


# --------------------------------------------------------------------------
# frame-level helpers
# --------------------------------------------------------------------------
def _sky(w, h, mood, horizon, u, rng, sun=(0.78, 0.22), stars=True):
    stops = SKY[mood]
    y = np.linspace(0, 1, h) / max(0.2, horizon)
    y = np.clip(y, 0, 1)
    ps = np.array([s[0] for s in stops])
    cs = np.array([s[1] for s in stops])
    col = np.stack([np.interp(y, ps, cs[:, i]) for i in range(3)], -1)
    img = np.broadcast_to(col[:, None, :], (h, w, 3)).astype(np.float32).copy()
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    sx, sy = sun[0] * w, sun[1] * h
    d = np.sqrt((xx - sx) ** 2 + (yy - sy) ** 2) / (min(w, h))
    if mood == "day":
        img += (np.exp(-d * 5.0) * 0.20)[..., None] * np.array([1.0, 0.95, 0.75])
    elif mood == "sunset":
        img += (np.exp(-d * 3.2) * 0.35)[..., None] * np.array([1.0, 0.65, 0.35])
    elif mood == "night":
        img += (np.exp(-d * 6.0) * 0.16)[..., None] * np.array([0.7, 0.75, 1.0])
    img *= _grain((h, w), 99, 0.6)[..., None]
    pil = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).convert("RGBA")
    dd = ImageDraw.Draw(pil)
    r = 46 * u
    if mood == "day":
        dd.ellipse((sx - r, sy - r, sx + r, sy + r), fill=(255, 248, 215))
    elif mood == "sunset":
        r *= 1.5
        dd.ellipse((sx - r, sy - r, sx + r, sy + r), fill=(255, 196, 120))
    elif mood == "night":
        moon = Image.new("RGBA", pil.size, (0, 0, 0, 0))
        md = ImageDraw.Draw(moon)
        r *= 0.9
        md.ellipse((sx - r - 3 * u, sy - r - 3 * u, sx + r + 3 * u, sy + r + 3 * u), fill=OUT + (255,))
        md.ellipse((sx - r, sy - r, sx + r, sy + r), fill=(250, 244, 214, 255))
        md.ellipse((sx - r * 0.5, sy - r * 0.2, sx - r * 0.1, sy + r * 0.2), fill=(232, 224, 196, 255))
        md.ellipse((sx + r * 0.15, sy + r * 0.25, sx + r * 0.45, sy + r * 0.55), fill=(232, 224, 196, 255))
        pil.alpha_composite(moon)
        if stars:
            n = int(170 * w * h / (1280 * 720))
            xs, ys = rng.integers(0, w, n), rng.integers(0, int(h * horizon * 0.95), n)
            for x, y, s in zip(xs, ys, rng.random(n)):
                c = (255, 250, 230) if s > 0.5 else (200, 210, 255)
                rr = (0.6 + 1.4 * s) * u
                dd.ellipse((x - rr, y - rr, x + rr, y + rr), fill=c)
    return pil


def _vignette(w, h, amt):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = ((xx / w - 0.5) ** 2 * 1.2 + (yy / h - 0.5) ** 2) * 2.2
    v = 1 - amt * np.clip(d, 0, 1) ** 1.6
    v = (np.clip(v, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(np.repeat(v[..., None], 3, -1))


def _rain_tile(w, h, u, n, length, width, alpha, slant, seed, color=(225, 233, 255)):
    rng = np.random.default_rng(seed)
    img = Image.new("RGBA", (w * 2, h * 2), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    for _ in range(n):
        x, y = rng.uniform(0, w), rng.uniform(0, h)
        L = length * rng.uniform(0.6, 1.2) * u
        a = int(alpha * rng.uniform(0.5, 1.0))
        for ox in (-w, 0, w, 2 * w):
            for oy in (-h, 0, h, 2 * h):
                d.line((x + ox, y + oy, x + ox + slant * L, y + oy + L), fill=color + (a,),
                       width=max(1, int(width * u)))
    return img


# --------------------------------------------------------------------------
# scene base
# --------------------------------------------------------------------------
class _Layer:
    __slots__ = ("img", "y", "factor", "scroll", "dy", "after")

    def __init__(self, img, y=0, factor=1.0, scroll=0.0, dy=None, after=None):
        self.img, self.y, self.factor, self.scroll, self.dy, self.after = img, y, factor, scroll, dy, after


class _Scene:
    seed = 1
    horizon = 0.70  # sky gradient reaches its bottom stop here
    sun = (0.78, 0.22)
    rain_slant = -0.12
    ground = 0.72  # y fraction of the walkable floor line
    interior = False

    def __init__(self, w, h, mood):
        if mood not in GRADE:
            mood = "day"
        self.w, self.h, self.mood = w, h, mood
        self.u = min(w, h) / 720.0
        self.pw = 2 * w
        self.night, self.rainy = mood == "night", mood == "rain"
        self.rng = np.random.default_rng(self.seed)
        self.layers, self.clouds, self.twinkle = [], [], []
        self.puddle = None  # (mask float32 (rows, pw), y0, mirror_y, layer_index)
        self.splash = None  # (y0, y1, layer_index) for rain splashes
        self.rain_at = None
        self.sky = self.make_sky()
        self.build()
        if self.rain_at is None:
            self.rain_at = len(self.layers)
        self.vig = _vignette(w, h, 0.42 if self.night else 0.26)
        if self.rainy:
            s = self.rain_slant
            self.rain_tiles = [
                (_rain_tile(w, h, self.u, int(260 * w / 1280), 26, 1.3, 90, s, 3), 900, s),
                (_rain_tile(w, h, self.u, int(90 * w / 1280), 48, 2.2, 120, s, 4), 1500, s),
            ]

    # -- hooks ---------------------------------------------------------
    def make_sky(self):
        return _sky(self.w, self.h, self.mood, self.horizon, self.u, self.rng, self.sun)

    def build(self):
        raise NotImplementedError

    def add(self, painter, factor=1.0, haze=0.0, scroll=0.0, dy=None, after=None, emissive=None,
            interior=0.0, grain=1.0):
        img, y0 = painter.done(self.mood, haze, interior, seed=len(self.layers) + 11 * self.seed,
                               grain=grain, emissive=emissive)
        self.layers.append(_Layer(img, y0, factor, scroll, dy, after))
        return len(self.layers) - 1

    def em(self, p):
        """Light-drawing proxy for painter p when the mood calls for lights (else None).
        Lit pixels keep their colour through mood grading and get a soft halo."""
        return _Lit(p) if self.lights > 0 else None

    @property
    def lights(self):
        return {"day": 0.0, "sunset": 0.22, "night": 0.62, "rain": 0.3}[self.mood]

    def make_clouds(self, n, ymin, ymax, smin, smax, speed=6.0, factor=0.06):
        top, bot = (255, 255, 255), (222, 230, 246)
        if self.rainy:
            n = int(n * 1.8)
            top, bot = (205, 210, 220), (165, 172, 188)
            smin, smax = smin * 1.3, smax * 1.4
        for i in range(n):
            s = self.rng.uniform(smin, smax) * self.u
            sw, sh = int(s * 2.4), int(s * 1.2)
            rng = np.random.default_rng(self.seed * 100 + i)
            spr = _sprite(lambda p: _cloud(p, sw / 2, sh * 0.55, s, rng, top, bot), sw, sh, self.u,
                          self.mood, haze=0.12, seed=i)
            if self.night:
                spr = Image.blend(spr, Image.new("RGBA", spr.size, (0, 0, 0, 0)), 0.0)
            self.clouds.append((spr, self.rng.uniform(0, self.w + sw), self.rng.uniform(ymin, ymax) * self.h,
                                speed * self.rng.uniform(0.6, 1.4) * self.u, factor))

    def draw_clouds(self, fr, t, cam):
        for spr, x0, y, sp, fac in self.clouds:
            span = self.w + spr.width + 60
            x = (x0 + sp * t - cam * fac) % span - spr.width
            _paste(fr, spr, x, y)

    def anim_sky(self, fr, t, cam):
        self.draw_clouds(fr, t, cam)

    def post(self, fr, t, cam):
        pass

    # -- per frame -----------------------------------------------------
    def _rain(self, fr, t):
        for tile, speed, slant in self.rain_tiles:
            oy = (speed * self.u * t) % self.h
            ox = (slant * speed * self.u * t) % self.w
            x0 = int(self.w - ox)
            y0 = int(self.h - oy)
            fr.alpha_composite(tile.crop((x0, y0, x0 + self.w, y0 + self.h)))

    def _splashes(self, fr, t, y0, y1):
        d = ImageDraw.Draw(fr, "RGBA")
        u = self.u
        for i in range(int(26 * self.w / 1280)):
            ph = (i * 0.618) % 1
            cyc = t * 2.2 + ph
            k = int(cyc)
            fr_ = cyc - k
            rr = np.random.default_rng(i * 7919 + k)
            x, y = rr.uniform(0, self.w), rr.uniform(y0, y1)
            r = (3 + 10 * fr_) * u * (0.6 + 0.6 * (y - y0) / max(1, y1 - y0))
            a = int(150 * (1 - fr_))
            d.ellipse((x - r, y - r * 0.35, x + r, y + r * 0.35), outline=(235, 242, 255, a), width=max(1, int(1.5 * u)))

    def frame(self, t, cam):
        fr = self.sky.copy()
        self.anim_sky(fr, t, cam)
        offs = []
        for i, L in enumerate(self.layers):
            if self.rainy and i == self.rain_at:
                self._rain(fr, t)
            off = cam * L.factor + L.scroll * t * self.u
            dy = int(round(L.dy(t) * self.u)) if L.dy else 0
            offs.append(off)
            _paste(fr, _wcrop(L.img, off, self.w), 0, L.y + dy)
            if L.after:
                L.after(fr, t, off, dy)
            if self.splash and self.rainy and i == self.splash[2]:
                self._splashes(fr, t, self.splash[0], self.splash[1])
        self.post(fr, t, cam)
        if self.rainy and self.rain_at >= len(self.layers):
            self._rain(fr, t)
        im = ImageChops.multiply(fr.convert("RGB"), self.vig)
        arr = np.array(im)
        if self.rainy and self.puddle is not None:
            self._reflect(arr, t, offs)
        return arr

    def _reflect(self, arr, t, offs):
        mask, y0, ym, li = self.puddle
        off = int(round(offs[li])) % self.pw
        cols = (np.arange(self.w) + off) % self.pw
        m = mask[:, cols]
        rows = np.nonzero(m.max(1) > 0.01)[0]
        if not len(rows):
            return
        r0, r1 = rows[0], rows[-1] + 1
        m = m[r0:r1, :, None]
        ys = np.arange(y0 + r0, y0 + r1)
        src = np.clip(2 * ym - ys, 0, self.h - 1)
        shift = (np.sin(ys * 0.45 + t * 6.0) * 2.5 * self.u).astype(np.int64)
        cc = (np.arange(self.w)[None, :] + shift[:, None]) % self.w
        refl = arr[src[:, None], cc].astype(np.float32)
        reg = arr[y0 + r0:y0 + r1].astype(np.float32)
        out = reg * (1 - m) + (refl * 0.78 + np.array([18, 22, 34], np.float32)) * m
        arr[y0 + r0:y0 + r1] = np.clip(out, 0, 255).astype(np.uint8)

    def puddles(self, li, ellipses, mirror_y, strength=0.62):
        """Register puddle ellipses (layer coords) on layer li for rain reflections."""
        y0 = int(min(e[1] for e in ellipses)) - 2
        y1 = int(max(e[3] for e in ellipses)) + 2
        m = Image.new("L", (self.pw * 2, (y1 - y0) * 2), 0)
        d = ImageDraw.Draw(m)
        for x0, ya, x1, yb in ellipses:
            for o in (-self.pw, 0, self.pw):
                d.ellipse(((x0 + o) * 2, (ya - y0) * 2, (x1 + o) * 2, (yb - y0) * 2), fill=int(255 * strength))
        m = m.resize((self.pw, y1 - y0), Image.LANCZOS).filter(ImageFilter.GaussianBlur(1.2 * self.u))
        self.puddle = (np.asarray(m, np.float32) / 255, y0, int(mirror_y), li)


# --------------------------------------------------------------------------
# shared props
# --------------------------------------------------------------------------
def _tree_canopy(p, cx, cy, s, rng, top=(168, 214, 128), bot=(92, 158, 98)):
    shapes = []
    for i in range(9):
        a = i / 9 * 2 * math.pi
        rr = s * rng.uniform(0.32, 0.46)
        x = cx + math.cos(a) * s * 0.52
        y = cy + math.sin(a) * s * 0.42
        shapes.append(("e", (x - rr, y - rr, x + rr, y + rr)))
    shapes.append(("e", (cx - s * 0.6, cy - s * 0.55, cx + s * 0.6, cy + s * 0.5)))
    p.blobs(shapes, top, bot, ow=2.8, light=0.4, dark=0.3, rim=s * 0.12)
    # leafy texture dabs
    for _ in range(14):
        x = cx + rng.uniform(-0.7, 0.7) * s
        y = cy + rng.uniform(-0.55, 0.45) * s
        r = s * rng.uniform(0.06, 0.1)
        c = _mix(top, (255, 255, 230), 0.35) if y < cy else _mix(bot, (40, 90, 70), 0.25)
        p.soft([("e", (x - r * 1.4, y - r, x + r * 1.4, y + r))], c, 150, blur=0.6, wrap=False)


def _lamp_post(p, em, x, base, top, u, glow=True):
    p.blobs([("r", (x - 4, top + 10, x + 4, base), 3)], (96, 104, 120), (70, 76, 92), ow=2.2, light=0.3)
    p.blobs([("r", (x - 10, base - 10, x + 10, base + 2), 4)], (96, 104, 120), (70, 76, 92), ow=2.2)
    p.blobs([("p", [(x - 16, top + 12), (x + 16, top + 12), (x + 10, top - 14), (x - 10, top - 14)])],
            (250, 240, 200), (230, 214, 170), ow=2.4)
    p.blobs([("r", (x - 20, top + 8, x + 20, top + 16), 3), ("p", [(x - 12, top - 14), (x + 12, top - 14), (x, top - 26)])],
            (96, 104, 120), (70, 76, 92), ow=2.4)
    if em is not None and glow:
        em.soft([("e", (x - 13, top - 12, x + 13, top + 11))], GLOW, 255)
        em.soft([("e", (x - 70, top - 60, x + 70, top + 70))], GLOW, 70, blur=18)


def _window_grid(p, em, x0, y0, x1, y1, cols, rows, rng, lit, glass=(205, 228, 240), frame=None,
                 gap=0.28, rad=2):
    cw, rh = (x1 - x0) / cols, (y1 - y0) / rows
    for i in range(cols):
        for j in range(rows):
            a = x0 + i * cw + cw * gap / 2
            b = y0 + j * rh + rh * gap / 2
            box = (a, b, a + cw * (1 - gap), b + rh * (1 - gap))
            if frame:
                p.fill([("r", (box[0] - 2, box[1] - 2, box[2] + 2, box[3] + 2), rad)], frame)
            p.blobs([("r", box, rad)], glass, _mix(glass, (120, 150, 190), 0.35), ow=1.5, light=0.5, dark=0.0, rim=2)
            if em is not None and rng.random() < lit:
                c = GLOW if rng.random() < 0.75 else (255, 236, 196)
                em.soft([("r", box, rad)], c, 235)
                r = rng.random()
                if r < 0.35:  # curtain
                    cw = (box[2] - box[0]) * 0.32
                    em.soft([("r", (box[0], box[1], box[0] + cw, box[3]), rad)], (226, 150, 110), 220)
                elif r < 0.5:  # lamp-shade blob
                    cx, cy = (box[0] + box[2]) / 2, box[3] - (box[3] - box[1]) * 0.3
                    rr = (box[2] - box[0]) * 0.16
                    em.soft([("e", (cx - rr, cy - rr, cx + rr, cy + rr))], (255, 250, 230), 255)


# --------------------------------------------------------------------------
# PARK
# --------------------------------------------------------------------------
class Park(_Scene):
    seed = 11
    horizon = 0.62
    sun = (0.74, 0.2)

    def build(self):
        w, h, u, pw, rng = self.w, self.h, self.u, self.pw, self.rng
        self.make_clouds(5, 0.04, 0.3, 40, 70)
        gy = int(h * 0.70)
        # far hills
        p = Painter(pw, h, u)
        p.blobs([("p", _ridge(pw, h * 0.47, [(30, 2, 0.4), (16, 5, 1.3), (8, 9, 2.0)], h * 0.8))],
                (190, 214, 196), (160, 194, 176), ow=2.0, wrap=False, light=0.3, rim=10)
        r2 = [(24, 3, 2.2), (12, 7, 0.3), (6, 13, 1.0)]
        for i in range(int(pw / (34 * u))):  # distant tree clumps peeking over the near ridge
            x = rng.uniform(0, pw)
            y = _periodic(pw, h * 0.54, r2, x)
            s = rng.uniform(10, 18) * u
            p.blobs([("e", (x - s, y - s * 1.5, x + s, y + s * 0.5)),
                     ("e", (x - s * 1.6, y - s * 0.9, x + s * 0.2, y + s * 0.6))], (150, 196, 140), (118, 170, 120),
                    ow=1.5, light=0.35, dark=0.15)
        p.blobs([("p", _ridge(pw, h * 0.54, r2, h * 0.8))],
                (176, 210, 160), (148, 192, 140), ow=2.0, wrap=False, light=0.3, rim=10)
        self.add(p, factor=0.12, haze=0.42)
        # mid meadow + tree row
        p = Painter(pw, h, u)
        mid = [(20, 2, 1.0), (10, 5, 2.4), (5, 11, 0.7)]
        for i in range(int(pw / (120 * u))):
            x = (i + rng.uniform(0.1, 0.9)) * 120 * u
            y = _periodic(pw, h * 0.61, mid, x) + 6 * u
            s = rng.uniform(26, 40) * u
            p.blobs([("r", (x - 3 * u, y - s * 0.7, x + 3 * u, y + 2), 2)], (150, 110, 85), (120, 85, 65), ow=1.6, rim=2)
            _tree_canopy(p, x, y - s * 1.1, s * 0.75, rng, (162, 208, 140), (110, 168, 112))
        p.blobs([("p", _ridge(pw, h * 0.61, mid, h * 0.85))], (170, 214, 140), (140, 196, 122), ow=2.2,
                wrap=False, light=0.35, rim=8)
        self.add(p, factor=0.32, haze=0.18)
        # big trees layer (canopies sway in after-hook)
        p = Painter(pw, h, u)
        em = self.em(p)
        self.canopies = []
        n = 4
        for i in range(n):
            x = (i + rng.uniform(0.25, 0.75)) * pw / n
            s = rng.uniform(80, 105) * u
            base = gy + 8 * u
            tw = s * 0.16
            p.blobs([("p", [(x - tw, base), (x - tw * 0.55, base - s * 1.5), (x - tw * 1.3, base - s * 2.1),
                            (x - tw * 0.2, base - s * 1.75), (x + tw * 0.9, base - s * 2.2),
                            (x + tw * 0.55, base - s * 1.45), (x + tw, base), (x + tw * 1.6, base + 4 * u),
                            (x - tw * 1.6, base + 4 * u)])],
                    (176, 124, 92), (132, 90, 66), ow=2.6, light=0.25, dark=0.3, rim=6)
            for k in range(2):
                bx = x + (-1 if k else 1) * s * rng.uniform(0.5, 0.8)
                bs = s * rng.uniform(0.26, 0.34)
                p.blobs([("e", (bx - bs, base - bs * 1.2, bx + bs, base + 4 * u)),
                         ("e", (bx - bs * 1.5, base - bs * 0.8, bx + bs * 0.2, base + 4 * u))],
                        (150, 204, 120), (102, 160, 98), ow=2.4, light=0.4, rim=6)
            cs = s * 1.35
            sw, sh = int(cs * 2.6), int(cs * 2.2)
            rr = np.random.default_rng(1000 + i)
            spr = _sprite(lambda q: _tree_canopy(q, sw / 2, sh / 2, cs, rr), sw, sh, u, self.mood, seed=i)
            self.canopies.append((spr, x - sw / 2, base - s * 1.95 - sh / 2, i * 1.7))
        for i in range(2):
            x = (i + 0.08) * pw / 2 + pw / 4
            _lamp_post(p, em, x, gy + 10 * u, gy - 210 * u, u, glow=self.lights > 0.2)
        self.add(p, factor=0.62, emissive=em, after=self._sway_trees)
        # ground
        p = Painter(pw, h, u)
        p.blobs([("p", _ridge(pw, gy, [(5, 4, 0.2), (3, 9, 1.1)], h + 40))], (158, 210, 120), (112, 176, 98),
                ow=2.6, wrap=False, light=0.3, rim=10, gy=(gy, h))
        path = _ridge(pw, h * 0.775, [(6, 3, 0.5), (4, 7, 2.0)], 0)[:-2]
        path2 = [(x, _periodic(pw, h * 0.93, [(5, 2, 1.7), (3, 6, 0.1)], x)) for x, _ in reversed(path)]
        p.blobs([("p", path + path2)], (240, 222, 186), (222, 196, 158), ow=2.4, wrap=False, light=0.4, rim=6)
        for i in range(int(pw / (40 * u))):  # pebbles
            x = rng.uniform(0, pw)
            y = rng.uniform(h * 0.80, h * 0.92)
            r = rng.uniform(3, 6) * u
            p.blobs([("e", (x - r * 1.4, y - r, x + r * 1.4, y + r))], (214, 196, 170), (184, 166, 142), ow=1.4,
                    light=0.4, dark=0.0, rim=2)
        flower_cols = [(250, 170, 190), (255, 230, 150), (200, 180, 250), (255, 255, 255)]
        for i in range(int(pw / (24 * u))):  # grass tufts and flowers along edges
            x = rng.uniform(0, pw)
            band = rng.random()
            y = gy + rng.uniform(4, 18) * u if band < 0.7 else h * rng.uniform(0.95, 0.99)
            s = rng.uniform(7, 13) * u
            p.blobs([("p", [(x - s, y), (x - s * 0.6, y - s * 1.3), (x - s * 0.2, y - s * 0.4), (x, y - s * 1.6),
                            (x + s * 0.25, y - s * 0.4), (x + s * 0.7, y - s * 1.2), (x + s, y)])],
                    (130, 196, 104), (98, 160, 90), ow=1.5, light=0.3, dark=0, rim=2)
            if rng.random() < 0.35:
                c = flower_cols[int(rng.integers(len(flower_cols)))]
                r = 4.5 * u
                p.blobs([("e", (x - r, y - s * 1.6 - r, x + r, y - s * 1.6 + r))], c, _mix(c, (200, 120, 140), 0.3),
                        ow=1.4, light=0.3, dark=0, rim=1)
        puds = []
        if self.rainy:
            for i in range(5):
                x = (i + rng.uniform(0.2, 0.8)) * pw / 5
                y = h * rng.uniform(0.83, 0.9)
                a, b = rng.uniform(70, 120) * u, rng.uniform(9, 14) * u
                p.blobs([("e", (x - a, y - b, x + a, y + b))], (150, 160, 175), (130, 140, 158), ow=1.6, light=0, dark=0)
                puds.append((x - a + 2, y - b + 2, x + a - 2, y + b - 2))
        li = self.add(p, factor=1.0, grain=1.2)
        if puds:
            self.puddles(li, puds, gy - 6 * u)
        self.splash = (h * 0.76, h * 0.97, li)
        # foreground leafy clumps along the bottom edge
        p = Painter(pw, h, u)
        for i in range(5):
            x = (i + rng.uniform(0.1, 0.9)) * pw / 5
            s = rng.uniform(22, 34) * u
            sh = [("e", (x - s * 1.6, h - s * 0.7, x - s * 0.1, h + s)), ("e", (x - s * 0.7, h - s * 1.2, x + s * 0.8, h + s)),
                  ("e", (x + s * 0.2, h - s * 0.8, x + s * 1.7, h + s))]
            p.blobs(sh, (136, 198, 110), (90, 150, 84), ow=2.4, light=0.4, dark=0.25, rim=5)
            for q in range(3):
                fx, fy = x + rng.uniform(-1.2, 1.2) * s, h - rng.uniform(0.3, 0.9) * s
                c = flower_cols[int(rng.integers(len(flower_cols)))]
                r = 5 * u
                p.blobs([("e", (fx - r, fy - r, fx + r, fy + r))], c, ow=1.4, light=0.3, dark=0, rim=1)
        self.add(p, factor=1.4)

    def _sway_trees(self, fr, t, off, dy):
        for spr, lx, ly, ph in self.canopies:
            a = 0.022 * math.sin(t * 1.1 + ph) + 0.01 * math.sin(t * 2.7 + ph * 2)
            if self.rainy:
                a *= 1.8
            _place(fr, _sway(spr, a), lx, ly + dy, off, self.pw, self.w)


# --------------------------------------------------------------------------
# CITY skyline helpers (shared by city, metro, rooftop, clinic windows)
# --------------------------------------------------------------------------
PASTEL_BLD = [(244, 196, 186), (250, 222, 170), (190, 214, 236), (204, 226, 196), (226, 204, 236),
              (246, 234, 214), (180, 200, 220), (240, 206, 214)]


def _building(p, em, x, y_base, bw, bh, col, rng, lit, detail=True, roof=True, cols=None, rows=None):
    top = y_base - bh
    sh = [("r", (x, top, x + bw, y_base + 6), 3)]
    if roof and rng.random() < 0.5:
        sh.append(("r", (x - 4, top - 8, x + bw + 4, top + 4), 2))
    p.blobs(sh, _mix(col, (255, 255, 255), 0.15), _mix(col, (120, 110, 150), 0.25), ow=2.2, light=0.3, dark=0.25,
            rim=6, gy=(top, y_base))
    if detail:
        cols = cols or max(2, int(bw / (30 * p.u)))
        rows = rows or max(2, int(bh / (42 * p.u)))
        _window_grid(p, em, x + bw * 0.08, top + bh * 0.06, x + bw * 0.92, y_base - bh * 0.06, cols, rows, rng, lit,
                     gap=0.34)
    return top


def _skyline(p, em, pw, base, hmin, hmax, wmin, wmax, rng, lit, detail=True, palette=PASTEL_BLD, gap=(0, 14)):
    x = 0.0
    tops = []
    while x < pw:
        bw = rng.uniform(wmin, wmax)
        bh = rng.uniform(hmin, hmax)
        col = palette[int(rng.integers(len(palette)))]
        top = _building(p, em, x, base, bw, bh, col, rng, lit, detail)
        tops.append((x, bw, top))
        x += bw + rng.uniform(*gap) * p.u
    return tops


# --------------------------------------------------------------------------
# CITY
# --------------------------------------------------------------------------
class City(_Scene):
    seed = 21
    horizon = 0.6
    sun = (0.22, 0.18)

    def build(self):
        w, h, u, pw, rng = self.w, self.h, self.u, self.pw, self.rng
        self.make_clouds(5, 0.03, 0.22, 36, 60, speed=9)
        gy = int(h * 0.70)
        L = self.lights
        # far silhouettes
        p = Painter(pw, h, u)
        em = self.em(p)
        _skyline(p, em, pw, h * 0.62, h * 0.18, h * 0.42, 50 * u, 110 * u, rng, L * 0.7,
                 palette=[(196, 206, 228), (210, 204, 228), (190, 214, 220)])
        self.add(p, factor=0.15, haze=0.42, emissive=em)
        # mid buildings
        p = Painter(pw, h, u)
        em = self.em(p)
        for x, bw, top in _skyline(p, em, pw, h * 0.66, h * 0.22, h * 0.40, 90 * u, 170 * u, rng, L, gap=(10, 40)):
            if rng.random() < 0.4:  # rooftop water tank / antenna
                tx = x + bw * rng.uniform(0.2, 0.6)
                p.blobs([("r", (tx, top - 26 * u, tx + 26 * u, top + 2), 4)], (120, 130, 150), (90, 98, 118), ow=2)
            elif rng.random() < 0.5:
                ax = x + bw * 0.7
                p.line([(ax, top), (ax, top - 34 * u)], OUT, 2.2)
                p.line([(ax - 8 * u, top - 24 * u), (ax + 8 * u, top - 24 * u)], OUT, 2)
        self.add(p, factor=0.38, haze=0.16, emissive=em)
        # near shopfronts
        p = Painter(pw, h, u)
        em = self.em(p)
        x = 0.0
        awn = [((240, 120, 120), (255, 245, 235)), ((110, 170, 210), (250, 250, 245)),
               ((120, 190, 150), (250, 250, 240)), ((250, 190, 90), (255, 250, 236))]
        while x < pw - 60 * u:
            bw = min(rng.uniform(190, 270) * u, pw - x)
            bh = rng.uniform(0.32, 0.5) * h
            col = PASTEL_BLD[int(rng.integers(len(PASTEL_BLD)))]
            top = gy - bh
            p.blobs([("r", (x, top, x + bw, gy + 4), 4)], _mix(col, (255, 255, 255), 0.2), _mix(col, (150, 120, 140), 0.18),
                    ow=2.8, light=0.3, dark=0.25, rim=8, gy=(top, gy))
            p.blobs([("r", (x - 5 * u, top - 10 * u, x + bw + 5 * u, top + 4 * u), 3)], _mix(col, (110, 90, 100), 0.35),
                    _mix(col, (80, 60, 80), 0.45), ow=2.4)
            # upper windows with balconies
            for j in range(2):
                wy = top + bh * (0.12 + 0.24 * j)
                if wy > gy - h * 0.2:
                    break
                for k in range(2):
                    wx = x + bw * (0.16 + 0.42 * k)
                    box = (wx, wy, wx + bw * 0.26, wy + h * 0.075)
                    p.blobs([("r", box, 6)], (205, 228, 242), (160, 190, 220), ow=2.2, light=0.5, dark=0, rim=3)
                    p.line([(box[0] + (box[2] - box[0]) / 2, box[1]), (box[0] + (box[2] - box[0]) / 2, box[3])], OUT, 1.6)
                    if em is not None and rng.random() < self.lights:
                        em.soft([("r", box, 6)], GLOW, 255)
                    p.blobs([("r", (box[0] - 6 * u, box[3] - 2, box[2] + 6 * u, box[3] + 7 * u), 2)],
                            (236, 236, 240), (200, 200, 214), ow=2)
            # shop window + door
            sy = gy - h * 0.17
            c1, c2 = awn[int(rng.integers(len(awn)))]
            p.blobs([("r", (x + bw * 0.08, sy, x + bw * 0.62, gy - 4 * u), 4)], (214, 234, 244), (176, 206, 226), ow=2.4,
                    light=0.5, dark=0.1, rim=4)
            if em is not None and rng.random() < 0.9:
                em.soft([("r", (x + bw * 0.08, sy, x + bw * 0.62, gy - 4 * u), 4)], (255, 220, 150), 150)
                em.soft([("r", (x + bw * 0.08, gy - h * 0.045, x + bw * 0.62, gy - 4 * u), 3)], (180, 120, 90), 120)
            p.blobs([("r", (x + bw * 0.7, sy - 4 * u, x + bw * 0.9, gy), 3)], (170, 120, 96), (130, 90, 72), ow=2.4)
            p.blobs([("e", (x + bw * 0.86 - 3 * u, (sy + gy) / 2, x + bw * 0.86 + 3 * u, (sy + gy) / 2 + 6 * u))],
                    (250, 220, 120), ow=1.2, light=0, dark=0)
            # striped awning
            ay = sy - 8 * u
            n = 7
            sx0, sx1 = x + bw * 0.04, x + bw * 0.66
            stw = (sx1 - sx0) / n
            p.blobs([("p", [(sx0 + 8 * u, ay - 26 * u), (sx1 - 8 * u, ay - 26 * u), (sx1, ay), (sx0, ay)])], c2, ow=2.4,
                    light=0.2, dark=0.1)
            for k in range(0, n, 2):
                a, b = sx0 + k * stw, sx0 + (k + 1) * stw
                p.fill([("p", [(a + 8 * u * (1 - k / n) + 0.0, ay - 26 * u), (b + 8 * u * (1 - (k + 1) / n) - 0, ay - 26 * u),
                                (b, ay), (a, ay)])], c1)
            for k in range(n):  # scalloped edge
                a = sx0 + k * stw
                p.blobs([("e", (a, ay - 6 * u, a + stw, ay + 7 * u))], c1 if k % 2 == 0 else c2, ow=2.0, light=0, dark=0.1)
            # sign
            p.blobs([("r", (x + bw * 0.16, top + bh * 0.62 - 14 * u, x + bw * 0.55, top + bh * 0.62 + 14 * u), 8)]
                    if bh > h * 0.4 else [("r", (x + bw * 0.7, sy - 34 * u, x + bw * 0.92, sy - 12 * u), 6)],
                    (255, 250, 236), (240, 226, 200), ow=2.2)
            x += bw + rng.uniform(6, 30) * u
        self.add(p, factor=0.7, emissive=em)
        # street
        p = Painter(pw, h, u)
        em = self.em(p)
        p.blobs([("r", (-40, gy, pw + 40, h * 0.9), 0)], (226, 214, 204), (200, 188, 182), ow=2.6, wrap=False,
                light=0.25, dark=0.0)
        for i in range(int(pw / (90 * u))):
            x = i * 90 * u
            p.line([(x, gy + 4 * u), (x - 30 * u, h * 0.9)], (176, 160, 156), 1.6)
        p.line([(-40, h * 0.80), (pw + 40, h * 0.80)], (186, 172, 168), 1.6, wrap=False)
        p.blobs([("r", (-40, h * 0.9, pw + 40, h * 0.925), 0)], (190, 186, 190), (160, 156, 166), ow=2.4, wrap=False)
        p.blobs([("r", (-40, h * 0.925, pw + 40, h + 30), 0)], (120, 124, 140), (96, 100, 118), ow=2.4, wrap=False,
                light=0.15, dark=0)
        for i in range(int(pw / (160 * u))):
            x = i * 160 * u
            p.blobs([("r", (x, h * 0.965, x + 80 * u, h * 0.975), 3)], (250, 244, 220), ow=1.2, light=0, dark=0)
        for i in range(4):
            x = (i + 0.5) * pw / 4
            _lamp_post(p, em, x, gy + 14 * u, gy - 250 * u, u, glow=self.lights > 0.2)
            tx = x + pw / 8
            p.blobs([("r", (tx - 26 * u, gy - 8 * u, tx + 26 * u, gy + 24 * u), 6)], (200, 140, 110), (160, 105, 85), ow=2.4)
            _tree_canopy(p, tx, gy - 62 * u, 46 * u, rng, (168, 214, 128), (100, 164, 100))
            p.blobs([("r", (tx - 4 * u, gy - 30 * u, tx + 4 * u, gy - 6 * u), 2)], (150, 105, 80), ow=1.6)
        puds = []
        if self.rainy:
            for i in range(6):
                x = (i + rng.uniform(0.2, 0.8)) * pw / 6
                y = h * rng.uniform(0.82, 0.88)
                a, b = rng.uniform(60, 110) * u, rng.uniform(8, 12) * u
                p.blobs([("e", (x - a, y - b, x + a, y + b))], (160, 160, 172), (140, 142, 156), ow=1.4, light=0, dark=0)
                puds.append((x - a + 2, y - b + 2, x + a - 2, y + b - 2))
        li = self.add(p, factor=1.0, emissive=em, grain=1.2)
        if puds:
            self.puddles(li, puds, gy - 4 * u)
        self.splash = (h * 0.74, h * 0.99, li)


# --------------------------------------------------------------------------
# METRO
# --------------------------------------------------------------------------
class Metro(_Scene):
    seed = 31
    horizon = 0.5
    sun = (0.7, 0.16)
    rain_slant = -0.9

    def build(self):
        w, h, u, pw, rng = self.w, self.h, self.u, self.pw, self.rng
        self.make_clouds(4, 0.02, 0.16, 30, 50, speed=40)
        L = self.lights
        # outside: far skyline, near skyline, viaduct posts — all scroll (train moving)
        p = Painter(pw, h, u)
        em = self.em(p)
        _skyline(p, em, pw, h * 0.47, h * 0.05, h * 0.15, 40 * u, 90 * u, rng, L * 0.8,
                 palette=[(200, 208, 230), (214, 206, 230), (196, 218, 222), (226, 214, 222)], gap=(0, 30))
        self.add(p, factor=0.1, scroll=-60, haze=0.38, emissive=em)
        p = Painter(pw, h, u)
        em = self.em(p)
        _skyline(p, em, pw, h * 0.5, h * 0.06, h * 0.17, 70 * u, 140 * u, rng, L, gap=(60, 200))
        for i in range(14):
            x = (i + rng.uniform(0.2, 0.8)) * pw / 14
            _tree_canopy(p, x, h * 0.45, rng.uniform(22, 34) * u, rng, (160, 208, 132), (104, 166, 104))
        self.add(p, factor=0.25, scroll=-260, haze=0.12, emissive=em)
        p = Painter(pw, h, u)
        for i in range(3):
            x = (i + 0.5) * pw / 3
            p.blobs([("r", (x, h * 0.1, x + 22 * u, h * 0.6), 2)], (190, 190, 200), (150, 150, 166), ow=2.2)
        p.line([(-10, h * 0.16), (pw + 10, h * 0.16)], (80, 80, 96), 2.2, wrap=False)
        self.add(p, factor=0.5, scroll=-900, haze=0.05)
        self.rain_at = len(self.layers)
        # interior (sways gently)
        night_int = 0.7 if self.night else 0.55
        p = Painter(pw, h, u)
        em = self.em(p)
        top_wall, win0, win1, seat0, floor_y = h * 0.11, h * 0.2, h * 0.47, h * 0.56, h * 0.70
        p.blobs([("r", (-40, -40, pw + 40, floor_y + 4), 0)], (250, 242, 222), (232, 222, 204), ow=0, wrap=False,
                light=0, dark=0)
        p.soft([("r", (-40, top_wall, pw + 40, top_wall + 40 * u), 0)], (255, 250, 230), 120, blur=14, wrap=False)
        p.blobs([("r", (-40, -40, pw + 40, top_wall), 0)], (232, 236, 240), (214, 220, 228), ow=2.6, wrap=False,
                light=0, dark=0.2)
        for k in range(2):  # module per screen width
            m0 = k * w
            # ceiling light strip
            p.blobs([("r", (m0 + w * 0.08, top_wall * 0.35, m0 + w * 0.92, top_wall * 0.7), 8)], (255, 252, 236),
                    (246, 240, 220), ow=2.0, light=0.0, dark=0.1)
            # door at module start
            dx0, dx1 = m0 - w * 0.08, m0 + w * 0.08
            p.blobs([("r", (dx0, top_wall + 8 * u, dx1, floor_y + 2), 6)], (206, 214, 222), (178, 188, 200), ow=2.6,
                    light=0.3, rim=6)
            p.line([(m0, top_wall + 8 * u), (m0, floor_y)], OUT, 2.4)
            for s in (-1, 1):
                gx0 = m0 + s * w * 0.018
                gx1 = m0 + s * w * 0.065
                p.fill([("r", (min(gx0, gx1), h * 0.17, max(gx0, gx1), h * 0.46), 10)], (0, 0, 0, 0))
                p.blobs([("r", (min(gx0, gx1), h * 0.17, max(gx0, gx1), h * 0.46), 10)], (0, 0, 0), ow=2.2, light=0,
                        dark=0)
                p.fill([("r", (min(gx0, gx1), h * 0.17, max(gx0, gx1), h * 0.46), 10)], (0, 0, 0, 0))
            # route map strip above door
            p.blobs([("r", (m0 - w * 0.07, top_wall + 14 * u, m0 + w * 0.07, top_wall + 30 * u), 4)], (255, 255, 255),
                    ow=1.8, light=0, dark=0)
            # ads above windows
            adc = [(255, 196, 206), (196, 226, 255), (255, 232, 170), (206, 240, 210)]
            # windows
            for j, (a, b) in enumerate(((0.14, 0.46), (0.54, 0.86))):
                x0, x1 = m0 + w * a, m0 + w * b
                ac = adc[(2 * k + j) % 4]
                p.blobs([("r", (x0 + 20 * u, h * 0.125, x1 - 20 * u, h * 0.18), 6)], ac, _mix(ac, (200, 160, 180), 0.3),
                        ow=2.0, light=0.3)
                p.blobs([("e", (x0 + 30 * u, h * 0.133, x0 + 52 * u, h * 0.172))], (255, 255, 255), ow=1.4, light=0, dark=0)
                for q in range(3):
                    p.line([(x0 + (64 + q * 6) * u, h * (0.142 + q * 0.011)),
                            (x1 - (40 + 40 * q) * u, h * (0.142 + q * 0.011))], _mix(ac, OUT, 0.6), 2.0)
                p.blobs([("r", (x0 - 10 * u, win0 - 10 * u, x1 + 10 * u, win1 + 10 * u), 26)], (214, 220, 228),
                        (190, 198, 210), ow=2.6, light=0.4, dark=0.2, rim=5)
                p.fill([("r", (x0, win0, x1, win1), 18)], (0, 0, 0, 0))
                p.line([((x0 + x1) / 2, win0), ((x0 + x1) / 2, win1)], (190, 198, 210), 6)
                # glass sheen
                for gx, gw_ in ((0.12, 0.08), (0.24, 0.03), (0.62, 0.07)):
                    sx = x0 + (x1 - x0) * gx
                    p.soft([("p", [(sx, win1), (sx + (x1 - x0) * gw_, win1), (sx + (x1 - x0) * gw_ + 70 * u, win0),
                                    (sx + 70 * u, win0)])], (255, 255, 255), 40 if not self.night else 60)
                # seats below window
                for q in range(3):
                    sx0 = x0 + (x1 - x0) * q / 3 + 4 * u
                    sx1 = x0 + (x1 - x0) * (q + 1) / 3 - 4 * u
                    p.blobs([("r", (sx0, seat0 - h * 0.085, sx1, seat0 + 6 * u), 14)], (120, 196, 214), (84, 160, 186),
                            ow=2.4, light=0.4, dark=0.25, rim=6)
                    p.blobs([("r", (sx0 - 2 * u, seat0, sx1 + 2 * u, seat0 + h * 0.035), 10)], (110, 186, 206),
                            (78, 150, 176), ow=2.4, light=0.4, dark=0.25, rim=5)
                p.blobs([("r", (x0, seat0 + h * 0.035, x1, floor_y + 2), 4)], (196, 202, 212), (170, 176, 190), ow=2.4,
                        light=0.2, dark=0.2)
        p.blobs([("r", (-40, floor_y, pw + 40, h + 40), 0)], (196, 190, 200), (156, 150, 170), ow=2.6, wrap=False,
                light=0.0, dark=0.0, gy=(floor_y, h))
        p.soft([("r", (-40, h * 0.77, pw + 40, h * 0.83), 0)], (255, 255, 255), 40, blur=8, wrap=False)
        for i in range(int(pw / (14 * u))):  # anti-slip dots
            x = i * 14 * u + (7 * u if i % 2 else 0)
            for yy in (0.895, 0.93, 0.965):
                p.fill([("e", (x - 1.6 * u, h * yy - 1.6 * u, x + 1.6 * u, h * yy + 1.6 * u))], (140, 136, 156))
        p.soft([("r", (-40, floor_y, pw + 40, floor_y + 12 * u), 0)], (60, 50, 70), 60, blur=4, wrap=False)
        p.blobs([("r", (-40, h * 0.86, pw + 40, h * 0.875), 0)], (232, 214, 120), ow=1.4, wrap=False, light=0, dark=0)
        if self.rainy:  # droplets on glass + wet floor patch
            for k in range(2):
                for a, b in ((0.14, 0.46), (0.54, 0.86)):
                    for _ in range(26):
                        x = k * w + w * rng.uniform(a + 0.01, b - 0.01)
                        y = rng.uniform(win0 + 6, win1 - 6)
                        r = rng.uniform(1.5, 3.5) * u
                        p.soft([("e", (x - r, y - r, x + r, y + r * 1.4))], (235, 242, 255), 120)
        if em is not None or True:
            em2 = Painter(pw, h, u)
            for k in range(2):
                m0 = k * w
                em2.soft([("r", (m0 + w * 0.1, top_wall * 0.4, m0 + w * 0.9, top_wall * 0.65), 6)], (255, 250, 228),
                         200 if (self.night or self.rainy) else 120)
            em = em2
        li = self.add(p, factor=1.0, dy=lambda t: 2.2 * math.sin(t * 2.3) + 0.8 * math.sin(t * 5.1), emissive=em,
                      interior=night_int)
        # poles + grab rail + handles
        p = Painter(pw, h, u)
        p.blobs([("r", (-40, h * 0.085, pw + 40, h * 0.098), 4)], (236, 206, 92), (210, 176, 70), ow=2.2, wrap=False,
                light=0.4, dark=0.2)
        for k in range(2):
            for xx in (k * w + w * 0.11, k * w + w * 0.89):
                p.blobs([("r", (xx - 7 * u, h * 0.085, xx + 7 * u, h * 0.86), 6)], (242, 214, 100), (214, 180, 72), ow=2.4,
                        light=0.45, dark=0.25, rim=3)
                p.blobs([("r", (xx - 14 * u, h * 0.85, xx + 14 * u, h * 0.87), 4)], (170, 176, 190), ow=2.0)
        self.handles = []
        hs = int(70 * u)
        def handle(q):
            q.line([(hs / 2, 0), (hs / 2, hs * 0.9)], (220, 222, 230), 4.0)
            q.line([(hs / 2, 0), (hs / 2, hs * 0.9)], OUT, 1.2)
            q.blobs([("r", (hs / 2 - 14 * u, hs * 0.9, hs / 2 + 14 * u, hs * 1.32), 10)], (250, 250, 250), (220, 222, 232),
                    ow=2.2, light=0.3, dark=0.2)
            q.fill([("r", (hs / 2 - 8 * u, hs * 0.98, hs / 2 + 8 * u, hs * 1.24), 6)], (0, 0, 0, 0))
        spr = _sprite(handle, hs, int(hs * 1.4), u, self.mood, interior=night_int)
        for i in range(10):
            x = (i + 0.5) * pw / 10
            if min(abs(x % w - w * 0.11), abs(x % w - w * 0.89)) > 30 * u:
                self.handles.append((spr, x - hs / 2, h * 0.095, i * 0.9))
        self.add(p, factor=1.08, dy=lambda t: 2.2 * math.sin(t * 2.3) + 0.8 * math.sin(t * 5.1),
                 after=self._swing, interior=night_int)

    def _swing(self, fr, t, off, dy):
        for spr, lx, ly, ph in self.handles:
            a = 7 * math.sin(t * 2.3 + 0.6 + ph * 0.15) + 2 * math.sin(t * 4.0 + ph)
            _place(fr, _swing(spr, a, (spr.width / 2, 0)), lx, ly + dy, off, self.pw, self.w)


# --------------------------------------------------------------------------
# CLINIC
# --------------------------------------------------------------------------
def _tooth(p, cx, cy, s, face=True, rng=None):
    sh = [("e", (cx - s * 0.62, cy - s * 0.62, cx + s * 0.05, cy + s * 0.15)),
          ("e", (cx - s * 0.05, cy - s * 0.62, cx + s * 0.62, cy + s * 0.15)),
          ("r", (cx - s * 0.58, cy - s * 0.35, cx + s * 0.58, cy + s * 0.3), s * 0.2),
          ("p", [(cx - s * 0.58, cy + s * 0.1), (cx - s * 0.05, cy + s * 0.1), (cx - s * 0.2, cy + s * 0.86),
                 (cx - s * 0.42, cy + s * 0.86)]),
          ("p", [(cx + s * 0.58, cy + s * 0.1), (cx + s * 0.05, cy + s * 0.1), (cx + s * 0.2, cy + s * 0.86),
                 (cx + s * 0.42, cy + s * 0.86)])]
    p.blobs(sh, (255, 255, 255), (226, 234, 246), ow=2.2, light=0.2, dark=0.25, rim=max(2, s * 0.08))
    if face:
        for ex in (-0.2, 0.2):
            r = s * 0.06
            p.fill([("e", (cx + ex * s - r, cy - s * 0.12 - r * 1.3, cx + ex * s + r, cy - s * 0.12 + r * 1.3))], OUT, wrap=False)
        p.d.arc(((cx - s * 0.16) * SS, (cy - s * 0.12) * SS, (cx + s * 0.16) * SS, (cy + s * 0.12) * SS), 20, 160,
                fill=OUT + (255,), width=max(1, int(2.2 * p.u * SS)))
        for ex in (-0.34, 0.34):
            r = s * 0.08
            p.soft([("e", (cx + ex * s - r, cy - r * 0.6, cx + ex * s + r, cy + r * 0.6))], (255, 160, 170), 170, wrap=False)


class Clinic(_Scene):
    seed = 41
    horizon = 0.55
    sun = (0.28, 0.2)
    interior = True

    def build(self):
        w, h, u, pw, rng = self.w, self.h, self.u, self.pw, self.rng
        self.make_clouds(3, 0.08, 0.25, 26, 40, speed=5, factor=0.2)
        L = self.lights
        # outside view: gardens & houses
        p = Painter(pw, h, u)
        em = self.em(p)
        _skyline(p, em, pw, h * 0.52, h * 0.08, h * 0.2, 60 * u, 120 * u, rng, L, gap=(30, 90))
        for i in range(12):
            x = (i + rng.uniform(0.1, 0.9)) * pw / 12
            _tree_canopy(p, x, h * 0.46, rng.uniform(28, 44) * u, rng, (162, 210, 136), (104, 166, 104))
        p.blobs([("r", (-40, h * 0.5, pw + 40, h * 0.7), 0)], (166, 210, 140), (140, 190, 124), ow=2, wrap=False)
        self.add(p, factor=0.25, haze=0.15, emissive=em)
        self.rain_at = len(self.layers)
        IN = 0.72
        floor_y = h * 0.72
        p = Painter(pw, h, u)
        em = self.em(p)
        wall_t, wall_b = (206, 236, 226), (188, 224, 214)
        p.blobs([("r", (-40, -40, pw + 40, floor_y), 0)], wall_t, wall_b, ow=0, wrap=False, light=0, dark=0)
        p.blobs([("r", (-40, h * 0.5, pw + 40, floor_y), 0)], (236, 246, 248), (214, 232, 238), ow=2.4, wrap=False,
                light=0, dark=0)
        for i in range(int(pw / (36 * u)) + 1):  # wainscot tiles
            p.line([(i * 36 * u, h * 0.5), (i * 36 * u, floor_y)], (196, 216, 224), 1.4)
        for j in range(1, 4):
            p.line([(-40, h * 0.5 + j * (floor_y - h * 0.5) / 4), (pw + 40, h * 0.5 + j * (floor_y - h * 0.5) / 4)],
                   (196, 216, 224), 1.4, wrap=False)
        p.blobs([("r", (-40, h * 0.49, pw + 40, h * 0.51), 0)], (150, 206, 196), ow=2.0, wrap=False, light=0, dark=0)
        self.clocks = []
        for k in range(2):
            m0 = k * w
            # window
            x0, x1, y0, y1 = m0 + w * 0.06, m0 + w * 0.34, h * 0.1, h * 0.44
            p.blobs([("r", (x0 - 14 * u, y0 - 14 * u, x1 + 14 * u, y1 + 14 * u), 12)], (255, 255, 255), (232, 236, 240),
                    ow=2.6, light=0.3, dark=0.2, rim=5)
            p.fill([("r", (x0, y0, x1, y1), 6)], (0, 0, 0, 0))
            p.blobs([("r", ((x0 + x1) / 2 - 5 * u, y0, (x0 + x1) / 2 + 5 * u, y1), 2),
                     ("r", (x0, (y0 + y1) / 2 - 5 * u, x1, (y0 + y1) / 2 + 5 * u), 2)], (255, 255, 255), ow=2.0, light=0,
                    dark=0.15)
            p.blobs([("r", (x0 - 24 * u, y1 + 10 * u, x1 + 24 * u, y1 + 24 * u), 4)], (255, 255, 255), (230, 234, 240),
                    ow=2.4)
            for sx in (0.15, 0.6):  # glass sheen
                a = x0 + (x1 - x0) * sx
                p.soft([("p", [(a, y1), (a + 30 * u, y1), (a + 90 * u, y0), (a + 60 * u, y0)])], (255, 255, 255), 45)
            # curtain
            for side, cx in ((-1, x0 - 6 * u), (1, x1 + 6 * u)):
                cw = 34 * u
                pts = [(cx - cw * (side < 0), y0 - 20 * u), (cx + cw * (side > 0), y0 - 20 * u)]
                pts += [(cx + cw * (side > 0) - side * 8 * u, y1 + 30 * u), (cx - cw * (side < 0) - side * 8 * u, y1 + 30 * u)]
                p.blobs([("p", pts)], (255, 214, 214), (240, 186, 192), ow=2.2, light=0.3, dark=0.25)
            p.blobs([("r", (x0 - 50 * u, y0 - 26 * u, x1 + 50 * u, y0 - 18 * u), 3)], (200, 170, 140), ow=2)
            # posters
            pc = [((255, 222, 228), (250, 200, 210)), ((222, 236, 255), (200, 220, 250)), ((255, 240, 200), (250, 226, 170))]
            px = m0 + w * (0.42 if k == 0 else 0.4)
            for q, (pxx, py, pwid, phh) in enumerate([(px, h * 0.12, w * 0.13, h * 0.26), (px + w * 0.16, h * 0.16, w * 0.1, h * 0.17)]):
                c1, c2 = pc[(q + k) % 3]
                p.blobs([("r", (pxx - 6 * u, py - 6 * u, pxx + pwid + 6 * u, py + phh + 6 * u), 6)], (200, 160, 120),
                        (170, 130, 96), ow=2.4)
                p.blobs([("r", (pxx, py, pxx + pwid, py + phh), 4)], c1, c2, ow=1.6, light=0.2, dark=0)
                if q == 0:
                    _tooth(p, pxx + pwid / 2, py + phh * 0.42, min(pwid, phh) * 0.36)
                    for z in range(2):
                        p.line([(pxx + pwid * 0.2, py + phh * (0.82 + z * 0.08)), (pxx + pwid * (0.8 - z * 0.2),
                                py + phh * (0.82 + z * 0.08))], _mix(c2, OUT, 0.5), 2.4)
                else:  # toothbrush + sparkle
                    bx, by = pxx + pwid * 0.25, py + phh * 0.8
                    ex, ey = pxx + pwid * 0.78, py + phh * 0.25
                    p.line([(bx, by), (ex, ey)], (120, 170, 230), 7)
                    p.line([(bx, by), (ex, ey)], OUT, 1.2)
                    p.blobs([("r", (ex - 10 * u, ey - 16 * u, ex + 8 * u, ey + 2 * u), 3)], (255, 255, 255), ow=1.6)
                    for sxp, syp, ss in ((0.25, 0.28, 9), (0.7, 0.62, 6)):
                        cx, cy, r = pxx + pwid * sxp, py + phh * syp, ss * u
                        p.blobs([("p", [(cx, cy - r), (cx + r * 0.3, cy - r * 0.3), (cx + r, cy), (cx + r * 0.3, cy + r * 0.3),
                                        (cx, cy + r), (cx - r * 0.3, cy + r * 0.3), (cx - r, cy), (cx - r * 0.3, cy - r * 0.3)])],
                                (255, 236, 150), ow=1.4, light=0, dark=0)
            # clock
            cx, cy, cr = m0 + w * 0.5, h * 0.08 if k == 0 else h * 0.39, 30 * u
            if k == 0:
                cx = m0 + w * 0.63
            p.blobs([("e", (cx - cr, cy - cr, cx + cr, cy + cr))], (255, 255, 255), (232, 236, 244), ow=3.0, light=0.2,
                    dark=0.25)
            for a in range(12):
                aa = a / 12 * 2 * math.pi
                p.line([(cx + math.cos(aa) * cr * 0.75, cy + math.sin(aa) * cr * 0.75),
                        (cx + math.cos(aa) * cr * 0.88, cy + math.sin(aa) * cr * 0.88)], OUT, 1.6)
            self.clocks.append((cx, cy, cr))
            # cabinet + sink + shelf with jars on module 1, shelf on module 0
            if k == 1:
                c0, c1 = m0 + w * 0.6, m0 + w * 0.9
                p.blobs([("r", (c0, h * 0.52, c1, floor_y + 2), 6)], (255, 255, 255), (226, 232, 240), ow=2.6, light=0.3)
                p.blobs([("r", (c0 - 8 * u, h * 0.5, c1 + 8 * u, h * 0.54), 4)], (176, 214, 222), ow=2.4)
                for q in range(3):
                    a = c0 + (c1 - c0) * q / 3
                    p.blobs([("r", (a + 6 * u, h * 0.56, a + (c1 - c0) / 3 - 6 * u, floor_y - 6 * u), 4)],
                            (246, 250, 252), (222, 230, 238), ow=2.0, light=0.2)
                    p.blobs([("r", (a + (c1 - c0) / 6 - 12 * u, h * 0.58, a + (c1 - c0) / 6 + 12 * u, h * 0.59), 2)],
                            (170, 176, 190), ow=1.4, light=0, dark=0)
                p.blobs([("r", (c0 + 60 * u, h * 0.43, c0 + 74 * u, h * 0.5), 3),
                         ("r", (c0 + 60 * u, h * 0.43, c0 + 110 * u, h * 0.445), 3)], (196, 200, 214), ow=2.0)
                p.blobs([("r", (c0 + 40 * u, h * 0.2, c1 - 40 * u, h * 0.4), 14)], (214, 236, 246), (190, 216, 232),
                        ow=2.6, light=0.6, dark=0.1)
            if k == 0:
                continue
            sx0, sx1, sy = m0 + w * 0.62, m0 + w * 0.86, h * 0.14
            p.blobs([("r", (sx0, sy, sx1, sy + 10 * u), 3)], (214, 170, 130), (190, 146, 110), ow=2.2)
            jx = sx0 + 10 * u
            for q in range(5):
                jw = rng.uniform(22, 34) * u
                jh = rng.uniform(26, 46) * u
                c = [(255, 200, 210), (200, 230, 255), (210, 240, 200), (255, 236, 180), (230, 214, 250)][q]
                p.blobs([("r", (jx, sy - jh, jx + jw, sy), 6)], c, _mix(c, (150, 130, 170), 0.25), ow=2.0, light=0.4)
                p.blobs([("r", (jx + 2 * u, sy - jh - 6 * u, jx + jw - 2 * u, sy - jh + 2 * u), 2)], (255, 255, 255), ow=1.6)
                jx += jw + 8 * u
                if jx > sx1 - 30 * u:
                    break
        # floor
        p.blobs([("r", (-40, floor_y, pw + 40, h + 40), 0)], (232, 222, 206), (210, 196, 178), ow=2.6, wrap=False,
                light=0, dark=0)
        for i in range(-10, int(pw / (60 * u)) + 10):  # perspective tile lines
            xa = i * 60 * u
            p.line([(xa, floor_y), (xa + (xa - pw / 2) * 0.35, h + 10)], (214, 200, 182), 1.4, wrap=False)
        for j, yy in enumerate((0.76, 0.81, 0.87, 0.94)):
            p.line([(-40, h * yy), (pw + 40, h * yy)], (214, 200, 182), 1.4, wrap=False)
        p.soft([("r", (-40, floor_y, pw + 40, floor_y + 14 * u), 0)], (80, 60, 70), 50, blur=5, wrap=False)
        li = self.add(p, factor=0.82, emissive=em, interior=IN)
        # furniture: dental chair + lamp; plant
        p = Painter(pw, h, u)
        em = Painter(pw, h, u)
        for k in range(1):
            cx = k * w + w * 0.80
            base = h * 0.80
            TEAL = ((130, 212, 210), (80, 168, 178))
            METAL = ((228, 232, 240), (186, 192, 208))
            # lamp: ceiling mount + jointed arm + head
            p.blobs([("r", (cx + 70 * u, -10, cx + 84 * u, h * 0.08), 4)], *METAL, ow=2.2)
            arm = [(cx + 77 * u, h * 0.08), (cx + 20 * u, h * 0.15), (cx - 10 * u, h * 0.24)]
            p.line(arm, (214, 218, 228), 9)
            p.line(arm, OUT, 1.4)
            for jx, jy in arm[1:]:
                p.blobs([("e", (jx - 8 * u, jy - 8 * u, jx + 8 * u, jy + 8 * u))], *METAL, ow=2.0)
            lx, ly = cx - 20 * u, h * 0.27
            p.blobs([("p", [(lx - 50 * u, ly + 16 * u), (lx + 44 * u, ly + 22 * u), (lx + 34 * u, ly - 12 * u),
                            (lx - 30 * u, ly - 16 * u)])], (250, 252, 255), (214, 220, 232), ow=2.4, light=0.4, dark=0.3)
            p.blobs([("e", (lx - 42 * u, ly + 10 * u, lx + 38 * u, ly + 26 * u))], (255, 250, 220), ow=2.0, light=0, dark=0)
            em.soft([("p", [(lx - 40 * u, ly + 20 * u), (lx + 36 * u, ly + 24 * u), (lx + 110 * u, h * 0.6),
                            (lx - 150 * u, h * 0.6)])], (255, 248, 210), 55 if not self.night else 85, blur=12)
            em.soft([("e", (lx - 40 * u, ly + 12 * u, lx + 36 * u, ly + 24 * u))], (255, 250, 225), 255)
            # floor shadow, base, column
            p.soft([("e", (cx - 190 * u, base - 6 * u, cx + 170 * u, base + 22 * u))], (60, 50, 60), 70, blur=6, wrap=False)
            p.blobs([("e", (cx - 80 * u, base - 10 * u, cx + 80 * u, base + 14 * u))], *METAL, ow=2.4)
            p.blobs([("r", (cx - 16 * u, h * 0.63, cx + 16 * u, base), 6)], *METAL, ow=2.4, rim=4)
            # instrument tray on a swing arm
            p.line([(cx - 10 * u, h * 0.66), (cx - 150 * u, h * 0.66), (cx - 150 * u, h * 0.5)], (214, 218, 228), 8)
            p.line([(cx - 10 * u, h * 0.66), (cx - 150 * u, h * 0.66), (cx - 150 * u, h * 0.5)], OUT, 1.4)
            p.blobs([("r", (cx - 205 * u, h * 0.485, cx - 95 * u, h * 0.505), 5)], *METAL, ow=2.2)
            for q, c in enumerate(((150, 200, 240), (240, 170, 180), (200, 200, 210))):
                ix = cx - 190 * u + q * 30 * u
                p.blobs([("r", (ix, h * 0.47, ix + 22 * u, h * 0.487), 3)], c, ow=1.4, light=0.2, dark=0)
            # leg rest, seat, back rest, head rest, arm rest
            p.blobs([("p", [(cx - 70 * u, h * 0.585), (cx - 70 * u, h * 0.645), (cx - 175 * u, h * 0.72), (cx - 205 * u, h * 0.70),
                            (cx - 200 * u, h * 0.665)])], *TEAL, ow=2.6, light=0.35, dark=0.3, rim=6)
            p.blobs([("r", (cx - 90 * u, h * 0.575, cx + 70 * u, h * 0.645), 16)], *TEAL, ow=2.8, light=0.4, dark=0.3, rim=7)
            p.blobs([("p", [(cx + 30 * u, h * 0.6), (cx + 82 * u, h * 0.63), (cx + 160 * u, h * 0.4), (cx + 118 * u, h * 0.37)])],
                    *TEAL, ow=2.8, light=0.4, dark=0.3, rim=7)
            p.blobs([("r", (cx + 112 * u, h * 0.29, cx + 170 * u, h * 0.37), 16)], *TEAL, ow=2.6, light=0.4, dark=0.3, rim=5)
            p.blobs([("r", (cx - 30 * u, h * 0.545, cx + 50 * u, h * 0.565), 6),
                     ("r", (cx - 4 * u, h * 0.555, cx + 8 * u, h * 0.585), 2)], *METAL, ow=2.0)
        self.plants = []
        ps = int(150 * u)

        def plant(q):
            rr = np.random.default_rng(5)
            for i in range(9):
                a = -math.pi / 2 + rr.uniform(-1.1, 1.1)
                L = ps * rr.uniform(0.35, 0.48)
                bx, by = ps / 2, ps * 0.95
                tx, ty = bx + math.cos(a) * L, by + math.sin(a) * L
                nx, ny = -math.sin(a) * L * 0.18, math.cos(a) * L * 0.18
                q.blobs([("p", [(bx, by), ((bx + tx) / 2 + nx, (by + ty) / 2 + ny), (tx, ty),
                                ((bx + tx) / 2 - nx, (by + ty) / 2 - ny)])], (150, 208, 120), (96, 160, 96), ow=2.0,
                        light=0.4, dark=0.25, rim=3)
        spr = _sprite(plant, ps, ps, u, self.mood, interior=IN)
        for k in range(2):
            px = k * w + (w * 0.4 if k == 0 else w * 0.5)
            p.blobs([("p", [(px - 34 * u, h * 0.78), (px + 34 * u, h * 0.78), (px + 26 * u, h * 0.86), (px - 26 * u, h * 0.86)])],
                    (240, 170, 140), (210, 136, 112), ow=2.4, light=0.3)
            p.blobs([("r", (px - 38 * u, h * 0.77, px + 38 * u, h * 0.79), 3)], (232, 160, 130), ow=2.2)
            self.plants.append((spr, px - ps / 2, h * 0.78 - ps * 0.97, k * 2.0))
        self.add(p, factor=1.0, emissive=em, interior=IN, after=self._anim)

    def _anim(self, fr, t, off, dy):
        for spr, lx, ly, ph in self.plants:
            _place(fr, _sway(spr, 0.03 * math.sin(t * 1.3 + ph)), lx, ly, off, self.pw, self.w)
        # clock hands on the wall layer (factor 0.82)
        wall_off = off * 0.82
        d = ImageDraw.Draw(fr)
        u = self.u
        for cx, cy, cr in self.clocks:
            sx = (cx - wall_off) % self.pw
            for x in (sx, sx - self.pw):
                if -cr < x < self.w + cr:
                    for ang, L, wd in ((t * 0.05, 0.5, 4), (t * 0.6, 0.72, 3), (t * 6.0, 0.8, 1.5)):
                        a = ang - math.pi / 2
                        d.line((x, cy, x + math.cos(a) * cr * L, cy + math.sin(a) * cr * L),
                               fill=OUT if wd > 2 else (230, 90, 90), width=max(1, int(wd * u)))
                    d.ellipse((x - 3 * u, cy - 3 * u, x + 3 * u, cy + 3 * u), fill=OUT)


# --------------------------------------------------------------------------
# SPACE
# --------------------------------------------------------------------------
NEB = {
    "day": ((0.30, 0.26, 0.52), (0.55, 0.40, 0.70), [(255, 170, 210), (170, 200, 255), (200, 170, 255)]),
    "sunset": ((0.28, 0.12, 0.30), (0.55, 0.25, 0.40), [(255, 150, 120), (255, 120, 180), (255, 200, 140)]),
    "night": ((0.02, 0.03, 0.10), (0.06, 0.08, 0.22), [(80, 120, 220), (120, 80, 200), (60, 170, 200)]),
    "rain": ((0.06, 0.10, 0.16), (0.14, 0.22, 0.30), [(90, 200, 200), (120, 160, 220), (160, 220, 210)]),
}


class Space(_Scene):
    seed = 51

    def make_sky(self):
        w, h, u = self.w, self.h, self.u
        top, bot, _ = NEB[self.mood]
        y = np.linspace(0, 1, h)[:, None, None]
        img = np.array(top) * (1 - y) + np.array(bot) * y
        img = np.broadcast_to(img, (h, w, 3)).astype(np.float32) * _grain((h, w), 5, 0.8)[..., None]
        return Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).convert("RGBA")

    def build(self):
        w, h, u, pw, rng = self.w, self.h, self.u, self.pw, self.rng
        neb = NEB[self.mood][2]
        # nebula: big soft blobs (no outline), slow drift
        p = Painter(pw, h, u)
        for i in range(26):
            x, y = rng.uniform(0, pw), rng.uniform(-0.1, 0.8) * h
            r = rng.uniform(80, 200) * u
            c = neb[i % 3]
            p.soft([("e", (x - r * 1.6, y - r, x + r * 1.6, y + r))], c, int(rng.uniform(40, 90)), blur=r * 0.35)
        img, y0 = p.done("day", grain=0.4)
        self.layers.append(_Layer(img, y0, 0.05, -3))
        # stars far + near (with sparkles)
        for li, (n, rmax, fac, sc) in enumerate(((420, 1.6, 0.1, -5), (90, 3.0, 0.3, -12))):
            p = Painter(pw, h, u)
            for _ in range(int(n * w / 1280)):
                x, y = rng.uniform(0, pw), rng.uniform(0, h * 0.78)
                r = rng.uniform(0.6, rmax)
                c = [(255, 250, 230), (210, 220, 255), (255, 220, 240)][int(rng.integers(3))]
                p.soft([("e", (x - r, y - r, x + r, y + r))], c, 255)
                if li == 1 and rng.random() < 0.3:
                    s = r * 3.5
                    p.blobs([("p", [(x, y - s), (x + s * 0.25, y - s * 0.25), (x + s, y), (x + s * 0.25, y + s * 0.25),
                                    (x, y + s), (x - s * 0.25, y + s * 0.25), (x - s, y), (x - s * 0.25, y - s * 0.25)])],
                            (255, 250, 220), ow=1.0, light=0, dark=0)
            img, y0 = p.done("day", grain=0)
            self.layers.append(_Layer(img, y0, fac, sc))
        self.twinkle = [(rng.uniform(0, w), rng.uniform(0, h * 0.7), rng.uniform(0, 6)) for _ in range(int(30 * w / 1280))]
        # planet with rings
        pr = 120 * u
        sw = int(pr * 3.4)
        sh = int(pr * 2.6)
        pal = {"day": ((255, 196, 170), (230, 140, 150)), "sunset": ((255, 190, 120), (226, 110, 110)),
               "night": ((160, 190, 255), (90, 110, 200)), "rain": ((160, 230, 220), (80, 160, 170))}[self.mood]

        def planet(q):
            cx, cy = sw / 2, sh / 2
            q.blobs([("e", (cx - pr * 1.6, cy - pr * 0.42, cx + pr * 1.6, cy + pr * 0.42))], (250, 236, 200), (220, 196, 160),
                    ow=2.4, light=0.2, dark=0.1)
            q.fill([("e", (cx - pr * 1.25, cy - pr * 0.26, cx + pr * 1.25, cy + pr * 0.26))], (0, 0, 0, 0), wrap=False)
            q.blobs([("e", (cx - pr, cy - pr, cx + pr, cy + pr))], pal[0], pal[1], ow=2.8, light=0.45, dark=0.35, rim=16)
            for (a, b, c) in ((-0.5, -0.38, 0.2), (0.0, 0.1, 0.15), (0.35, 0.45, 0.2)):
                q.soft([("e", (cx - pr * 0.95, cy + pr * a, cx + pr * 0.95, cy + pr * b))], _mix(pal[1], (255, 255, 255), c), 120,
                       blur=3, wrap=False)
            for (dx, dy_, r) in ((-0.5, 0.1, 0.12), (0.2, 0.52, 0.08), (0.55, -0.1, 0.06), (-0.1, -0.62, 0.07)):
                q.blobs([("e", (cx + pr * (dx - r), cy + pr * (dy_ - r * 0.8), cx + pr * (dx + r), cy + pr * (dy_ + r * 0.8)))],
                        _mix(pal[1], (120, 80, 120), 0.2), ow=1.6, light=0.0, dark=0.3)
            # front (lower) half of the ring, drawn over the planet
            ring = [(cx + math.cos(a) * pr * 1.6, cy + math.sin(a) * pr * 0.42) for a in np.linspace(0, math.pi, 40)]
            ring += [(cx + math.cos(a) * pr * 1.25, cy + math.sin(a) * pr * 0.26) for a in np.linspace(math.pi, 0, 40)]
            q.blobs([("p", ring)], (250, 236, 200), (220, 196, 160), ow=2.4, light=0.2, dark=0.15)
            q.line([(cx - pr * 1.42, cy + 1), (cx - pr * 0.6, cy + pr * 0.33)], (255, 255, 240, 140), 2.0, wrap=False)
        self.planet = _sprite(planet, sw, sh, u, "day", seed=3)
        self.planet_pos = (w * 0.68, h * 0.12)
        # small moons / asteroids drifting
        self.rocks = []
        for i in range(5):
            r = rng.uniform(10, 26) * u
            s = int(r * 2.6)
            rr = np.random.default_rng(i)

            def rock(q, r=r, s=s, rr=rr):
                pts = []
                for k in range(9):
                    a = k / 9 * 2 * math.pi
                    rad = r * rr.uniform(0.75, 1.05)
                    pts.append((s / 2 + math.cos(a) * rad, s / 2 + math.sin(a) * rad))
                q.blobs([("p", pts)], (210, 196, 220), (150, 130, 170), ow=2.0, light=0.4, dark=0.35)
                q.blobs([("e", (s / 2 - r * 0.3, s / 2 - r * 0.2, s / 2 + r * 0.05, s / 2 + r * 0.1))], (170, 150, 190), ow=1.2,
                        light=0, dark=0.2)
            self.rocks.append((_sprite(rock, s, s, u, "day", seed=i), rng.uniform(0, w), rng.uniform(0.05, 0.6) * h,
                               rng.uniform(6, 16) * u, rng.uniform(-20, 20), rng.uniform(0.3, 0.7)))
        # lunar ground
        gy = h * 0.74
        p = Painter(pw, h, u)
        gpal = {"day": ((226, 214, 236), (190, 176, 210)), "sunset": ((240, 206, 210), (204, 160, 180)),
                "night": ((150, 160, 200), (100, 106, 150)), "rain": ((170, 210, 214), (120, 160, 172))}[self.mood]
        p.blobs([("p", _ridge(pw, h * 0.66, [(14, 3, 0.3), (8, 7, 1.4)], h + 40))], _mix(gpal[0], (120, 120, 170), 0.3),
                _mix(gpal[1], (90, 90, 140), 0.3), ow=2.2, wrap=False, light=0.3, rim=8)
        img, y0 = p.done("day", haze=0.0, grain=0.8)
        self.layers.append(_Layer(img, y0, 0.5, 0))
        self.rock_at = len(self.layers)
        p = Painter(pw, h, u)
        p.blobs([("p", _ridge(pw, gy, [(8, 2, 1.0), (5, 5, 0.2), (3, 11, 2.0)], h + 40))], gpal[0], gpal[1], ow=2.6,
                wrap=False, light=0.35, rim=10, gy=(gy, h))
        for i in range(int(pw / (130 * u))):
            x = rng.uniform(0, pw)
            y = rng.uniform(h * 0.79, h * 0.96)
            a = rng.uniform(26, 60) * u * (0.6 + (y - gy) / h * 2)
            b = a * 0.28
            p.blobs([("e", (x - a, y - b, x + a, y + b))], _mix(gpal[1], (100, 80, 130), 0.25), gpal[1], ow=2.0, light=0,
                    dark=0.0)
            p.blobs([("e", (x - a * 0.85, y - b * 0.55, x + a * 0.85, y + b * 0.85))], _mix(gpal[1], (90, 70, 120), 0.15),
                    _mix(gpal[0], (255, 255, 255), 0.2), ow=0, light=0, dark=0)
        for i in range(int(pw / (70 * u))):
            x, y = rng.uniform(0, pw), rng.uniform(h * 0.76, h * 0.98)
            r = rng.uniform(4, 11) * u
            p.blobs([("e", (x - r * 1.3, y - r, x + r * 1.3, y + r * 0.4))], _mix(gpal[0], (255, 255, 255), 0.2), gpal[1],
                    ow=1.6, light=0.4, dark=0.3, rim=2)
        img, y0 = p.done("day", grain=1.2)
        self.layers.append(_Layer(img, y0, 1.0, 0))
        self.rain_at = len(self.layers)
        if self.rainy:  # meteor shower instead of rain
            self.rainy = False
            self.meteors = [(rng.uniform(0, w * 1.4), rng.uniform(-0.2, 0.5) * h, rng.uniform(0, 3), rng.uniform(0.8, 1.6))
                            for _ in range(10)]
        else:
            self.meteors = [(w * 0.3, h * 0.1, 1.0, 1.0)] if self.mood == "night" else []

    def anim_sky(self, fr, t, cam):
        pass

    def post(self, fr, t, cam):
        pass

    def frame(self, t, cam):
        w, h, u = self.w, self.h, self.u
        fr = self.sky.copy()
        for i, L in enumerate(self.layers):
            off = cam * L.factor + L.scroll * t * u
            _paste(fr, _wcrop(L.img, off, w), 0, L.y)
            if i == 2:  # after near stars: twinkles, planet, rocks
                d = ImageDraw.Draw(fr, "RGBA")
                for x, y, ph in self.twinkle:
                    k = max(0.0, math.sin(t * 2.2 + ph))
                    s = (2 + 7 * k) * u
                    a = int(230 * k)
                    xx = (x - off * 0.0) % w
                    d.line((xx - s, y, xx + s, y), fill=(255, 250, 230, a), width=max(1, int(1.5 * u)))
                    d.line((xx, y - s, xx, y + s), fill=(255, 250, 230, a), width=max(1, int(1.5 * u)))
                for mx, my, ph, sp in self.meteors:
                    period = 3.5 / sp
                    k = ((t + ph) % period) / period
                    if k < 0.35:
                        q = k / 0.35
                        x0 = mx - q * 600 * u
                        y0 = my + q * 300 * u
                        for j in range(8):
                            a = int(200 * (1 - j / 8) * math.sin(q * math.pi))
                            d.line((x0 + j * 12 * u, y0 - j * 6 * u, x0 + (j + 1) * 12 * u, y0 - (j + 1) * 6 * u),
                                   fill=(220, 255, 250, a), width=max(1, int((3 - j * 0.3) * u)))
                px = self.planet_pos[0] - cam * 0.2 - t * 1.5 * u
                py = self.planet_pos[1] + 6 * u * math.sin(t * 0.3)
                _paste(fr, self.planet, px % (w + self.planet.width * 2) - self.planet.width, py)
            if i == 3:
                for spr, x0, y0, sp, rot, fac in self.rocks:
                    span = w + spr.width * 2
                    x = (x0 - sp * t - cam * fac) % span - spr.width
                    _paste(fr, spr.rotate(rot * t, resample=Image.BILINEAR), x, y0 + 8 * u * math.sin(t * 0.5 + x0))
        im = ImageChops.multiply(fr.convert("RGB"), self.vig)
        return np.array(im)


# --------------------------------------------------------------------------
# ROOFTOP (Indian city)
# --------------------------------------------------------------------------
class Rooftop(_Scene):
    seed = 61
    horizon = 0.6
    sun = (0.2, 0.24)

    def build(self):
        w, h, u, pw, rng = self.w, self.h, self.u, self.pw, self.rng
        self.make_clouds(3, 0.03, 0.2, 30, 50, speed=5)
        L = self.lights
        # Aravalli-style hills
        p = Painter(pw, h, u)
        p.blobs([("p", _ridge(pw, h * 0.42, [(36, 2, 0.4), (22, 5, 2.0), (10, 11, 1.0)], h * 0.8))], (196, 200, 214),
                (176, 184, 200), ow=2.0, wrap=False, light=0.3, rim=10)
        self.add(p, factor=0.08, haze=0.42)
        # far skyline: palace domes, temple shikhara, blocks, tower
        p = Painter(pw, h, u)
        em = self.em(p)
        base = h * 0.56
        x = 0.0
        far = [(236, 222, 200), (232, 214, 220), (214, 222, 236), (240, 230, 206)]
        i = 0
        while x < pw:
            kind = ["block", "palace", "block", "temple", "block", "tower", "block", "dome"][i % 8]
            i += 1
            col = far[int(rng.integers(len(far)))]
            if kind == "block":
                bw = rng.uniform(60, 110) * u
                _building(p, em, x, base, bw, rng.uniform(0.08, 0.2) * h, col, rng, L * 0.8)
            elif kind == "palace":
                bw = 260 * u
                top = base - 0.15 * h
                p.blobs([("r", (x, top, x + bw, base + 4), 2)], (246, 236, 216), (226, 210, 190), ow=2.2, light=0.3)
                for q in range(6):  # arched windows
                    ax = x + 18 * u + q * 40 * u
                    p.blobs([("r", (ax, top + 30 * u, ax + 18 * u, top + 70 * u), 9)], (200, 180, 170), ow=1.6, light=0, dark=0)
                    if em is not None and rng.random() < L:
                        em.soft([("r", (ax, top + 30 * u, ax + 18 * u, top + 70 * u), 9)], GLOW, 255)
                for cxp in (0.12, 0.5, 0.88):  # chhatris
                    cx = x + bw * cxp
                    r = (34 if cxp == 0.5 else 22) * u
                    yb = top - (8 * u if cxp != 0.5 else 30 * u)
                    if cxp == 0.5:
                        p.blobs([("r", (cx - 50 * u, yb, cx + 50 * u, top + 2), 2)], (246, 236, 216), ow=2.0)
                    p.blobs([("r", (cx - r * 0.9, yb - 4 * u, cx + r * 0.9, yb + 2), 1)], (240, 228, 206), ow=1.8)
                    p.blobs([("e", (cx - r, yb - r * 1.6, cx + r, yb + r * 0.2)), ("p", [(cx - 3 * u, yb - r * 1.4), (cx + 3 * u, yb - r * 1.4), (cx, yb - r * 2.1)])],
                            (250, 240, 222), (226, 210, 190), ow=2.0, light=0.4)
                x += bw + 20 * u
                continue
            elif kind == "temple":
                bw = 120 * u
                top = base - 0.1 * h
                cx = x + bw / 2
                p.blobs([("r", (x, top, x + bw, base + 4), 2)], (250, 226, 196), (236, 200, 166), ow=2.2)
                sh = [("p", [(cx - 44 * u, top + 2), (cx + 44 * u, top + 2), (cx + 30 * u, top - 70 * u), (cx + 8 * u, top - 110 * u),
                             (cx - 8 * u, top - 110 * u), (cx - 30 * u, top - 70 * u)])]
                p.blobs(sh, (252, 214, 170), (232, 176, 130), ow=2.2, light=0.4, dark=0.3)
                for q in range(1, 5):
                    yy = top - q * 22 * u
                    half = 44 * u - q * 8.5 * u
                    p.line([(cx - half, yy), (cx + half, yy)], _mix((232, 176, 130), OUT, 0.4), 1.6)
                p.line([(cx, top - 110 * u), (cx, top - 140 * u)], OUT, 2.0)
                p.blobs([("p", [(cx, top - 140 * u), (cx + 26 * u, top - 133 * u), (cx, top - 124 * u)])], (255, 140, 60), ow=1.6,
                        light=0, dark=0)
                p.blobs([("e", (cx - 7 * u, top - 120 * u, cx + 7 * u, top - 106 * u))], (250, 210, 110), ow=1.6)
            elif kind == "tower":
                bw = 40 * u
                cx = x + bw / 2
                p.line([(cx - 14 * u, base), (cx, base - 0.36 * h), (cx + 14 * u, base)], (130, 130, 150), 2.4)
                for q in range(1, 7):
                    yy = base - q * 0.05 * h
                    hw = 14 * u * (1 - q * 0.05 * h / (0.36 * h))
                    p.line([(cx - hw, yy), (cx + hw, yy)], (130, 130, 150), 1.6)
                p.blobs([("r", (cx - 9 * u, base - 0.31 * h, cx - 3 * u, base - 0.27 * h), 1),
                         ("r", (cx + 3 * u, base - 0.31 * h, cx + 9 * u, base - 0.27 * h), 1)], (230, 230, 236), ow=1.4)
                if em is not None:
                    em.soft([("e", (cx - 4 * u, base - 0.37 * h, cx + 4 * u, base - 0.355 * h))], (255, 90, 80), 255)
            else:  # dome + minarets
                bw = 150 * u
                top = base - 0.1 * h
                cx = x + bw / 2
                p.blobs([("r", (x + 20 * u, top, x + bw - 20 * u, base + 4), 2)], (240, 240, 236), (220, 220, 220), ow=2.2)
                p.blobs([("e", (cx - 46 * u, top - 72 * u, cx + 46 * u, top + 16 * u)),
                         ("p", [(cx - 4 * u, top - 66 * u), (cx + 4 * u, top - 66 * u), (cx, top - 92 * u)])], (250, 250, 246),
                        (220, 224, 230), ow=2.2, light=0.4)
                for mx in (x + 12 * u, x + bw - 12 * u):
                    p.blobs([("r", (mx - 6 * u, top - 90 * u, mx + 6 * u, base + 4), 3),
                             ("e", (mx - 9 * u, top - 104 * u, mx + 9 * u, top - 84 * u))], (246, 246, 240), (220, 220, 220), ow=1.8)
            x += bw + rng.uniform(4, 24) * u
        self.add(p, factor=0.18, haze=0.35, emissive=em)
        # mid rooftops with Sintex tanks, dishes, clotheslines
        p = Painter(pw, h, u)
        em = self.em(p)
        cols = [(250, 190, 200), (255, 220, 150), (150, 214, 210), (250, 236, 200), (200, 214, 250), (255, 200, 160),
                (210, 236, 200)]
        x = 0.0
        while x < pw:
            bw = rng.uniform(130, 220) * u
            bh = rng.uniform(0.1, 0.2) * h
            col = cols[int(rng.integers(len(cols)))]
            top = h * 0.67 - bh
            _building(p, em, x, h * 0.70, bw, bh + 0.03 * h, col, rng, L, detail=True, roof=False, cols=3, rows=1)
            p.blobs([("r", (x - 3 * u, top - 4 * u, x + bw + 3 * u, top + 6 * u), 2)], _mix(col, (255, 255, 255), 0.3),
                    _mix(col, (150, 120, 130), 0.2), ow=2.0)
            if rng.random() < 0.85:  # Sintex tank
                tx = x + bw * rng.uniform(0.1, 0.6)
                tw, th = 46 * u, 40 * u
                p.blobs([("r", (tx - 2 * u, top - 10 * u, tx + tw + 2 * u, top - 2), 1)], (160, 150, 150), ow=1.8)
                p.blobs([("r", (tx, top - 10 * u - th, tx + tw, top - 8 * u), 8)], (80, 82, 94), (46, 48, 60), ow=2.2,
                        light=0.35, dark=0.2)
                for q in range(1, 4):
                    yy = top - 10 * u - th + q * th / 4
                    p.line([(tx + 3 * u, yy), (tx + tw - 3 * u, yy)], (110, 112, 126), 1.4)
                p.blobs([("e", (tx + 4 * u, top - 16 * u - th, tx + tw - 4 * u, top - 4 * u - th))], (96, 98, 110), ow=1.8)
            if rng.random() < 0.5:  # stair room
                sx = x + bw * rng.uniform(0.6, 0.75)
                p.blobs([("r", (sx, top - 0.07 * h, sx + 50 * u, top + 2), 2)], _mix(col, (255, 255, 255), 0.2), col, ow=2.0)
                p.blobs([("r", (sx + 14 * u, top - 0.05 * h, sx + 34 * u, top), 2)], (170, 120, 96), ow=1.6)
            if rng.random() < 0.5:  # dish
                dx = x + bw * 0.88
                p.blobs([("e", (dx - 14 * u, top - 30 * u, dx + 8 * u, top - 10 * u))], (236, 236, 240), (200, 200, 210), ow=1.8)
                p.line([(dx - 3 * u, top - 18 * u), (dx, top)], OUT, 1.6)
            if rng.random() < 0.6:  # mini clothesline
                a, b = x + bw * 0.1, x + bw * 0.55
                ly = top - 0.05 * h
                p.line([(a, top), (a, ly)], OUT, 1.6)
                p.line([(b, top), (b, ly)], OUT, 1.6)
                p.line([(a, ly), ((a + b) / 2, ly + 5 * u), (b, ly)], OUT, 1.0)
                for q in range(3):
                    cxq = a + (b - a) * (0.2 + q * 0.28)
                    c = [(240, 110, 110), (110, 170, 240), (250, 210, 90), (160, 210, 140)][int(rng.integers(4))]
                    p.blobs([("r", (cxq - 7 * u, ly + 3 * u, cxq + 7 * u, ly + 18 * u), 2)], c, ow=1.4, light=0.2, dark=0.1)
            x += bw + rng.uniform(-6, 18) * u
        self.add(p, factor=0.42, haze=0.1, emissive=em, after=self._kites)
        # near rooftop: parapet, floor, tank, plants, clotheslines
        p = Painter(pw, h, u)
        em = self.em(p)
        gy = h * 0.72
        p.blobs([("r", (-40, h * 0.62, pw + 40, gy + 4), 0)], (250, 214, 190), (236, 190, 166), ow=2.6, wrap=False,
                light=0.25, dark=0.2, rim=6)
        p.blobs([("r", (-40, h * 0.605, pw + 40, h * 0.63), 2)], (246, 236, 224), (226, 212, 200), ow=2.4, wrap=False,
                light=0.4, dark=0.2)
        for i in range(int(pw / (90 * u))):  # jaali openings
            jx = i * 90 * u + 20 * u
            p.blobs([("r", (jx, h * 0.645, jx + 50 * u, h * 0.69), 25)], (220, 160, 140), (200, 140, 124), ow=2.0, light=0,
                    dark=0.3)
            p.line([(jx + 25 * u, h * 0.645), (jx + 25 * u, h * 0.69)], _mix((236, 190, 166), OUT, 0.2), 2.2)
            p.line([(jx, h * 0.667), (jx + 50 * u, h * 0.667)], _mix((236, 190, 166), OUT, 0.2), 2.2)
        p.blobs([("r", (-40, gy, pw + 40, h + 40), 0)], (226, 214, 200), (204, 190, 176), ow=2.6, wrap=False, light=0,
                dark=0, gy=(gy, h))
        for i in range(int(pw / (110 * u)) + 1):
            xa = i * 110 * u
            p.line([(xa, gy), (xa + (xa - pw / 2) * 0.25, h + 10)], (206, 192, 178), 1.6, wrap=False)
        for yy in (0.79, 0.88):
            p.line([(-40, h * yy), (pw + 40, h * yy)], (206, 192, 178), 1.6, wrap=False)
        p.soft([("r", (-40, gy, pw + 40, gy + 12 * u), 0)], (90, 60, 60), 60, blur=4, wrap=False)
        self.clothes = []
        for k in range(2):
            m0 = k * w
            # big Sintex tank on stand
            tx = m0 + w * 0.03
            p.blobs([("r", (tx, h * 0.5, tx + 12 * u, gy + 8 * u), 2), ("r", (tx + 110 * u, h * 0.5, tx + 122 * u, gy + 8 * u), 2)],
                    (150, 150, 160), (120, 120, 132), ow=2.0)
            p.blobs([("r", (tx - 6 * u, h * 0.49, tx + 128 * u, h * 0.51), 2)], (150, 150, 160), ow=2.0)
            p.blobs([("r", (tx + 2 * u, h * 0.28, tx + 120 * u, h * 0.49), 18)], (76, 80, 94), (40, 42, 54), ow=2.8,
                    light=0.35, dark=0.2, rim=8)
            for q in range(1, 5):
                yy = h * 0.28 + q * (h * 0.21) / 5
                p.line([(tx + 6 * u, yy), (tx + 116 * u, yy)], (104, 108, 122), 1.8)
            p.blobs([("e", (tx + 34 * u, h * 0.262, tx + 88 * u, h * 0.292))], (96, 100, 114), ow=2.0)
            p.line([(tx + 120 * u, h * 0.45), (tx + 150 * u, h * 0.45), (tx + 150 * u, gy - 10 * u)], (230, 230, 236), 5)
            p.line([(tx + 120 * u, h * 0.45), (tx + 150 * u, h * 0.45), (tx + 150 * u, gy - 10 * u)], OUT, 1.2)
            # potted plants (tulsi) at parapet
            for px in (m0 + w * 0.32, m0 + w * 0.68):
                _tree_canopy(p, px, gy - 58 * u, 30 * u, rng, (150, 206, 116), (92, 152, 86))
                p.blobs([("p", [(px - 26 * u, gy - 34 * u), (px + 26 * u, gy - 34 * u), (px + 20 * u, gy + 6 * u),
                                (px - 20 * u, gy + 6 * u)])], (220, 130, 100), (190, 100, 80), ow=2.2, light=0.3)
            # clothesline posts + line
            a, b = m0 + w * 0.2, m0 + w * 0.8
            ly = h * 0.2
            for xx in (a, b):
                p.blobs([("r", (xx - 5 * u, ly - 10 * u, xx + 5 * u, h * 0.62), 2)], (176, 140, 110), (146, 110, 86), ow=2.0)
            p.line([(a, ly), ((a + b) / 2, ly + 22 * u), (b, ly)], OUT, 1.6)
            for q in range(5):
                f = 0.12 + q * 0.18
                cx = a + (b - a) * f
                cy = ly + 22 * u * (1 - (2 * f - 1) ** 2) + 2
                self.clothes.append((self._cloth(q + 5 * k), cx, cy, q * 1.3 + k))
        if self.lights > 0:  # fairy lights along parapet
            for i in range(int(pw / (30 * u))):
                x = i * 30 * u
                y = h * 0.6 + 6 * u * math.sin(i * 0.9)
                c = [(255, 220, 120), (255, 150, 150), (150, 220, 255), (190, 255, 160)][i % 4]
                p.fill([("e", (x - 4 * u, y - 4 * u, x + 4 * u, y + 4 * u))], OUT)
                em.soft([("e", (x - 3 * u, y - 3 * u, x + 3 * u, y + 3 * u))], c, 255)
        puds = []
        if self.rainy:
            for i in range(5):
                x = (i + rng.uniform(0.2, 0.8)) * pw / 5
                y = h * rng.uniform(0.82, 0.92)
                a, b = rng.uniform(70, 120) * u, rng.uniform(9, 14) * u
                p.blobs([("e", (x - a, y - b, x + a, y + b))], (170, 166, 170), (150, 148, 156), ow=1.4, light=0, dark=0)
                puds.append((x - a + 2, y - b + 2, x + a - 2, y + b - 2))
        li = self.add(p, factor=1.0, emissive=em, after=self._sway_clothes, grain=1.2)
        if puds:
            self.puddles(li, puds, h * 0.6)
        self.splash = (h * 0.74, h * 0.99, li)
        # kites
        self.kites = []
        kc = [((255, 120, 120), (255, 230, 120)), ((120, 200, 255), (255, 255, 255)), ((180, 140, 255), (255, 200, 120)),
              ((120, 220, 150), (255, 140, 180))]
        n = 0 if self.rainy else (2 if self.night else 4)
        for i in range(n):
            s = rng.uniform(26, 40) * u
            c1, c2 = kc[i % 4]
            ks = int(s * 2.6)

            def kite(q, s=s, c1=c1, c2=c2, ks=ks):
                cx, cy = ks / 2, ks / 2.6
                q.blobs([("p", [(cx, cy - s), (cx + s * 0.8, cy), (cx, cy + s), (cx - s * 0.8, cy)])], c1, ow=2.0, light=0.3,
                        dark=0.2)
                q.fill([("p", [(cx, cy - s), (cx + s * 0.8, cy), (cx, cy)])], c2, wrap=False)
                q.line([(cx, cy - s), (cx, cy + s)], OUT, 1.2)
                q.line([(cx - s * 0.8, cy), (cx + s * 0.8, cy)], OUT, 1.2)
                q.blobs([("p", [(cx, cy + s * 0.9), (cx + s * 0.3, cy + s * 1.35), (cx - s * 0.3, cy + s * 1.35)])], c2, ow=1.6,
                        light=0, dark=0)
            self.kites.append((_sprite(kite, ks, int(ks * 0.8), u, self.mood, seed=i), rng.uniform(0.1, 0.9) * w,
                               rng.uniform(0.06, 0.3) * h, rng.uniform(0, 6), rng.uniform(0.1, 0.9) * w))

    def _cloth(self, i):
        u = self.u
        rr = np.random.default_rng(300 + i)
        kind = rr.integers(3)
        c = [(240, 110, 120), (110, 170, 240), (250, 210, 90), (160, 210, 140), (250, 160, 210), (255, 255, 255)][int(rr.integers(6))]
        sw, sh = int(90 * u), int(120 * u)

        def draw(q):
            cx = sw / 2
            if kind == 0:  # kurta / shirt
                q.blobs([("p", [(cx - 26 * u, 6 * u), (cx + 26 * u, 6 * u), (cx + 40 * u, 26 * u), (cx + 28 * u, 34 * u),
                                (cx + 24 * u, 26 * u), (cx + 24 * u, 92 * u), (cx - 24 * u, 92 * u), (cx - 24 * u, 26 * u),
                                (cx - 28 * u, 34 * u), (cx - 40 * u, 26 * u)])], c, _mix(c, (120, 80, 120), 0.25), ow=2.0,
                        light=0.3, dark=0.25)
                q.line([(cx, 8 * u), (cx, 40 * u)], _mix(c, OUT, 0.5), 1.4)
            elif kind == 1:  # dupatta strip
                q.blobs([("p", [(cx - 18 * u, 6 * u), (cx + 18 * u, 6 * u), (cx + 22 * u, 110 * u), (cx - 14 * u, 104 * u)])], c,
                        _mix(c, (120, 80, 120), 0.25), ow=2.0, light=0.3, dark=0.25)
                for yy in (90, 100):
                    q.line([(cx - 16 * u, yy * u), (cx + 21 * u, yy * u + 4)], (255, 220, 120), 2.4)
            else:  # towel
                q.blobs([("r", (cx - 30 * u, 6 * u, cx + 30 * u, 74 * u), 4)], c, _mix(c, (120, 80, 120), 0.25), ow=2.0, light=0.3)
                for yy in (56, 64):
                    q.line([(cx - 28 * u, yy * u), (cx + 28 * u, yy * u)], (255, 255, 255), 2.4)
            for px in (cx - 14 * u, cx + 14 * u):
                q.blobs([("r", (px - 3 * u, 0, px + 3 * u, 12 * u), 1)], (250, 200, 90), ow=1.2, light=0, dark=0)
        return _sprite(draw, sw, sh, u, self.mood, seed=i)

    def _sway_clothes(self, fr, t, off, dy):
        for spr, cx, cy, ph in self.clothes:
            a = 0.08 * math.sin(t * 1.6 + ph) + 0.03 * math.sin(t * 3.3 + ph * 1.7)
            if self.rainy:
                a *= 0.4
            im = spr.transpose(Image.FLIP_TOP_BOTTOM)
            im = _sway(im, a).transpose(Image.FLIP_TOP_BOTTOM)  # pivot at the top (the line)
            _place(fr, im, cx - spr.width / 2, cy, off, self.pw, self.w)

    def _kites(self, fr, t, off, dy):
        d = ImageDraw.Draw(fr, "RGBA")
        u = self.u
        for spr, x0, y0, ph, ax in self.kites:
            x = x0 + 30 * u * math.sin(t * 0.5 + ph) - off * 0.1
            x = x % (self.w + 200 * u) - 100 * u
            y = y0 + 14 * u * math.sin(t * 1.1 + ph * 1.3)
            rot = 10 * math.sin(t * 1.3 + ph)
            im = spr.rotate(rot, resample=Image.BILINEAR)
            kx, ky = x + im.width / 2, y + im.height / 2.6
            d.line((kx, ky, kx + (ax - x0) * 0.4 + 60 * u, self.h * 0.6), fill=(250, 250, 250, 150), width=max(1, int(1.2 * u)))
            _paste(fr, im, x, y)


_CLASSES = {"park": Park, "metro": Metro, "clinic": Clinic, "city": City, "space": Space, "rooftop": Rooftop}
_CACHE = {}
_CACHE_MAX = 8


def _scene(name, w, h, mood):
    key = (name, int(w), int(h), mood)
    sc = _CACHE.get(key)
    if sc is None:
        if len(_CACHE) >= _CACHE_MAX:
            _CACHE.pop(next(iter(_CACHE)))
        sc = _CACHE[key] = _CLASSES[name](int(w), int(h), mood)
    return sc


def clear_cache():
    _CACHE.clear()


def render_scene(name, w, h, t, mood="day", camera_x=0.0):
    """Render scene ``name`` at time ``t`` (s). camera_x pans in foreground pixels."""
    if name not in _CLASSES:
        raise KeyError(f"unknown scene {name!r}; choose from {SCENES}")
    if mood not in GRADE:
        mood = "day"
    return _scene(name, w, h, mood).frame(float(t), float(camera_x))


def render(w, h, t, mood="day", bg="sky"):
    """Drop-in replacement for backgrounds.render: known scene names use this
    library, anything else falls back to backgrounds.render."""
    if bg in _CLASSES:
        return render_scene(bg, w, h, t, mood)
    return backgrounds.render(w, h, t, mood, bg)
