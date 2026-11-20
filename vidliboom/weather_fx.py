"""Personal Mausam: a tiny weather system that floats above each character's
head and mirrors their emotion.

    fx = WeatherFX(seed=0)
    fx.draw(frame, head_xy=(640, 260), size=130, emotion="sad", intensity=1.0, t=t)

* ``head_xy`` is the top-centre of the character's head in frame pixels.
* ``size`` is roughly the head width in pixels; everything scales from it.
* ``intensity`` 0..1 fades the effect in/out (opacity plus a small scale pop).
  Ramp it over ~0.3 s when the emotion changes for smooth transitions.
* ``t`` is time in seconds. All motion is a pure function of ``t`` and the
  seed, so any frame can be rendered independently and stays consistent.

Everything is drawn procedurally with PIL on a small local patch at 2x and
downsampled (LANCZOS) for clean anti-aliased edges, in a pastel cartoon style
with dark warm outlines to sit next to AI-art character cut-outs.

``scene_tint(emotion, intensity)`` returns an RGBA overlay colour for an
optional subtle full-frame grade; ``apply_tint`` applies it.
"""
import math
import zlib

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

SS = 2                      # supersampling factor
OUTLINE = (50, 35, 30, 255)
EMOTIONS = ("happy", "sad", "angry", "scared", "excited", "calm", "smug",
            "surprised", "neutral")
ALIASES = {"joy": "happy", "proud": "smug", "worried": "scared", "fear": "scared",
           "mad": "angry", "shocked": "surprised", "startled": "surprised",
           "zen": "calm", "relaxed": "calm", "giggle": "excited"}

_TINTS = {
    "happy": (255, 214, 140, 34),
    "sad": (80, 105, 150, 46),
    "angry": (170, 40, 40, 38),
    "scared": (95, 110, 175, 42),
    "excited": (255, 170, 215, 30),
    "calm": (140, 205, 195, 30),
    "smug": (255, 196, 80, 30),
    "surprised": (255, 245, 190, 34),
    "neutral": (0, 0, 0, 0),
}


def _norm(emotion):
    e = (emotion or "neutral").lower().strip()
    return ALIASES.get(e, e if e in _TINTS else "neutral")


def scene_tint(emotion, intensity):
    """(r, g, b, a) overlay colour for a subtle full-frame grade."""
    r, g, b, a = _TINTS[_norm(emotion)]
    k = max(0.0, min(1.0, float(intensity)))
    return (r, g, b, int(round(a * k)))


def apply_tint(img, emotion, intensity):
    """Blend scene_tint over ``img`` in place (soft-light-ish via alpha mix)."""
    r, g, b, a = scene_tint(emotion, intensity)
    if a <= 0:
        return img
    over = Image.new(img.mode, img.size, (r, g, b, 255) if img.mode == "RGBA" else (r, g, b))
    img.paste(Image.blend(img, over, a / 255.0))
    return img


def _ease(x):
    x = max(0.0, min(1.0, x))
    return x * x * (3 - 2 * x)


def _mix(c1, c2, k):
    return tuple(int(round(a + (b - a) * k)) for a, b in zip(c1[:3], c2[:3])) + (255,)


class WeatherFX:
    def __init__(self, seed=0):
        self.seed = int(seed)
        self._parts = {}

    # ------------------------------------------------------------------ util
    def _rng_params(self, key, n, cols):
        """Cached per-emotion particle table: n rows x cols uniforms in [0,1)."""
        k = (key, n, cols)
        if k not in self._parts:
            rng = np.random.default_rng(self.seed * 7919 + zlib.crc32(key.encode()))
            P = rng.random((n, cols))
            # stratify column 0 (x placement) so particles spread evenly
            P[:, 0] = (rng.permutation(n) + 0.15 + 0.7 * P[:, 0]) / n
            self._parts[k] = P
        return self._parts[k]

    # --------------------------------------------------------------- public
    def draw(self, img, head_xy, size, emotion, intensity, t):
        emo = _norm(emotion)
        e = _ease(float(intensity))
        if emo == "neutral" or e <= 0.003 or size < 4:
            return
        hx, hy, size, t = float(head_xy[0]), float(head_xy[1]), float(size), float(t)
        # local patch (frame px) covering the weather and the head top
        pw, ph = int(size * 3.4), int(size * 2.0)
        ox, oy = int(round(hx - pw / 2)), int(round(hy - size * 1.75))
        u = size * SS * (0.72 + 0.28 * e)                # unit (patch px at SS)
        W2, H2 = pw * SS, ph * SS
        cx = (hx - ox) * SS
        head_y = (hy - oy) * SS
        bob = math.sin(t * 2.1 + self.seed) * 0.035 * size * SS
        cy = head_y - (0.5 + 0.3 * e) * size * SS + bob

        main = Image.new("RGBA", (W2, H2), (0, 0, 0, 0))
        glow = Image.new("RGBA", (W2, H2), (0, 0, 0, 0))
        dm, dg = ImageDraw.Draw(main), ImageDraw.Draw(glow)
        lw = max(2, int(round(u * 0.032)))
        getattr(self, "_fx_" + emo)(main, dm, dg, cx, cy, u, lw, t, head_y)

        if glow.getbbox():
            q = 4                                  # blur at 1/4 res: soft glow is low-frequency
            small = glow.convert("RGBa").resize((max(1, W2 // q), max(1, H2 // q)), Image.BILINEAR)
            small = small.filter(ImageFilter.GaussianBlur(u * 0.09 / q))
            blur = small.resize((W2, H2), Image.BILINEAR).convert("RGBA")
            main = Image.alpha_composite(blur, main)
        layer = main.resize((pw, ph), Image.LANCZOS)
        if e < 0.999:
            a = np.asarray(layer).copy()
            a[..., 3] = (a[..., 3].astype(np.float32) * e).astype(np.uint8)
            layer = Image.fromarray(a, "RGBA")
        # clip to frame and composite in place
        x0, y0 = max(0, ox), max(0, oy)
        x1, y1 = min(img.width, ox + pw), min(img.height, oy + ph)
        if x1 <= x0 or y1 <= y0:
            return
        layer = layer.crop((x0 - ox, y0 - oy, x1 - ox, y1 - oy))
        if img.mode == "RGBA":
            img.alpha_composite(layer, (x0, y0))
        else:
            img.paste(layer.convert(img.mode), (x0, y0), layer)

    # ------------------------------------------------------------ primitives
    @staticmethod
    def _ell(cx, cy, rx, ry=None):
        ry = rx if ry is None else ry
        return (cx - rx, cy - ry, cx + rx, cy + ry)

    def _cloud(self, d, cx, cy, w, fill, shade, lw, face=None, t=0.0, wobble=0.0):
        """Puffy outlined cloud centred at (cx, cy), width ~w."""
        bumps = [(-0.27, 0.02, 0.21), (-0.04, -0.12, 0.27), (0.24, -0.01, 0.21)]
        if wobble:
            bumps = [(bx, by + wobble * math.sin(t * 9 + i * 2.1), r * (1 + 0.4 * wobble * math.sin(t * 11 + i)))
                     for i, (bx, by, r) in enumerate(bumps)]
        base = (cx - 0.46 * w, cy - 0.02 * w, cx + 0.46 * w, cy + 0.2 * w)
        rad = 0.11 * w
        # outline pass (union)
        for bx, by, r in bumps:
            d.ellipse(self._ell(cx + bx * w, cy + by * w, r * w + lw), fill=OUTLINE)
        d.rounded_rectangle((base[0] - lw, base[1] - lw, base[2] + lw, base[3] + lw), rad + lw, fill=OUTLINE)
        # shade then fill shifted up -> soft darker belly
        for bx, by, r in bumps:
            d.ellipse(self._ell(cx + bx * w, cy + by * w, r * w), fill=shade)
        d.rounded_rectangle(base, rad, fill=shade)
        s = 0.045 * w
        for bx, by, r in bumps:
            d.ellipse(self._ell(cx + bx * w, cy + by * w - s * 0.8, r * w - s * 0.7), fill=fill)
        d.rounded_rectangle((base[0] + s, base[1] - s * 0.8, base[2] - s, base[3] - s * 1.1), rad * 0.8, fill=fill)
        # highlight
        hl = _mix(fill, (255, 255, 255), 0.65)
        d.ellipse(self._ell(cx - 0.12 * w, cy - 0.25 * w, 0.075 * w, 0.045 * w), fill=hl)
        d.ellipse(self._ell(cx - 0.02 * w, cy - 0.3 * w, 0.025 * w), fill=hl)
        if face:
            self._face(d, cx, cy + 0.06 * w, w * 0.9, face, fill, lw, t)

    def _face(self, d, cx, cy, w, kind, skin, lw, t):
        ex, er = 0.13 * w, 0.042 * w
        blush = _mix(skin, (255, 120, 140), 0.45)
        eye_l, eye_r = (cx - ex, cy - 0.02 * w), (cx + ex, cy - 0.02 * w)
        k = max(1, int(lw * 0.9))
        if kind in ("happy", "sleepy"):
            for x, y in (eye_l, eye_r):
                box = self._ell(x, y + (0.02 * w if kind == "happy" else 0), er * 1.1)
                if kind == "happy":
                    d.arc(box, 200, 340, fill=OUTLINE, width=k)
                else:
                    d.arc(box, 20, 160, fill=OUTLINE, width=k)
        else:
            blink = ((t * 0.7 + cx * 0.0013) % 1.0) < 0.05
            for x, y in (eye_l, eye_r):
                if blink:
                    d.line((x - er, y, x + er, y), fill=OUTLINE, width=k)
                else:
                    ry = er * (1.35 if kind in ("scared", "surprised") else 1.1)
                    d.ellipse(self._ell(x, y, er * 0.85, ry), fill=OUTLINE)
                    d.ellipse(self._ell(x - er * 0.25, y - ry * 0.35, er * 0.3), fill=(255, 255, 255, 255))
        if kind != "angry":
            for x, y in (eye_l, eye_r):
                d.ellipse(self._ell(x + (0.03 * w if x > cx else -0.03 * w), y + 0.075 * w, 0.05 * w, 0.025 * w), fill=blush)
        my = cy + 0.07 * w
        if kind == "happy":
            d.chord(self._ell(cx, my - 0.02 * w, 0.06 * w, 0.05 * w), 0, 180, fill=OUTLINE)
            d.ellipse(self._ell(cx, my + 0.012 * w, 0.025 * w, 0.012 * w), fill=(240, 120, 120, 255))
        elif kind == "sleepy":
            d.arc(self._ell(cx, my - 0.02 * w, 0.035 * w, 0.025 * w), 20, 160, fill=OUTLINE, width=k)
        elif kind == "sad":
            d.arc(self._ell(cx, my + 0.03 * w, 0.045 * w, 0.03 * w), 200, 340, fill=OUTLINE, width=k)
            for x, y in (eye_l, eye_r):
                sgn = 1 if x < cx else -1
                d.line((x - sgn * 0.05 * w, y - 0.075 * w, x + sgn * 0.03 * w, y - 0.1 * w), fill=OUTLINE, width=k)
        elif kind == "angry":
            d.line((cx - 0.05 * w, my + 0.01 * w, cx + 0.05 * w, my), fill=OUTLINE, width=k + 1)
            for x, y in (eye_l, eye_r):
                sgn = 1 if x < cx else -1
                d.line((x - sgn * 0.07 * w, y - 0.11 * w, x + sgn * 0.04 * w, y - 0.055 * w), fill=OUTLINE, width=k + 2)
        elif kind == "scared":
            pts = [(cx - 0.06 * w + i * 0.02 * w, my + (0.012 * w if i % 2 else -0.004 * w)) for i in range(7)]
            d.line(pts, fill=OUTLINE, width=k, joint="curve")
        elif kind == "surprised":
            d.ellipse(self._ell(cx, my + 0.01 * w, 0.03 * w, 0.04 * w), fill=OUTLINE)

    def _sun(self, d, cx, cy, r, lw, t, fill=(255, 214, 92, 255), ray=(255, 178, 70, 255),
             n=10, spin=0.5, face="happy"):
        a0 = t * spin
        for i in range(n):
            a = a0 + i * 2 * math.pi / n
            L = r * (1.62 if i % 2 == 0 else 1.42) + r * 0.05 * math.sin(t * 4 + i)
            hw = 0.17 if i % 2 == 0 else 0.15
            pts = [(cx + math.cos(a - hw) * r * 1.02, cy + math.sin(a - hw) * r * 1.02),
                   (cx + math.cos(a) * L, cy + math.sin(a) * L),
                   (cx + math.cos(a + hw) * r * 1.02, cy + math.sin(a + hw) * r * 1.02)]
            d.polygon(pts, fill=ray, outline=OUTLINE, width=lw)
        d.ellipse(self._ell(cx, cy, r + lw), fill=OUTLINE)
        d.ellipse(self._ell(cx, cy, r), fill=_mix(fill, ray, 0.55))
        d.ellipse(self._ell(cx - r * 0.06, cy - r * 0.07, r * 0.9), fill=fill)
        d.ellipse(self._ell(cx - r * 0.42, cy - r * 0.45, r * 0.2, r * 0.12), fill=_mix(fill, (255, 255, 255), 0.6))
        if face:
            self._face(d, cx, cy + r * 0.05, r * 2.0, face, fill, lw, t)

    def _sparkle(self, d, x, y, r, col=(255, 250, 220, 255), lw=2):
        if r < 1:
            return
        q = r * 0.28
        pts = [(x, y - r), (x + q, y - q), (x + r, y), (x + q, y + q),
               (x, y + r), (x - q, y + q), (x - r, y), (x - q, y - q)]
        d.polygon(pts, fill=col, outline=OUTLINE, width=max(1, lw - 1))

    def _drop(self, d, x, y, r, lw, col=(140, 195, 245, 255)):
        d.ellipse(self._ell(x, y, r + lw), fill=OUTLINE)
        d.polygon([(x - r * 0.95 - lw * 0.7, y - r * 0.2), (x, y - r * 2.4 - lw * 1.4),
                   (x + r * 0.95 + lw * 0.7, y - r * 0.2)], fill=OUTLINE)
        d.ellipse(self._ell(x, y, r), fill=col)
        d.polygon([(x - r * 0.93, y - r * 0.25), (x, y - r * 2.3), (x + r * 0.93, y - r * 0.25)], fill=col)
        d.ellipse(self._ell(x - r * 0.35, y - r * 0.3, r * 0.22, r * 0.35), fill=(235, 248, 255, 255))

    def _flake(self, d, x, y, r, rot, lw):
        for pass_, (col, wd) in enumerate(((OUTLINE, lw * 2 + 2), ((240, 248, 255, 255), lw + 1))):
            for i in range(3):
                a = rot + i * math.pi / 3
                dx, dy = math.cos(a) * r, math.sin(a) * r
                d.line((x - dx, y - dy, x + dx, y + dy), fill=col, width=int(wd))
            d.ellipse(self._ell(x, y, r * 0.28 + (lw * 0.5 if pass_ == 0 else 0)), fill=col)
            for i in range(6):
                a = rot + i * math.pi / 3
                bx, by = x + math.cos(a) * r * 0.62, y + math.sin(a) * r * 0.62
                for s in (-1, 1):
                    b = a + s * 0.9
                    d.line((bx, by, bx + math.cos(b) * r * 0.3, by + math.sin(b) * r * 0.3), fill=col, width=int(wd * 0.8))

    def _bolt(self, d, x, y, s, lw, col=(255, 228, 90, 255)):
        pts = [(0.10, 0), (-0.22, 0.52), (0.02, 0.50), (-0.14, 1.0), (0.30, 0.36), (0.06, 0.38), (0.24, 0)]
        P = [(x + px * s, y + py * s) for px, py in pts]
        d.polygon(P, fill=col, outline=OUTLINE, width=lw)
        d.line([P[0], P[1]], fill=(255, 250, 210, 255), width=max(1, lw // 2))

    # ---------------------------------------------------------------- effects
    def _fx_happy(self, im, d, g, cx, cy, u, lw, t, head_y):
        r = 0.34 * u
        g.ellipse(self._ell(cx, cy, r * 2.0), fill=(255, 220, 120, 150))
        self._sun(d, cx, cy, r, lw, t)
        P = self._rng_params("happy", 7, 4)
        for i, (a, rad, sp, ph) in enumerate(P):
            ang = a * 2 * math.pi + t * 0.25
            dist = r * (2.0 + 0.7 * rad)
            tw = max(0.0, math.sin(t * (2.5 + 2 * sp) + ph * 6.28))
            self._sparkle(d, cx + math.cos(ang) * dist, cy + math.sin(ang) * dist * 0.8,
                          (0.05 + 0.06 * rad) * u * tw, lw=lw)

    def _fx_sad(self, im, d, g, cx, cy, u, lw, t, head_y):
        w = 1.0 * u
        P = self._rng_params("sad", 11, 3)
        top, bot = cy + 0.14 * w, head_y - 0.02 * u
        for x, ph, sp in P:                       # rain behind cloud
            p = (ph + t * (0.85 + 0.35 * sp)) % 1.0
            dx = (x - 0.5) * 0.78 * w
            y = top + (bot - top) * p * p * 0.3 + (bot - top) * p * 0.7
            if p > 0.93:                         # splash on the head
                k = (p - 0.93) / 0.07
                rr = 0.05 * u * (0.5 + k)
                d.arc(self._ell(cx + dx, bot, rr, rr * 0.45), 190, 350, fill=(120, 175, 230, 255), width=lw)
            else:
                rr = 0.052 * u * min(1.0, p * 8)
                if rr > 1:
                    self._drop(d, cx + dx, y, rr, lw)
        self._cloud(d, cx, cy, w, (196, 202, 218, 255), (150, 158, 182, 255), lw, face="sad", t=t)

    def _fx_angry(self, im, d, g, cx, cy, u, lw, t, head_y):
        w = 1.05 * u
        ph = (t * 1.3) % 1.0
        flash = ph < 0.11 or 0.17 < ph < 0.27
        shake = math.sin(t * 47) * 0.012 * u
        P = self._rng_params("angry", 5, 3)        # heavy dark drops
        top, bot = cy + 0.14 * w, head_y
        for x, p0, sp in P:
            p = (p0 + t * 1.5) % 1.0
            if p < 0.92:
                self._drop(d, cx + (x - 0.5) * 0.7 * w, top + (bot - top) * p, 0.035 * u * min(1, p * 8),
                           max(1, lw - 1), col=(110, 130, 170, 255))
        if flash:
            g.ellipse(self._ell(cx, cy + 0.45 * u, 0.6 * u, 0.55 * u), fill=(255, 240, 140, 210))
            self._bolt(d, cx - 0.05 * u + shake, cy + 0.1 * w, 0.62 * u, lw)
            self._bolt(d, cx + 0.26 * u, cy + 0.12 * w, 0.38 * u, lw)
        fill = (150, 140, 170, 255) if flash else (112, 104, 132, 255)
        self._cloud(d, cx + shake, cy, w, fill, _mix(fill, (40, 30, 50), 0.4), lw, face="angry", t=t)
        # cartoon anger-vein mark, pulsing
        vs = 0.14 * u * (1 + 0.15 * math.sin(t * 14))
        vx, vy = cx + 0.5 * w + shake, cy - 0.3 * w
        for qx in (-1, 1):
            for qy in (-1, 1):
                g_, L = 0.22 * vs, 0.95 * vs
                p0, c, p1 = (qx * g_, qy * L), (qx * g_ * 1.1, qy * g_ * 1.1), (qx * L, qy * g_)
                pts = [(vx + (1 - k) ** 2 * p0[0] + 2 * k * (1 - k) * c[0] + k * k * p1[0],
                        vy + (1 - k) ** 2 * p0[1] + 2 * k * (1 - k) * c[1] + k * k * p1[1])
                       for k in (i / 10 for i in range(11))]
                d.line(pts, fill=OUTLINE, width=lw * 3, joint="curve")
                d.line(pts, fill=(235, 70, 80, 255), width=lw + 1, joint="curve")

    def _fx_scared(self, im, d, g, cx, cy, u, lw, t, head_y):
        w = 0.92 * u
        jit = math.sin(t * 38) * 0.018 * u
        P = self._rng_params("scared", 6, 4)
        top, bot = cy - 0.1 * w, head_y + 0.02 * u
        for x, p0, sp, rot in P:
            p = (p0 + t * (0.22 + 0.12 * sp)) % 1.0
            sh = math.sin(t * 34 + rot * 20) * 0.02 * u
            fx = cx + (x - 0.5) * 1.5 * w + math.sin(t * 1.5 + rot * 6) * 0.06 * u + sh
            r = 0.085 * u * (0.75 + 0.45 * sp) * min(1.0, p * 6, (1 - p) * 6)
            if r > 1.5:
                self._flake(d, fx, top + (bot - top) * p, r, rot * 3 + t * 0.8, max(2, int(lw * 0.7)))
        self._cloud(d, cx + jit, cy, w, (214, 222, 244, 255), (168, 178, 214, 255), lw,
                    face="scared", t=t, wobble=0.03)
        for s in (-1, 1):                          # shiver marks
            x = cx + s * 0.62 * w + jit
            for j in range(2):
                yy = cy - 0.05 * w + j * 0.12 * w
                d.line((x, yy, x + s * 0.1 * w, yy - 0.04 * w), fill=OUTLINE, width=lw)

    def _fx_excited(self, im, d, g, cx, cy, u, lw, t, head_y):
        R = 0.62 * u
        by = cy + 0.38 * u
        bands = [(255, 128, 128), (255, 186, 110), (255, 232, 120), (140, 220, 150), (120, 180, 245), (186, 150, 235)]
        bw = 0.05 * u
        sweep = 360
        outer = R + bw * len(bands) / 2
        d.arc(self._ell(cx, by, outer + lw), 180, sweep, fill=OUTLINE, width=int(bw * len(bands) + 2 * lw))
        for i, c in enumerate(bands):
            rr = outer - i * bw
            d.arc(self._ell(cx, by, rr), 180, sweep, fill=c + (255,), width=int(bw + 1))
        d.arc(self._ell(cx, by, outer - 0.35 * bw), 200, 250, fill=(255, 255, 255, 170), width=max(1, int(bw * 0.3)))
        for s in (-1, 1):
            self._cloud(d, cx + s * R, by - 0.02 * u, 0.42 * u, (252, 252, 255, 255), (206, 214, 236, 255), lw)
        P = self._rng_params("excited", 16, 5)
        cols = [(255, 120, 150), (255, 210, 90), (120, 210, 160), (120, 170, 255), (200, 140, 240), (255, 160, 90)]
        top, bot = cy - 0.75 * u, head_y + 0.05 * u
        for i, (x, p0, sp, rot, cc) in enumerate(P):
            p = (p0 + t * (0.35 + 0.25 * sp)) % 1.0
            fx = cx + (x - 0.5) * 2.2 * u + math.sin(t * 4 + rot * 9) * 0.07 * u
            fy = top + (bot - top) * p
            a = rot * 6.28 + t * (3 + 4 * sp)
            sw, sh = 0.05 * u, 0.028 * u * abs(math.cos(t * 5 + rot * 7)) + 0.006 * u
            ca, sa = math.cos(a), math.sin(a)
            pts = [(fx + ca * px - sa * py, fy + sa * px + ca * py)
                   for px, py in ((-sw, -sh), (sw, -sh), (sw, sh), (-sw, sh))]
            d.polygon(pts, fill=cols[int(cc * len(cols)) % len(cols)] + (255,), outline=OUTLINE, width=max(1, lw // 2))

    def _fx_calm(self, im, d, g, cx, cy, u, lw, t, head_y):
        for k, (ox, oy, w, sp) in enumerate(((-0.24, 0.04, 0.78, 0.35), (0.36, -0.2, 0.58, 0.27))):
            x = cx + (ox + 0.12 * math.sin(t * sp + k * 2)) * u
            self._cloud(d, x, cy + oy * u, w * u, (250, 250, 255, 255), (208, 222, 238, 255), lw,
                        face="sleepy" if k == 0 else None, t=t)
        P = self._rng_params("calm", 6, 4)
        for x, y, sp, ph in P:
            fx = cx + (x - 0.5) * 1.8 * u + math.sin(t * (0.6 + sp * 0.5) + ph * 6.28) * 0.12 * u
            fy = cy + (0.2 + 0.45 * y) * u + math.cos(t * (0.5 + sp * 0.4) + ph * 4) * 0.08 * u
            b = 0.55 + 0.45 * math.sin(t * (1.6 + sp) + ph * 6.28)
            g.ellipse(self._ell(fx, fy, 0.11 * u * b + 0.03 * u), fill=(255, 240, 140, int(255 * b)))
            r = 0.032 * u * (0.7 + 0.3 * b)
            d.ellipse(self._ell(fx, fy, r + max(1, lw // 2)), fill=_mix((120, 110, 60), (255, 240, 140), b))
            d.ellipse(self._ell(fx, fy, r), fill=_mix((230, 220, 140), (255, 255, 210), b))
        # gentle "z" drifting up from the sleepy cloud
        k = (t * 0.45) % 1.0
        zx, zy, zs = cx - 0.02 * u + 0.12 * u * k, cy - (0.28 + 0.3 * k) * u, 0.07 * u * (0.6 + 0.6 * k)
        if k < 0.85:
            pts = [(zx - zs, zy - zs), (zx + zs, zy - zs), (zx - zs, zy + zs), (zx + zs, zy + zs)]
            d.line(pts, fill=OUTLINE, width=lw + 2, joint="curve")
            d.line(pts, fill=(170, 190, 240, 255), width=max(1, lw - 1), joint="curve")

    def _fx_smug(self, im, d, g, cx, cy, u, lw, t, head_y):
        r = 0.33 * u
        pulse = 0.5 + 0.5 * math.sin(t * 2.0)
        g.ellipse(self._ell(cx, cy, r * (2.0 + 0.25 * pulse)), fill=(255, 200, 70, 170))
        self._sun(d, cx, cy, r, lw, t, fill=(255, 205, 70, 255), ray=(245, 160, 40, 255), n=12, spin=0.18, face=None)
        # sunglasses
        gy = cy - r * 0.08
        for s in (-1, 1):
            box = (cx + s * r * 0.06 - (r * 0.72 if s < 0 else 0), gy - r * 0.2,
                   cx + s * r * 0.06 + (r * 0.72 if s > 0 else 0), gy + r * 0.3)
            d.rounded_rectangle((box[0] - lw, box[1] - lw, box[2] + lw, box[3] + lw), r * 0.17, fill=OUTLINE)
            d.rounded_rectangle(box, r * 0.14, fill=(40, 34, 52, 255))
            d.line((box[0] + r * 0.12, box[1] + r * 0.12, box[0] + r * 0.28, box[1] + r * 0.04),
                   fill=(170, 170, 210, 255), width=max(1, lw))
        d.line((cx - r * 0.95, gy - r * 0.08, cx + r * 0.95, gy - r * 0.08), fill=OUTLINE, width=lw + 1)
        # smirk
        d.arc(self._ell(cx + r * 0.15, cy + r * 0.32, r * 0.3, r * 0.17), 20, 120, fill=OUTLINE, width=lw + 1)
        d.ellipse(self._ell(cx + r * 0.58, cy + r * 0.32, r * 0.16, r * 0.08), fill=(255, 150, 120, 255))
        k = (t * 0.55) % 1.0                        # glint sweeping across the lens
        if k < 0.25:
            self._sparkle(d, cx - r * 0.65 + r * 1.3 * (k / 0.25), gy - r * 0.15,
                          0.09 * u * math.sin(k / 0.25 * math.pi), col=(255, 255, 255, 255), lw=lw)

    def _fx_surprised(self, im, d, g, cx, cy, u, lw, t, head_y):
        k = (t * 1.4) % 1.0
        pop = 1 + 0.12 * math.exp(-k * 7) * math.cos(k * 25)
        n, R = 11, 0.4 * u * pop
        g.ellipse(self._ell(cx, cy, R * 1.6), fill=(255, 240, 150, 190))
        pts = []
        for i in range(2 * n):
            a = -math.pi / 2 + i * math.pi / n + 0.05 * math.sin(t * 3)
            rr = R if i % 2 == 0 else R * 0.62
            pts.append((cx + math.cos(a) * rr, cy + math.sin(a) * rr))
        d.polygon(pts, fill=(255, 205, 80, 255), outline=OUTLINE, width=lw + 1)
        inner = [(cx + (x - cx) * 0.78, cy + (y - cy) * 0.78) for x, y in pts]
        d.polygon(inner, fill=(255, 238, 140, 255))
        # "!"
        bw, top = 0.07 * u, cy - 0.2 * u
        d.rounded_rectangle((cx - bw - lw, top - lw, cx + bw + lw, cy + 0.06 * u + lw), bw, fill=OUTLINE)
        d.rounded_rectangle((cx - bw, top, cx + bw, cy + 0.06 * u), bw, fill=(255, 110, 90, 255))
        d.ellipse(self._ell(cx, cy + 0.17 * u, 0.06 * u + lw), fill=OUTLINE)
        d.ellipse(self._ell(cx, cy + 0.17 * u, 0.06 * u), fill=(255, 110, 90, 255))
        # shock lines
        for i in range(8):
            a = i * math.pi / 4 + math.pi / 8
            r0 = R * (1.15 + 0.15 * k)
            r1 = r0 + 0.14 * u * (1 - k)
            if k < 0.75:
                d.line((cx + math.cos(a) * r0, cy + math.sin(a) * r0, cx + math.cos(a) * r1, cy + math.sin(a) * r1),
                       fill=OUTLINE, width=lw + 1)
