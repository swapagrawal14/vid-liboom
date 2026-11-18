"""Character layer. Uses assets/characters/<NAME>.png when present
(AI-art cut-outs), otherwise a simple stand-in. Adds idle bob and talk squash."""
import math
import os
from PIL import Image, ImageDraw

_cache = {}


def _placeholder(name, size=360):
    im = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    hue = (hash(name) % 200) + 30
    d.ellipse((60, 20, 300, 260), fill=(255, 214, 170, 255))
    d.rounded_rectangle((90, 230, 270, 360), 40, fill=(hue, 120, 220 - hue // 2, 255))
    d.ellipse((130, 110, 155, 140), fill=(30, 30, 30, 255))
    d.ellipse((205, 110, 230, 140), fill=(30, 30, 30, 255))
    return im


def load(name, assets_dir="assets/characters"):
    if name not in _cache:
        p = os.path.join(assets_dir, f"{name}.png")
        _cache[name] = Image.open(p).convert("RGBA") if os.path.exists(p) else _placeholder(name)
    return _cache[name]


def pose(name, t, talking, h_target):
    im = load(name)
    s = h_target / im.height
    sq = 1 + (0.04 * math.sin(t * 22) if talking else 0)
    im = im.resize((int(im.width * s / sq), int(im.height * s * sq)))
    bob = int(6 * math.sin(t * 2.2))
    return im, bob
