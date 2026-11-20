"""Contact sheet + timing for vidliboom.scenes.

    python examples/demo_scenes.py [--moods day,night] [--out output/scenes_sheet.png]
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from vidliboom import scenes  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--moods", default="day,night")
    ap.add_argument("--out", default="output/scenes_sheet.png")
    ap.add_argument("--w", type=int, default=1280)
    ap.add_argument("--h", type=int, default=720)
    ap.add_argument("--thumb", type=int, default=640)
    ap.add_argument("--scenes", default=",".join(scenes.SCENES))
    a = ap.parse_args()
    moods = a.moods.split(",")
    names = a.scenes.split(",")
    tw = a.thumb
    th = tw * a.h // a.w
    pad = 26
    sheet = Image.new("RGB", (len(moods) * (tw + 8) + 8, len(names) * (th + pad + 8) + 8), (40, 34, 32))
    d = ImageDraw.Draw(sheet)
    print(f"{'scene':8s} {'mood':7s} {'build':>8s} {'frame avg':>10s} {'max':>8s}")
    for r, name in enumerate(names):
        for c, mood in enumerate(moods):
            t0 = time.perf_counter()
            scenes.render_scene(name, a.w, a.h, 0.0, mood)
            build = time.perf_counter() - t0
            ts = []
            for k in range(24):
                t1 = time.perf_counter()
                arr = scenes.render_scene(name, a.w, a.h, 1.0 + k / 24, mood, camera_x=k * 6.0)
                ts.append(time.perf_counter() - t1)
            assert arr.shape == (a.h, a.w, 3) and arr.dtype == np.uint8
            print(f"{name:8s} {mood:7s} {build * 1000:7.0f}ms {np.mean(ts) * 1000:8.1f}ms {np.max(ts) * 1000:6.1f}ms")
            im = Image.fromarray(scenes.render_scene(name, a.w, a.h, 2.3, mood, camera_x=40.0)).resize((tw, th), Image.LANCZOS)
            x, y = 8 + c * (tw + 8), 8 + r * (th + pad + 8)
            d.text((x + 4, y + 6), f"{name} / {mood}", fill=(240, 230, 220))
            sheet.paste(im, (x, y + pad))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    sheet.save(a.out)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
