"""Personal Mausam demo: three characters cycle through every emotion's
weather in 4 seconds.  Run from the repo root:

    python examples/demo_weather_fx.py [out.mp4] [--frames 1.0,2.6]

--frames also dumps PNG stills at those times (seconds) next to the video.
"""
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from vidliboom import backgrounds, characters  # noqa: E402
from vidliboom.render import _font  # noqa: E402
from vidliboom.weather_fx import WeatherFX, apply_tint  # noqa: E402

W, H, FPS, DUR = 1280, 720, 24, 4.0
CAST = ["BABLU", "DIDI", "UNCLE"]
# each character covers three emotions -> all nine in one clip
SCHEDULE = [["happy", "sad", "angry"], ["excited", "scared", "calm"], ["smug", "surprised", "neutral"]]
SEG, FADE = DUR / 3, 0.22


def emotion_at(i, t):
    k = min(2, int(t / SEG))
    local = t - k * SEG
    inten = min(1.0, local / FADE, (SEG - local) / FADE)
    if k == 0:
        inten = min(1.0, (SEG - local) / FADE)            # start fully on
    elif k == 2:
        inten = min(1.0, local / FADE)                    # end fully on
    return SCHEDULE[i][k], max(0.0, inten)


def head_top(im):
    """Top-centre of the opaque area and approximate head width."""
    a = np.asarray(im)[..., 3] > 40
    ys = np.where(a.any(1))[0]
    y0 = ys[0]
    band = a[y0: y0 + max(4, im.height // 7)]
    xs = np.where(band.any(0))[0]
    return (xs[0] + xs[-1]) / 2, y0, xs[-1] - xs[0]


def frame(fx, t, font):
    img = Image.fromarray(backgrounds.render(W, H, t, "day", "sky")).convert("RGBA")
    slots = [(i + 1) * W // (len(CAST) + 1) for i in range(len(CAST))]
    hgt = int(H * 0.5)
    states = []
    for i, name in enumerate(CAST):
        ch, bob = characters.pose(name, t, 0.0, hgt, "idle", phase=i * 1.7)
        x, y = slots[i] - ch.width // 2, int(H * 0.86) - ch.height + bob
        img.alpha_composite(ch, (max(0, x), y))
        hx, hy, hw = head_top(ch)
        states.append((x + hx, y + hy, hw) + emotion_at(i, t))
    # subtle grade from the first character's mood, then the weather on top
    apply_tint(img, states[0][3], states[0][4] * 0.6)
    d = ImageDraw.Draw(img)
    for i, (hx, hy, hw, emo, inten) in enumerate(states):
        fx.draw(img, (hx, hy + hw * 0.08), max(110, min(150, hw)), emo, inten, t)
        lab = f"{CAST[i].title()} · {emo}"
        d.text((slots[i] - d.textlength(lab, font=font) / 2, H - 52), lab, font=font,
               fill="white", stroke_width=3, stroke_fill=(50, 35, 30))
    return img.convert("RGB")


def main():
    out = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "output/weather_fx_demo.mp4"
    stills = []
    if "--frames" in sys.argv:
        stills = [float(x) for x in sys.argv[sys.argv.index("--frames") + 1].split(",")]
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fx, font = WeatherFX(seed=3), _font(26)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", out]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in range(int(DUR * FPS)):
        t = f / FPS
        im = frame(fx, t, font)
        p.stdin.write(im.tobytes())
    p.stdin.close(); p.wait()
    for s in stills:
        frame(fx, s, font).save(os.path.splitext(out)[0] + f"_t{s:.2f}.png")
    print(out)


if __name__ == "__main__":
    main()
