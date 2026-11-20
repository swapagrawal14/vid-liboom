"""Close-up lip-sync + blink demo.

    python examples/demo_lipsync.py [--wav output/cache/xxxx.wav] [--name BABLU]
                                    [--pose idle] [--seconds 3] [--out output/lipsync_demo.mp4]

Picks the loudest-speech window of the wav, renders the character's head
talking and blinking at 1280x720 / 24 fps, and muxes the audio with ffmpeg.
"""
import argparse
import glob
import math
import os
import subprocess
import sys
import wave

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from vidliboom import characters, lipsync, voice  # noqa: E402

W, H, FPS, SR = 1280, 720, 24, 44100


def best_window(audio, seconds):
    n = int(seconds * SR)
    if len(audio) <= n:
        return np.pad(audio, (0, n - len(audio)))
    hop = SR // 20
    e = np.array([np.sqrt((audio[i:i + n] ** 2).mean()) for i in range(0, len(audio) - n + 1, hop)])
    s = int(e.argmax()) * hop
    return audio[s:s + n]


def backdrop():
    y = np.linspace(0, 1, H)[:, None, None]
    top, bot = np.array([255, 214, 170]), np.array([250, 160, 140])
    g = (top * (1 - y) + bot * y) * np.ones((1, W, 1))
    yy, xx = np.mgrid[0:H, 0:W]
    v = 1 - 0.35 * (((xx - W / 2) / W) ** 2 + ((yy - H / 2) / H) ** 2)
    img = Image.fromarray(np.clip(g * v[..., None], 0, 255).astype(np.uint8)).convert("RGBA")
    d = ImageDraw.Draw(img)
    for i, (x, y_, r) in enumerate([(160, 140, 60), (1120, 220, 90), (1040, 600, 50), (220, 560, 80)]):
        d.ellipse((x - r, y_ - r, x + r, y_ + r), fill=(255, 255, 255, 40))
    return img.filter(ImageFilter.GaussianBlur(6))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", default=None)
    ap.add_argument("--name", default="BABLU")
    ap.add_argument("--pose", default="idle")
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--out", default="output/lipsync_demo.mp4")
    ap.add_argument("--frames", default=None, help="dir to also dump PNG frames for inspection")
    a = ap.parse_args()

    wav = a.wav or max(sorted(glob.glob("output/cache/*.wav")),
                       key=lambda p: os.path.getsize(p))
    audio = best_window(voice.read_wav(wav), a.seconds)
    track = lipsync.mouth_track(audio)
    nframes = int(round(a.seconds * FPS))
    blinks = lipsync.blink_schedule(a.seconds, seed=a.seed)

    # close-up framing: scale the pose so the face fills the frame
    base = characters.load(a.name, a.pose)
    _, L = lipsync.landmarks(a.name, a.pose)
    if L is None:
        sys.exit(f"no faces.json entry for {a.name}/{a.pose}")
    # close-up: frame the face from just above the eyes to just under the chin,
    # centred between the face and the middle of the cut-out
    pts = L["eyes"] + [L.get("mouth") or L.get("mask")]
    face_x = sum(p[0] for p in pts) / len(pts)
    eye_y = min(p[1] for p in pts)
    top = max(0.0, eye_y - 0.12)
    s = H / ((eye_y + 0.10 - top) * base.height)
    big = base.resize((int(base.width * s), int(base.height * s)), Image.LANCZOS)
    fx = (0.5 * 0.35 + face_x * 0.65) * big.width
    fy = (top * base.height) * s + H * 0.47
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    tmp_wav = os.path.splitext(a.out)[0] + "_audio.wav"
    with wave.open(tmp_wav, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-i", tmp_wav, "-shortest",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-c:a", "aac", "-b:a", "160k", a.out]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    bg = backdrop()
    if a.frames:
        os.makedirs(a.frames, exist_ok=True)
    for f in range(nframes):
        t = f / FPS
        o, v = (float(track.openness[f]), track.viseme[f]) if f < len(track) else (0.0, "closed")
        face = lipsync.apply_face(big, a.name, a.pose, o, v, float(blinks[f]))
        # a little life: breathing bob + a nod that follows the voice
        tilt = 1.2 * math.sin(t * 1.3) + 1.5 * o * math.sin(t * 7.0)
        bob = 6 * math.sin(t * 2.0) - 8 * o
        # crop to the visible area (plus margin) before rotating, for speed
        m = 200
        ox, oy = int(fx - W / 2 - m), int(fy - H * 0.47 - m - bob)
        face = face.crop((ox, oy, ox + W + 2 * m, oy + H + 2 * m))
        face = face.rotate(tilt, resample=Image.BICUBIC, center=(fx - ox, fy - oy + H * 0.6))
        frame = bg.copy()
        frame.alpha_composite(face.crop((m, m, m + W, m + H)))
        rgb = frame.convert("RGB")
        if a.frames:
            rgb.save(os.path.join(a.frames, f"f{f:03d}_{v}_{o:.2f}_b{blinks[f]:.2f}.png"))
        p.stdin.write(rgb.tobytes())
    p.stdin.close()
    p.wait()
    os.remove(tmp_wav)
    print(f"wrote {a.out}  ({nframes} frames, wav {os.path.basename(wav)})")


if __name__ == "__main__":
    main()
