"""Composite frames and encode the episode with ffmpeg."""
import subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from . import backgrounds, characters

W, H, FPS = 1280, 720, 24


def _font(sz):
    for f in ("/usr/share/fonts/truetype/noto/NotoSansDevanagari-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(f, sz)
        except OSError:
            pass
    return ImageFont.load_default()


def _subtitle(img, text, font):
    d = ImageDraw.Draw(img)
    words, lines, cur = text.split(), [], ""
    for w in words:
        if d.textlength(cur + " " + w, font=font) > W * 0.8:
            lines.append(cur); cur = w
        else:
            cur = (cur + " " + w).strip()
    lines.append(cur)
    y = H - 40 - len(lines) * 44
    for ln in lines:
        x = (W - d.textlength(ln, font=font)) / 2
        d.text((x, y), ln, font=font, fill="white", stroke_width=4, stroke_fill="black")
        y += 44


def render(script, shots, total, out="episode.mp4", audio=None):
    font, tfont = _font(36), _font(64)
    speakers = sorted({s.speaker for s in shots})
    slots = {n: (i + 1) * W // (len(speakers) + 1) for i, n in enumerate(speakers)}
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-"]
    if audio:
        cmd += ["-i", audio, "-shortest"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", out]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in range(int(total * FPS)):
        t = f / FPS
        shot = next((s for s in shots if s.start <= t < s.start + s.duration), None)
        sc = shot.scene if shot else script.scenes[0]
        img = Image.fromarray(backgrounds.render(W, H, t, sc.mood, sc.bg))
        if shot is None:
            d = ImageDraw.Draw(img)
            x = (W - d.textlength(script.title, font=tfont)) / 2
            d.text((x, H / 2 - 40), script.title, font=tfont, fill="white", stroke_width=5, stroke_fill="black")
        else:
            present = {s.speaker for s in shots if s.scene is sc}
            for name in present:
                ch, bob = characters.pose(name, t, name == shot.speaker, 340)
                img.paste(ch, (slots[name] - ch.width // 2, int(H * 0.86) - ch.height + bob), ch)
            _subtitle(img, f"{shot.speaker.title()}: {shot.text}", font)
        p.stdin.write(np.asarray(img.convert("RGB")).tobytes())
    p.stdin.close(); p.wait()
    return out
