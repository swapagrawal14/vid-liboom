"""Turn scenes into timed shots."""
from dataclasses import dataclass

WORDS_PER_SEC = 2.6
MIN_SHOT = 1.8


@dataclass
class Shot:
    scene: object
    speaker: str
    text: str
    start: float
    duration: float


def plan(script, title_card=2.5):
    shots, t = [], title_card
    for sc in script.scenes:
        for ln in sc.lines:
            d = max(MIN_SHOT, len(ln.text.split()) / WORDS_PER_SEC + 0.6)
            shots.append(Shot(sc, ln.speaker, ln.text, t, d))
            t += d
    return shots, t
