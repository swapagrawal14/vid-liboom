"""AI director: rule-based emotion, pose, camera and timing for a parsed Script.

No external API. For every spoken line it decides:

* **emotion** + **intensity** (0..1) from a Hinglish / Hindi / English lexicon
  plus punctuation (``!``, ``?``, ``...``) and SHOUTED words;
* a **pose** when the script left it at ``idle``, picked from the PNGs that
  actually exist in ``assets/characters/<NAME>/``;
* a **camera shot** (``wide``, ``two-shot``, ``close``, ``punch-in``,
  ``reaction``) with a zoom factor and the character to frame;
* a **pause** to hold after the line (longer after ``?`` / ``...``, shorter on
  quick back-and-forth).

Usage::

    from vidliboom import script, director
    s = script.parse(open("examples/bablu_mausam.md").read())
    dirs = director.direct(s)
    director.apply(s, dirs)        # writes the chosen poses into the lines
"""
import os
import re
from dataclasses import dataclass

EMOTIONS = ("happy", "excited", "surprised", "angry", "sad", "scared", "calm", "smug", "neutral")

# word/phrase -> (emotion, weight). Matched on whole words, case-insensitive.
LEXICON = {
    "happy": ["haha", "hahaha", "hehe", "lol", "mast", "maza", "mazaa", "khush", "badhiya", "accha", "achha",
              "pyaar", "sundar", "love", "great", "nice", "happy", "shukriya", "thanks", "हाहा", "खुश", "मज़ा", "मस्त"],
    "excited": ["wah", "waah", "wow", "yay", "zabardast", "kamaal", "awesome", "amazing", "chalo", "jaldi",
                "yahoo", "वाह", "चलो"],
    "surprised": ["arre", "arey", "are baap", "kya", "sach mein", "sachchi", "really", "what", "whoa", "omg",
                  "achanak", "ajeeb", "अरे", "क्या"],
    "angry": ["gussa", "uff", "uf", "bas karo", "chup", "pagal", "bakwas", "nonsense", "kyun", "kyon", "hate",
              "stop", "hadd", "गुस्सा", "उफ़", "उफ", "क्यों", "चुप"],
    "sad": ["sorry", "dukh", "udaas", "rona", "afsos", "kaash", "sad", "miss", "akela", "bechara", "maaf",
            "सॉरी", "दुख", "उदास", "माफ़"],
    "scared": ["dar", "darr", "darna", "darr lag", "bhoot", "bachao", "help", "scared", "afraid", "khatra",
               "baap re", "mummy", "डर", "बचाओ", "भूत"],
    "calm": ["shaant", "shant", "aaram", "dheere", "relax", "theek hai", "koi baat nahi", "sukoon", "calm",
             "beta", "शांत", "आराम"],
    "smug": ["dekha", "bola tha", "maine kaha", "obviously", "apni marzi", "mera apna", "hmph", "bilkul",
             "dekha na", "देखा"],
}
LAUGHS = ("haha", "hehe", "lol", "हाहा", "😂", "🤣")

# Emotion -> pose-name keywords, in order of preference.
POSE_KEYWORDS = {
    "happy": ["proud", "wave", "giggle"],
    "excited": ["wave", "flail", "armsup"],
    "surprised": ["startled", "flinch", "lookup"],
    "scared": ["flinch", "worried"],
    "angry": ["flail", "armsup"],
    "sad": ["slump", "worried"],
    "calm": ["zen", "idle", "read"],
    "smug": ["smug", "sideeye"],
    "neutral": ["idle"],
}

BASE_ZOOM = {"wide": 1.0, "two-shot": 1.08, "close": 1.2, "punch-in": 1.35, "reaction": 1.15}
# What to switch to when a shot would repeat a third time in a row.
ALTERNATE = {"wide": "two-shot", "two-shot": "close", "close": "two-shot", "punch-in": "close", "reaction": "close"}
GAP = 0.35  # matches planner.GAP


@dataclass
class Direction:
    line: object          # the script.Line this direction is for
    emotion: str          # one of EMOTIONS
    intensity: float      # 0..1
    pose: str             # pose to use (script's own, or auto-picked)
    shot: str             # wide | two-shot | close | punch-in | reaction
    zoom: float           # camera zoom factor, 1.0 = full frame
    target: object        # speaker name to frame, or None for the whole cast
    pause_after: float    # seconds of silence to hold after the line


# --------------------------------------------------------------------- emotion
def _has(text, phrase):
    return re.search(r"(?<![\w\u0900-\u097F])" + re.escape(phrase) + r"(?![\w\u0900-\u097F])", text) is not None


def detect_emotion(text):
    """Return ``(emotion, intensity)`` for one line of dialogue."""
    low = text.lower()
    scores = dict.fromkeys(EMOTIONS, 0.0)
    for emo, words in LEXICON.items():
        for w in words:
            if _has(low, w):
                scores[emo] += 1.0
    if any(l in low for l in LAUGHS):
        scores["happy"] += 0.5

    bangs, qs = text.count("!"), text.count("?")
    ellipsis = "..." in text or "…" in text
    caps = [w for w in re.findall(r"[A-Za-z]{2,}", text) if w.isupper()]
    lexical = max(scores.values())

    if bangs:
        lead = max(("surprised", "excited", "angry", "happy", "scared", "smug"), key=scores.get)
        scores[lead if scores[lead] else "excited"] += 0.4 * min(bangs, 3)
    if qs:
        if bangs or scores["surprised"]:
            scores["surprised"] += 0.5
        elif scores["angry"]:
            scores["angry"] += 0.3
    if ellipsis:
        scores["sad" if scores["sad"] or not lexical else max(scores, key=scores.get)] += 0.4
    if caps:
        scores["angry" if scores["angry"] >= scores["excited"] else "excited"] += 0.5 * len(caps)

    emotion = max(EMOTIONS, key=lambda e: (scores[e], -EMOTIONS.index(e)))
    if scores[emotion] == 0:
        return "neutral", 0.1 + (0.1 if qs else 0.0)
    intensity = 0.25 + 0.2 * scores[emotion] + 0.1 * min(bangs, 3) + 0.12 * len(caps)
    if ellipsis:
        intensity -= 0.1
    return emotion, round(min(1.0, max(0.1, intensity)), 2)


# ------------------------------------------------------------------------ pose
def available_poses(name, assets_dir="assets/characters"):
    """Pose names (file stems) present for a character, sorted."""
    d = os.path.join(assets_dir, name)
    if not os.path.isdir(d):
        return []
    return sorted(os.path.splitext(f)[0] for f in os.listdir(d) if f.lower().endswith(".png"))


def pick_pose(emotion, poses, avoid=None):
    """Best pose for an emotion among ``poses``; avoids repeating ``avoid`` if it can."""
    matches = []
    for kw in POSE_KEYWORDS.get(emotion, []):
        matches += [p for p in poses if kw in p.lower() and p not in matches]
    if not matches:
        return "idle"
    fresh = [p for p in matches if p != avoid]
    return (fresh or matches)[0]


# ---------------------------------------------------------------------- camera
def _is_joke(line, nxt):
    """A line is a punchline if it laughs, or the next line laughs at it."""
    own = any(l in line.text.lower() for l in LAUGHS)
    answered = nxt is not None and any(l in nxt.text.lower() for l in LAUGHS)
    return own or answered


def _listener(i, lines):
    """Who is listening to line i: next other speaker, else previous, else None."""
    me = lines[i].speaker
    for j in list(range(i + 1, len(lines))) + list(range(i - 1, -1, -1)):
        if lines[j].speaker != me:
            return lines[j].speaker
    return None


def plan_shot(i, lines, emotion, intensity, history):
    """Return ``(shot, zoom, target)`` for line i of a scene."""
    line = lines[i]
    nxt = lines[i + 1] if i + 1 < len(lines) else None
    cast = list(dict.fromkeys(l.speaker for l in lines))
    listener = _listener(i, lines)

    if emotion in ("surprised", "excited") and intensity >= 0.6:
        shot = "punch-in"
    elif listener and (_is_joke(line, nxt) or line.text.rstrip().endswith("?")):
        shot = "reaction"
    elif i == 0 and len(cast) > 1:
        shot = "wide"
    elif intensity >= 0.45:
        shot = "close"
    else:
        shot = "two-shot" if len(cast) > 1 else "close"

    if len(history) >= 2 and history[-1] == history[-2] == shot:
        shot = ALTERNATE[shot]
        if shot == "two-shot" and len(cast) < 2:
            shot = "close"

    zoom = BASE_ZOOM[shot]
    if shot in ("close", "punch-in"):
        zoom += round(0.15 * intensity, 2)
    target = None if shot == "wide" else listener if shot == "reaction" else line.speaker
    return shot, round(zoom, 2), target


# ---------------------------------------------------------------------- timing
def beat(line, nxt, emotion):
    """Seconds to hold after ``line``; ``nxt`` is the next line in the scene or None."""
    t = line.text.rstrip()
    pause = GAP
    if t.endswith("?"):
        pause += 0.3
    if t.endswith("...") or t.endswith("…"):
        pause += 0.5
    if emotion in ("sad", "calm"):
        pause += 0.15
    if nxt is None:
        pause += 0.4  # let the scene breathe before the cut
    elif nxt.speaker != line.speaker and len(t.split()) <= 6 and len(nxt.text.split()) <= 6:
        pause *= 0.6  # quick back-and-forth
    return round(min(1.5, max(0.1, pause)), 2)


# ------------------------------------------------------------------------- api
def direct(script, assets_dir="assets/characters"):
    """Direct every spoken line of ``script``; returns one Direction per line, in order."""
    out, pose_cache = [], {}
    for sc in script.scenes:
        lines, history, last_pose = sc.lines, [], {}
        for i, line in enumerate(lines):
            nxt = lines[i + 1] if i + 1 < len(lines) else None
            emotion, intensity = detect_emotion(line.text)
            if line.pose == "idle":
                poses = pose_cache.setdefault(line.speaker, available_poses(line.speaker, assets_dir))
                pose = pick_pose(emotion, poses, avoid=last_pose.get(line.speaker))
            else:
                pose = line.pose
            last_pose[line.speaker] = pose
            shot, zoom, target = plan_shot(i, lines, emotion, intensity, history)
            history.append(shot)
            out.append(Direction(line, emotion, intensity, pose, shot, zoom, target, beat(line, nxt, emotion)))
    return out


def apply(script, directions):
    """Write each Direction's pose into its Line, in place. Returns the script."""
    for d in directions:
        d.line.pose = d.pose
    return script
