"""Branching (choose-your-own-adventure) episodes.

A branching script is a normal vid-liboom script with a few extras:

    # Bablu ka Faisla

    ## Scene: Baarish | id=rain | mood=rain
    BABLU [flail]: Arre! Baarish!
    ? Bablu kya kare?
    -> Chhata kholo | goto=umbrella
    -> Baarish mein naacho | goto=dance

    ## Scene: Chhata | id=umbrella | mood=day | next=home
    ...
    ## Scene: Ghar | id=home | end
    ...

Rules
    * ``id=<id>`` names a scene (default: a slug of the scene name).
    * ``? question`` is the prompt shown above the choice buttons (optional).
    * ``-> label | goto=<id>`` adds a choice. A scene with choices stops there.
    * A scene without choices plays on to the next scene in file order,
      unless it has ``next=<id>`` (jump) or ``end`` (this branch ends here).
      The last scene in the file is an ending too.
    * The first scene is the start, unless the title line says otherwise:
      ``# Title | start=<id>``.

Every scene becomes one *segment*: a plain markdown script (title + that one
scene, without the branching extras) that ``vidliboom.script.parse`` and the
normal renderer understand. Render each segment to its own mp4, then
``export_player`` writes a single self-contained ``index.html`` that plays the
graph.

    story = parse_branching(open("examples/bablu_choice.md").read())
    for issue in validate(story): print(issue)
    videos = render_all(story, "output/choice", my_render_fn)
    export_player(story, videos, "output/choice")

Command line (renders by calling ``python -m vidliboom`` once per segment):

    python -m vidliboom.branching examples/bablu_choice.md -o output/choice [--voice]
    python -m vidliboom.branching examples/bablu_choice.md --check
"""
from __future__ import annotations

import html
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Callable, Optional

__all__ = [
    "Choice", "Segment", "Story", "Issue",
    "parse_branching", "validate", "reachable",
    "render_all", "export_player", "subprocess_render_fn",
]

TEMPLATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "player_template.html")
_BRANCH_KEYS = {"id", "next", "goto", "start"}
_FLAGS = {"end"}


# --------------------------------------------------------------------------- model

@dataclass
class Choice:
    label: str
    goto: str


@dataclass
class Segment:
    id: str
    name: str
    script_text: str                 # plain vid-liboom script for this scene alone
    prompt: str = ""                 # the "? ..." question
    choices: list = field(default_factory=list)
    next: Optional[str] = None       # resolved follow-on scene (None for choices / endings)
    end: bool = False                # explicitly marked `| end`
    explicit_next: bool = False      # `next=` was written in the script
    line_no: int = 0                 # 1-based line of the `## Scene` header
    n_lines: int = 0                 # spoken lines in the scene

    @property
    def is_choice(self) -> bool:
        return bool(self.choices)

    @property
    def is_ending(self) -> bool:
        return not self.choices and self.next is None

    def targets(self) -> list:
        """Ids this segment can lead to, in order."""
        if self.choices:
            return [c.goto for c in self.choices]
        return [self.next] if self.next else []


@dataclass
class Story:
    title: str
    segments: dict                   # id -> Segment, in file order
    start: str
    duplicate_ids: list = field(default_factory=list)
    stray: list = field(default_factory=list)   # (line_no, text) of choices/prompts outside a scene

    def __getitem__(self, sid: str) -> Segment:
        return self.segments[sid]

    def __iter__(self):
        return iter(self.segments.values())

    @property
    def endings(self) -> list:
        return [s for s in self if s.is_ending]

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "start": self.start,
            "segments": {
                s.id: {
                    "name": s.name,
                    "prompt": s.prompt,
                    "choices": [{"label": c.label, "goto": c.goto} for c in s.choices],
                    "next": s.next,
                    "ending": s.is_ending,
                }
                for s in self
            },
        }


@dataclass
class Issue:
    level: str                       # "error" | "warning"
    segment: Optional[str]
    message: str

    def __str__(self):
        where = f"[{self.segment}] " if self.segment else ""
        return f"{self.level}: {where}{self.message}"


# --------------------------------------------------------------------------- parsing

def _slug(name: str) -> str:
    s = re.sub(r"[^\w]+", "-", name.strip().lower(), flags=re.UNICODE).strip("-")
    return s or "scene"


def _split_opts(parts):
    """`a=b` pairs and bare flags from the `|`-separated parts of a header."""
    opts, flags = {}, set()
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if "=" in p:
            k, v = p.split("=", 1)
            opts[k.strip().lower()] = v.strip()
        else:
            flags.add(p.lower())
    return opts, flags


def _segment_script(title: str, name: str, opts: dict, flags: set, body: list) -> str:
    """Rebuild a plain scene header + body that script.parse understands."""
    header = f"## Scene: {name}"
    for k, v in opts.items():
        if k not in _BRANCH_KEYS:
            header += f" | {k}={v}"
    out = [f"# {title}", "", header] + body
    while out and not out[-1].strip():
        out.pop()
    return "\n".join(out) + "\n"


_SPOKEN = re.compile(r"^([A-Za-z\u0900-\u097F ]+?)\s*(?:\[(\w+)\])?\s*:\s*(.+)$")


def parse_branching(text: str, segment_title: str = "scene") -> Story:
    """Parse a branching script into a Story graph.

    ``segment_title`` decides the title card each segment's video opens with
    (the renderer always shows one): ``"scene"`` uses the story title for the
    start segment and the scene name for the rest; ``"story"`` uses the story
    title everywhere.
    """
    title, start = "Untitled", None
    raw = []          # dicts per scene, in file order
    stray = []
    for no, line in enumerate(text.splitlines(), 1):
        s = line.strip()
        if s.startswith("## "):
            parts = s[3:].split("|")
            name = parts[0].strip().removeprefix("Scene:").strip()
            opts, flags = _split_opts(parts[1:])
            raw.append({"name": name, "opts": opts, "flags": flags, "body": [],
                        "prompt": "", "choices": [], "line_no": no, "n_lines": 0})
        elif s.startswith("# "):
            parts = s[2:].split("|")
            title = parts[0].strip()
            topts, _ = _split_opts(parts[1:])
            start = topts.get("start", start)
        elif s.startswith("->"):
            parts = s[2:].split("|")
            label = parts[0].strip()
            copts, _ = _split_opts(parts[1:])
            goto = copts.get("goto", "")
            if raw:
                raw[-1]["choices"].append(Choice(label, goto))
            else:
                stray.append((no, s))
        elif s.startswith("?") and not _SPOKEN.match(s):
            if raw:
                raw[-1]["prompt"] = s[1:].strip()
            else:
                stray.append((no, s))
        elif raw:
            raw[-1]["body"].append(line.rstrip())
            if _SPOKEN.match(s) and not s.startswith("("):
                raw[-1]["n_lines"] += 1

    segments, dups, order = {}, [], []
    for r in raw:
        sid = r["opts"].get("id") or _slug(r["name"])
        if sid in segments:
            dups.append(sid)
            base, k = sid, 2
            while f"{base}-{k}" in segments:
                k += 1
            sid = f"{base}-{k}"
        r["id"] = sid
        order.append(sid)
        segments[sid] = None  # reserve order
    if start is None and order:
        start = order[0]

    for i, r in enumerate(raw):
        sid = r["id"]
        seg_title = title if (segment_title == "story" or sid == start) else r["name"]
        end = "end" in r["flags"]
        explicit = "next" in r["opts"]
        nxt = None
        if not r["choices"] and not end:
            if explicit:
                nxt = r["opts"]["next"] or None
            elif i + 1 < len(raw):
                nxt = raw[i + 1]["id"]
        segments[sid] = Segment(
            id=sid, name=r["name"],
            script_text=_segment_script(seg_title, r["name"], r["opts"], r["flags"], r["body"]),
            prompt=r["prompt"], choices=r["choices"], next=nxt, end=end,
            explicit_next=explicit, line_no=r["line_no"], n_lines=r["n_lines"],
        )
    return Story(title, segments, start or "", dups, stray)


# --------------------------------------------------------------------------- validation

def reachable(story: Story) -> list:
    """Segment ids reachable from the start, breadth-first."""
    if story.start not in story.segments:
        return []
    seen, queue = [story.start], [story.start]
    while queue:
        for t in story.segments[queue.pop(0)].targets():
            if t in story.segments and t not in seen:
                seen.append(t)
                queue.append(t)
    return seen


def validate(story: Story) -> list:
    """Check the graph. Returns a list of Issue (errors first). Cycles are allowed.

    Errors: no scenes, unknown start, duplicate ids, a goto/next to a missing id,
    a choice with no label or no goto, `end` combined with choices or next,
    a scene with no spoken lines (the renderer needs at least one).
    Warnings: unreachable scenes, a prompt with no choices, a single-choice
    question, duplicate choice labels, scenes that can never reach an ending.
    """
    errors, warns = [], []
    E = lambda seg, msg: errors.append(Issue("error", seg, msg))
    W = lambda seg, msg: warns.append(Issue("warning", seg, msg))

    if not story.segments:
        E(None, "no scenes found")
        return errors
    if story.start not in story.segments:
        E(None, f"start scene '{story.start}' does not exist")
    for d in story.duplicate_ids:
        E(d, f"id '{d}' is used by more than one scene (later ones were renamed)")
    for no, s in story.stray:
        E(None, f"line {no}: '{s}' is outside any scene")

    for seg in story:
        if seg.n_lines == 0:
            E(seg.id, "scene has no spoken lines; the renderer needs at least one")
        if seg.end and (seg.choices or seg.explicit_next):
            E(seg.id, "scene is marked `end` but also has choices or next=")
        if seg.choices and seg.explicit_next:
            W(seg.id, "next= is ignored because the scene has choices")
        if seg.explicit_next and not seg.end and not seg.choices and seg.next is None:
            E(seg.id, "next= is empty")
        if seg.prompt and not seg.choices:
            W(seg.id, f"question '{seg.prompt}' has no choices")
        if len(seg.choices) == 1:
            W(seg.id, "only one choice; consider next= instead")
        labels = [c.label.lower() for c in seg.choices]
        if len(set(labels)) != len(labels):
            W(seg.id, "two choices have the same label")
        for c in seg.choices:
            if not c.label:
                E(seg.id, "a choice has no label")
            if not c.goto:
                E(seg.id, f"choice '{c.label}' has no goto=")
            elif c.goto not in story.segments:
                E(seg.id, f"choice '{c.label}' goes to missing id '{c.goto}'")
        if not seg.choices and seg.next and seg.next not in story.segments:
            E(seg.id, f"next goes to missing id '{seg.next}'")

    reach = reachable(story)
    for seg in story:
        if seg.id not in reach and story.start in story.segments:
            W(seg.id, "unreachable from the start scene")

    # Scenes from which no ending can ever be reached (a loop with no way out).
    can_end = {s.id for s in story if s.is_ending}
    changed = True
    while changed:
        changed = False
        for s in story:
            if s.id not in can_end and any(t in can_end for t in s.targets()):
                can_end.add(s.id)
                changed = True
    for sid in reach:
        if sid not in can_end:
            W(sid, "no path from here reaches an ending (endless loop)")
    if story.start in story.segments and not any(story[s].is_ending for s in reach):
        W(None, "no ending is reachable from the start")
    return errors + warns


# --------------------------------------------------------------------------- rendering

def render_all(story: Story, out_dir: str, render_fn: Callable[[str, str], object],
               ext: str = ".mp4", only_reachable: bool = True, skip_existing: bool = False,
               write_scripts: bool = True) -> dict:
    """Render every segment with ``render_fn(script_text, out_path)``.

    Videos go to ``<out_dir>/segments/<id><ext>``; the segment scripts are also
    written next to them as ``<id>.md`` (handy for re-rendering one by hand).
    Returns ``{id: relative path from out_dir}``, ready for ``export_player``.
    """
    seg_dir = os.path.join(out_dir, "segments")
    os.makedirs(seg_dir, exist_ok=True)
    ids = reachable(story) if only_reachable else list(story.segments)
    video_map = {}
    for sid in ids:
        seg = story[sid]
        out_path = os.path.join(seg_dir, f"{sid}{ext}")
        if write_scripts:
            with open(os.path.join(seg_dir, f"{sid}.md"), "w", encoding="utf-8") as f:
                f.write(seg.script_text)
        if not (skip_existing and os.path.exists(out_path)):
            render_fn(seg.script_text, out_path)
        video_map[sid] = os.path.relpath(out_path, out_dir).replace(os.sep, "/")
    return video_map


def subprocess_render_fn(*extra_args: str, python: str = sys.executable, cwd: Optional[str] = None):
    """A render_fn that runs ``python -m vidliboom <seg.md> -o <out> [extra_args]``."""
    def render_fn(script_text: str, out_path: str):
        md = os.path.splitext(out_path)[0] + ".md"
        if not os.path.exists(md):
            with open(md, "w", encoding="utf-8") as f:
                f.write(script_text)
        subprocess.run([python, "-m", "vidliboom", md, "-o", out_path, *extra_args],
                       check=True, cwd=cwd)
        return out_path
    return render_fn


# --------------------------------------------------------------------------- player

def _json_for_script(obj) -> str:
    """JSON that is safe inside an inline <script> tag."""
    s = json.dumps(obj, ensure_ascii=False)
    return (s.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
             .replace("\u2028", "\\u2028").replace("\u2029", "\\u2029"))


def export_player(story: Story, video_map: dict, out_dir: str, filename: str = "index.html",
                  countdown: int = 10, auto_pick: str = "first", allow_missing: bool = False,
                  template: Optional[str] = None) -> str:
    """Write a self-contained interactive player and return its path.

    ``video_map`` maps segment id -> video path relative to ``out_dir``.
    Every segment reachable from the start needs a video unless
    ``allow_missing`` (missing ones then show a placeholder card).
    ``auto_pick`` is ``"first"`` or ``"random"``: what the countdown picks.
    ``countdown`` is in seconds; 0 turns the auto-pick off.
    """
    if auto_pick not in ("first", "random"):
        raise ValueError("auto_pick must be 'first' or 'random'")
    errors = [i for i in validate(story) if i.level == "error"
              and not i.message.startswith("scene has no spoken lines")]
    if errors:
        raise ValueError("story has errors:\n" + "\n".join(map(str, errors)))
    missing = [s for s in reachable(story) if s not in video_map]
    if missing and not allow_missing:
        raise KeyError(f"no video for segment(s): {', '.join(missing)}")
    for sid, path in video_map.items():
        is_url = re.match(r"^[a-z][a-z0-9+.-]*:", path, re.I)
        if os.path.isabs(path) or (is_url and not path.lower().startswith(("http:", "https:"))):
            raise ValueError(f"video for '{sid}' must be a relative path or http(s) url: {path}")

    data = story.to_dict()
    for sid, seg in data["segments"].items():
        seg["video"] = video_map.get(sid)
    data["countdown"] = max(0, int(countdown))
    data["autoPick"] = auto_pick

    with open(template or TEMPLATE, encoding="utf-8") as f:
        page = f.read()
    page = page.replace("__TITLE__", html.escape(story.title))
    page = page.replace("/*__STORY_JSON__*/null", _json_for_script(data))

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(page)
    return path


# --------------------------------------------------------------------------- CLI

def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="python -m vidliboom.branching",
                                 description="Render a branching script to an interactive HTML player.")
    ap.add_argument("script")
    ap.add_argument("-o", "--out", default="output/branching", help="output folder")
    ap.add_argument("--check", action="store_true", help="only validate and print the graph")
    ap.add_argument("--player-only", action="store_true", help="skip rendering; use existing segments/<id>.mp4")
    ap.add_argument("--skip-existing", action="store_true", help="do not re-render segments that exist")
    ap.add_argument("--countdown", type=int, default=10)
    ap.add_argument("--auto-pick", default="first", choices=["first", "random"])
    a, passthrough = ap.parse_known_args(argv)

    story = parse_branching(open(a.script, encoding="utf-8").read())
    issues = validate(story)
    print(f"{story.title}: {len(story.segments)} scenes, start={story.start}, "
          f"{sum(s.is_choice for s in story)} choice points, {len(story.endings)} endings")
    for s in story:
        arrow = " | ".join(f"{c.label} -> {c.goto}" for c in s.choices) or (f"-> {s.next}" if s.next else "END")
        print(f"  {s.id:<14} {arrow}")
    for i in issues:
        print(" ", i)
    if any(i.level == "error" for i in issues):
        return 1
    if a.check:
        return 0
    if a.player_only:
        video_map = {sid: f"segments/{sid}.mp4" for sid in reachable(story)}
    else:
        video_map = render_all(story, a.out, subprocess_render_fn(*passthrough),
                               skip_existing=a.skip_existing)
    print("wrote", export_player(story, video_map, a.out, countdown=a.countdown, auto_pick=a.auto_pick))
    return 0


if __name__ == "__main__":
    sys.exit(main())
