import os

import pytest

from vidliboom import director, script
from vidliboom.script import Line

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLE = os.path.join(ROOT, "examples", "bablu_mausam.md")


@pytest.fixture
def assets(tmp_path):
    for name, poses in {"BABLU": ["idle", "proud", "flinch", "slump", "smug", "wave"],
                        "DIDI": ["idle", "giggle", "startled", "sideeye"]}.items():
        d = tmp_path / name
        d.mkdir()
        for p in poses:
            (d / f"{p}.png").write_bytes(b"")
    return str(tmp_path)


@pytest.mark.parametrize("text,emotion", [
    ("Haha, kya baat hai, maza aa gaya", "happy"),
    ("Wah! Zabardast!", "excited"),
    ("Arre! Yeh kya hua?", "surprised"),
    ("Uff, bas karo, mujhe gussa aa raha hai", "angry"),
    ("Sorry didi... mujhe dukh hai", "sad"),
    ("Bachao! Mujhe dar lag raha hai", "scared"),
    ("Shaant ho jao, aaram se", "calm"),
    ("Dekha? Maine kaha tha, obviously", "smug"),
    ("Kal school jaana hai", "neutral"),
    ("अरे! यह क्या?", "surprised"),
    ("मुझे डर लग रहा है", "scared"),
])
def test_detect_emotion(text, emotion):
    emo, inten = director.detect_emotion(text)
    assert emo == emotion
    assert 0.0 <= inten <= 1.0


def test_intensity_rises_with_punctuation_and_caps():
    _, calm = director.detect_emotion("wah")
    _, loud = director.detect_emotion("WAH WAH!!!")
    assert loud > calm
    assert director.detect_emotion("WHY ARE YOU HERE")[0] in ("angry", "excited")


def test_kyunki_is_not_kyun():
    assert director.detect_emotion("Kyunki tumhara mausam personal hai")[0] == "neutral"


def test_pick_pose_uses_only_available():
    poses = ["idle", "flinch", "worried"]
    assert director.pick_pose("surprised", poses) == "flinch"
    assert director.pick_pose("happy", poses) == "idle"  # nothing matches -> idle
    assert director.pick_pose("scared", poses, avoid="flinch") == "worried"
    assert director.pick_pose("angry", []) == "idle"


def test_available_poses(assets):
    assert director.available_poses("DIDI", assets) == ["giggle", "idle", "sideeye", "startled"]
    assert director.available_poses("NOBODY", assets) == []


def test_auto_pose_only_when_idle(assets):
    s = script.parse("# T\n## Scene: A\nBABLU: Arre! Kya hua?!\nDIDI [sideeye]: Haha, kuch nahi\n")
    dirs = director.direct(s, assets)
    assert dirs[0].pose == "flinch" and dirs[0].emotion == "surprised"
    assert dirs[1].pose == "sideeye"  # explicit pose is kept
    assert s.scenes[0].lines[0].pose == "idle"  # direct() does not mutate
    director.apply(s, dirs)
    assert s.scenes[0].lines[0].pose == "flinch"


def test_missing_character_falls_back_to_idle(assets):
    s = script.parse("# T\n## Scene: A\nGHOST: Wah! Kamaal!\n")
    assert director.direct(s, assets)[0].pose == "idle"


def test_punch_in_and_reaction(assets):
    s = script.parse("# T\n## Scene: A\nBABLU: Hello\nDIDI: ARRE! KYA?!\nBABLU: Tum kahan ja rahi ho?\nDIDI: Ghar\n")
    dirs = director.direct(s, assets)
    assert dirs[1].shot == "punch-in" and dirs[1].zoom > 1.3 and dirs[1].target == "DIDI"
    assert dirs[2].shot == "reaction" and dirs[2].target == "DIDI"


def test_joke_cuts_to_listener(assets):
    s = script.parse("# T\n## Scene: A\nBABLU: Main toh chand pe jaunga\nDIDI: Hahaha\n")
    d = director.direct(s, assets)[0]
    assert d.shot == "reaction" and d.target == "DIDI"


def test_no_shot_three_times_in_a_row():
    lines = [Line("BABLU" if i % 2 else "DIDI", "Arre! Wah! Kamaal!") for i in range(8)]
    s = script.Script("T", [script.Scene("A", lines=lines)])
    shots = [d.shot for d in director.direct(s, "/nonexistent")]
    assert all(not (a == b == c) for a, b, c in zip(shots, shots[1:], shots[2:]))
    assert set(shots) <= set(director.BASE_ZOOM)


def test_beat_timing():
    q, plain, dots = Line("A", "Tum kaun ho?"), Line("A", "Main Bablu hoon aur yeh meri didi hai"), Line("A", "Pata nahi...")
    reply = Line("B", "Main Bablu hoon aur yeh meri didi hai")
    assert director.beat(q, reply, "neutral") > director.beat(plain, reply, "neutral")
    assert director.beat(dots, reply, "neutral") > director.beat(q, reply, "neutral")
    fast = director.beat(Line("A", "Haan?"), Line("B", "Nahi."), "neutral")
    slow = director.beat(Line("A", "Haan?"), Line("A", "Nahi."), "neutral")
    assert fast < slow
    assert director.beat(plain, None, "neutral") > director.beat(plain, reply, "neutral")


def test_example_script():
    s = script.parse(open(EXAMPLE, encoding="utf-8").read())
    dirs = director.direct(s, os.path.join(ROOT, "assets", "characters"))
    assert len(dirs) == sum(len(sc.lines) for sc in s.scenes)
    for d in dirs:
        assert d.emotion in director.EMOTIONS
        assert 0 <= d.intensity <= 1 and d.zoom >= 1.0 and d.pause_after > 0
    by_text = {d.line.text.split()[0]: d for d in dirs}
    assert by_text["Arre!"].emotion == "surprised"
    assert by_text["Shaant"].emotion == "calm"
