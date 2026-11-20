import json
import os
import re
import shutil
import subprocess

import pytest

from vidliboom import script
from vidliboom.branching import (export_player, parse_branching, reachable,
                                 render_all, validate)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLE = os.path.join(ROOT, "examples", "bablu_choice.md")

SRC = """# Bablu ka Faisla
## Scene: Baarish | id=rain | mood=rain
BABLU [flail]: Arre! Baarish!
? Bablu kya kare?
-> Chhata kholo | goto=umbrella
-> Baarish mein naacho | goto=dance
## Scene: Chhata | id=umbrella | mood=day | next=home
DIDI: Smart move.
## Scene: Naach | id=dance | mood=rain
UNCLE [zen]: Wah beta.
## Scene: Ghar | id=home | mood=night | end
BABLU: Ghar aa gaye.
"""


def _issues(story, level=None):
    return [i for i in validate(story) if level is None or i.level == level]


def test_graph_shape():
    st = parse_branching(SRC)
    assert st.title == "Bablu ka Faisla"
    assert st.start == "rain"
    assert list(st.segments) == ["rain", "umbrella", "dance", "home"]
    rain = st["rain"]
    assert rain.prompt == "Bablu kya kare?"
    assert [(c.label, c.goto) for c in rain.choices] == [("Chhata kholo", "umbrella"), ("Baarish mein naacho", "dance")]
    assert rain.next is None and rain.is_choice and not rain.is_ending
    assert st["umbrella"].next == "home"          # explicit next=
    assert st["dance"].next == "home"             # falls through in file order
    assert st["home"].is_ending and st["home"].end
    assert [s.id for s in st.endings] == ["home"]
    assert _issues(st) == []


def test_segment_text_is_plain_script():
    st = parse_branching(SRC)
    text = st["rain"].script_text
    assert "->" not in text and "kya kare" not in text
    assert "id=" not in text and "goto" not in text
    sc = script.parse(text)
    assert sc.title == "Bablu ka Faisla"           # start segment keeps the story title
    assert len(sc.scenes) == 1 and sc.scenes[0].mood == "rain"
    assert [(l.speaker, l.pose) for l in sc.scenes[0].lines] == [("BABLU", "flail")]
    home = script.parse(st["home"].script_text)
    assert home.title == "Ghar" and home.scenes[0].mood == "night"   # later segments open on the scene name
    assert "end" not in st["home"].script_text.split("\n")[2]
    story_titles = parse_branching(SRC, segment_title="story")
    assert script.parse(story_titles["home"].script_text).title == "Bablu ka Faisla"


def test_last_scene_and_default_ids():
    st = parse_branching("# T\n## Scene: Pehla Scene\nA: x\n## Scene: Doosra\nB: y\n")
    assert list(st.segments) == ["pehla-scene", "doosra"]
    assert st["pehla-scene"].next == "doosra"
    assert st["doosra"].is_ending and not st["doosra"].end


def test_start_override():
    st = parse_branching("# T | start=b\n## Scene: A | id=a\nA: x\n## Scene: B | id=b | next=a\nB: y\n")
    assert st.start == "b"
    assert reachable(st) == ["b", "a"]


def test_missing_ids_are_errors():
    st = parse_branching(SRC.replace("goto=dance", "goto=dnace").replace("next=home", "next=hom"))
    errs = [str(i) for i in _issues(st, "error")]
    assert any("dnace" in e for e in errs)
    assert any("hom'" in e for e in errs)


def test_unreachable_warning_and_cycles_allowed():
    src = """# Loop
## Scene: A | id=a
X: one
? again?
-> Haan | goto=a
-> Nahi | goto=b
## Scene: B | id=b | end
X: two
## Scene: Lost | id=lost | end
X: three
"""
    st = parse_branching(src)
    issues = validate(st)
    assert not [i for i in issues if i.level == "error"]          # a -> a cycle is fine
    warn = [i for i in issues if i.level == "warning"]
    assert [i.segment for i in warn] == ["lost"]
    assert "unreachable" in warn[0].message


def test_endless_loop_warning():
    st = parse_branching("# L\n## Scene: A | id=a | next=b\nX: 1\n## Scene: B | id=b | next=a\nX: 2\n")
    msgs = [i.message for i in validate(st)]
    assert any("endless loop" in m for m in msgs)
    assert any("no ending is reachable" in m for m in msgs)


def test_other_errors():
    src = """# E
-> stray | goto=a
## Scene: A | id=a
## Scene: Dup | id=a | end
X: hi
"""
    st = parse_branching(src)
    errs = " ".join(str(i) for i in _issues(st, "error"))
    assert "outside any scene" in errs
    assert "more than one scene" in errs
    assert "no spoken lines" in errs
    assert list(st.segments) == ["a", "a-2"]


def test_end_with_choices_is_error():
    st = parse_branching("# E\n## Scene: A | id=a | end\nX: hi\n-> go | goto=b\n-> stay | goto=a\n## Scene: B | id=b\nX: yo\n")
    assert any("marked `end`" in i.message for i in _issues(st, "error"))


def test_example_file():
    st = parse_branching(open(EXAMPLE, encoding="utf-8").read())
    assert _issues(st) == [], [str(i) for i in validate(st)]
    assert len(st.segments) == 6
    assert sum(s.is_choice for s in st) == 2
    assert len(st.endings) == 3
    assert reachable(st) == list(st.segments) or set(reachable(st)) == set(st.segments)
    chars = os.path.join(ROOT, "assets", "characters")
    for seg in st:
        for ln in script.parse(seg.script_text).scenes[0].lines:
            assert os.path.exists(os.path.join(chars, ln.speaker, ln.pose + ".png")), (seg.id, ln.speaker, ln.pose)


def test_render_all_injected(tmp_path):
    st = parse_branching(SRC)
    calls = []

    def fake(text, out):
        calls.append((text, out))
        open(out, "wb").write(b"\0")

    vm = render_all(st, str(tmp_path), fake)
    assert vm == {sid: f"segments/{sid}.mp4" for sid in ["rain", "umbrella", "dance", "home"]}
    assert len(calls) == 4 and all(os.path.exists(o) for _, o in calls)
    assert (tmp_path / "segments" / "rain.md").read_text(encoding="utf-8") == st["rain"].script_text
    calls.clear()
    render_all(st, str(tmp_path), fake, skip_existing=True)
    assert calls == []


def _story_json(page):
    m = re.search(r"var STORY = (\{.*?\});\n", page, re.S)
    assert m, "story json not embedded"
    return json.loads(m.group(1))


def test_export_player(tmp_path):
    st = parse_branching(open(EXAMPLE, encoding="utf-8").read())
    vm = {sid: f"segments/{sid}.mp4" for sid in st.segments}
    path = export_player(st, vm, str(tmp_path), countdown=7)
    page = open(path, encoding="utf-8").read()
    assert "__TITLE__" not in page and "__STORY_JSON__" not in page
    assert "<title>Bablu ka Personal Mausam: Tum Chuno!</title>" in page
    assert not re.search(r"""(src|href)=["']https?://""", page), "no external resources"
    assert "@import" not in page
    data = _story_json(page)
    assert data["start"] == "dome" and data["countdown"] == 7
    assert data["segments"]["rain"]["choices"][0] == {"label": "Remote ka button dabao", "goto": "button"}
    assert data["segments"]["storm"]["ending"] is True
    assert data["segments"]["dome"]["video"] == "segments/dome.mp4"


def test_export_player_script_safe(tmp_path):
    st = parse_branching("# </script><b>x\n## Scene: A | id=a\nX: hi\n? </script>?\n-> <b>go</b> | goto=b\n-> no | goto=b\n## Scene: B | id=b\nX: bye\n")
    page = open(export_player(st, {"a": "a.mp4", "b": "b.mp4"}, str(tmp_path)), encoding="utf-8").read()
    assert page.count("</script>") == 1
    assert "&lt;/script&gt;" in page
    assert _story_json(page)["segments"]["a"]["choices"][0]["label"] == "<b>go</b>"


def test_export_player_rejects_bad_input(tmp_path):
    st = parse_branching(SRC)
    with pytest.raises(KeyError):
        export_player(st, {"rain": "r.mp4"}, str(tmp_path))
    export_player(st, {"rain": "r.mp4"}, str(tmp_path), allow_missing=True)
    with pytest.raises(ValueError):
        export_player(st, {k: "/abs/x.mp4" for k in st.segments}, str(tmp_path))
    with pytest.raises(ValueError):
        export_player(st, {k: "x.mp4" for k in st.segments}, str(tmp_path), auto_pick="last")
    bad = parse_branching(SRC.replace("goto=dance", "goto=nowhere"))
    with pytest.raises(ValueError):
        export_player(bad, {k: "x.mp4" for k in bad.segments}, str(tmp_path))


@pytest.mark.skipif(not shutil.which("node"), reason="node not installed")
def test_player_js_syntax(tmp_path):
    st = parse_branching(open(EXAMPLE, encoding="utf-8").read())
    page = open(export_player(st, {s: f"segments/{s}.mp4" for s in st.segments}, str(tmp_path)), encoding="utf-8").read()
    js = re.search(r"<script>(.*?)</script>", page, re.S).group(1)
    f = tmp_path / "player.js"
    f.write_text(js, encoding="utf-8")
    r = subprocess.run(["node", "--check", str(f)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
