"""Usage: python -m vidliboom examples/bablu_mausam.md -o output/episode.mp4 [--voice]"""
import argparse
import os
from . import script as S, planner, render, voice, director


def main():
    ap = argparse.ArgumentParser(prog="vidliboom")
    ap.add_argument("script")
    ap.add_argument("-o", "--out", default="output/episode.mp4")
    ap.add_argument("--voice", action="store_true", help="voice lines with Cartesia (needs CARTESIA_API_KEY)")
    ap.add_argument("--voices", default="voices.json", help="speaker to voice id map")
    ap.add_argument("--lang", default="hi")
    ap.add_argument("--format", default="landscape", choices=["landscape", "vertical", "square"])
    ap.add_argument("--no-camera", action="store_true", help="disable the zoom toward the speaker")
    ap.add_argument("--no-director", action="store_true", help="skip auto emotion, pose, shot and timing")
    ap.add_argument("--no-weather", action="store_true", help="skip the personal emotion-weather above heads")
    ap.add_argument("--no-lipsync", action="store_true", help="skip lip-sync and blinks")
    ap.add_argument("--audio", help="use this audio track instead of the generated mix")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    sc = S.parse(open(a.script, encoding="utf-8").read())
    voicer = None
    if a.voice:
        vmap = voice.load_voices(a.voices)
        def voicer(ln):
            vid = vmap.get(ln.speaker) or vmap.get("DEFAULT")
            return voice.read_wav(voice.tts(ln.text, vid, lang=a.lang)) if vid else None
    dirs = None
    if not a.no_director:
        dirs = director.direct(sc)
        director.apply(sc, dirs)
        for d in dirs:
            print(f"  {d.line.speaker:6} {d.emotion:9} {d.pose:8} {d.shot}")
    pauses = {id(d.line): d.pause_after for d in dirs} if dirs else None
    shots, total = planner.plan(sc, voicer=voicer, pauses=pauses)
    print(f"{sc.title}: {len(sc.scenes)} scenes, {len(shots)} shots, {total:.1f}s")
    print("wrote", render.render(sc, shots, total, a.out, a.audio, a.format, not a.no_camera,
                                       directions=dirs, weather=not a.no_weather, lips=not a.no_lipsync))


if __name__ == "__main__":
    main()
