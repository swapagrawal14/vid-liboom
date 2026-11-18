"""Usage: python -m vidliboom examples/bablu_mausam.md -o episode.mp4"""
import argparse
from . import script as S, planner, render


def main():
    ap = argparse.ArgumentParser(prog="vidliboom")
    ap.add_argument("script")
    ap.add_argument("-o", "--out", default="output/episode.mp4")
    ap.add_argument("--audio", help="optional voice/music track")
    a = ap.parse_args()
    import os; os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    sc = S.parse(open(a.script, encoding="utf-8").read())
    shots, total = planner.plan(sc)
    print(f"{sc.title}: {len(sc.scenes)} scenes, {len(shots)} shots, {total:.1f}s")
    print("wrote", render.render(sc, shots, total, a.out, a.audio))


if __name__ == "__main__":
    main()
