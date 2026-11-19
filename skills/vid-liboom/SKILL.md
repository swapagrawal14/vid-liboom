---
name: vid-liboom
description: Turn a written script into a narrated cartoon episode with rigged AI-art characters, procedural mood backgrounds, Cartesia voices, generated music and subtitles.
---

# vid-liboom skill

Use this skill when someone asks for a short cartoon, an animated explainer, or a story video from a script.

## Steps

1. Write the script in `docs/SCRIPT_FORMAT.md` format: a title, scenes with a `mood`, and lines as `NAME [pose]: text`.
2. Pick poses that exist in `assets/characters/<NAME>/`. Unknown characters render as stand-ins, so add PNG cut-outs for new ones.
3. Map each speaker to a voice in `voices.json`.
4. Render:
   ```bash
   python -m vidliboom script.md -o output/episode.mp4 --voice
   python -m vidliboom script.md -o output/reel.mp4 --voice --format vertical
   ```
5. Check a few frames (`ffmpeg -ss 5 -i output/episode.mp4 -frames:v 1 frame.png`) before sharing. Subtitles are also written as `episode.srt`.

## Tips

- Keep lines short (under 15 words) so subtitles fit on two lines.
- One mood per scene; a mood change gets a fade through black.
- Use `--no-camera` for static shots, such as a talking-head explainer.
