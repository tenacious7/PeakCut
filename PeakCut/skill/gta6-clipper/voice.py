# Your voice on the clip: places your recorded line at AT seconds and ducks the game/streamer audio under it.
# Usage: python voice.py clip.mp4 line.wav AT out.mp4
# The line is what makes the clip yours (commentary = transformation). Video is copied untouched.
import subprocess, sys

def main(clip, line, at, out):
    ms = int(float(at) * 1000)
    graph = (f"[1:a]loudnorm=I=-16:TP=-1.5,aresample=48000,aformat=channel_layouts=stereo,adelay={ms}|{ms},apad,asplit=2[v][sc];"
             "[0:a]aresample=48000,aformat=channel_layouts=stereo[g];"
             # sidechain: the game drops about 15 dB only while you talk, returns over 0.4 s
             "[g][sc]sidechaincompress=threshold=0.015:ratio=20:attack=15:release=400:makeup=1[duck];"
             "[duck][v]amix=inputs=2:duration=first:normalize=0,alimiter=limit=0.95[a]")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", clip, "-i", line, "-filter_complex", graph,
                    "-map", "0:v", "-map", "[a]", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", out], check=True)
    print("ok", out)

if __name__ == "__main__":
    main(*sys.argv[1:5])
