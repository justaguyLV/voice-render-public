"""Joins the per-line chunks made by habibi-tts into one voice track + segments.json (start/end per script line)."""
import sys, os, re, json, glob, numpy as np, soundfile as sf, scipy.signal as ss
chunk_dir, lines_json, out_dir = sys.argv[1:4]
L = json.load(open(lines_json, encoding="utf-8"))
files = sorted(glob.glob(f"{chunk_dir}/*.wav"), key=lambda p: int(re.match(r"(\d+)_", os.path.basename(p)).group(1)))
assert len(files) == len(L), f"{len(files)} chunks but {len(L)} lines"
SR = 24000; out = [np.zeros(int(.35 * SR), np.float32)]; segs = []; t = .35
def trim(x, th=0.01):
    idx = np.where(np.abs(x) > th * np.abs(x).max())[0]; return x[max(0, idx[0] - int(.02 * SR)):idx[-1] + int(.06 * SR)]
for f, (disp, spoken, pause) in zip(files, L):
    x, sr = sf.read(f, dtype="float32")
    if x.ndim > 1: x = x.mean(1)
    if sr != SR: x = ss.resample_poly(x, SR, sr).astype(np.float32)
    x = trim(x); out.append(np.zeros(int(pause * SR), np.float32)); t += pause
    segs.append({"start": round(t, 2), "end": round(t + len(x) / SR, 2), "draft": ""}); out.append(x); t += len(x) / SR
out.append(np.zeros(int(1.0 * SR), np.float32)); y = np.concatenate(out); os.makedirs(out_dir, exist_ok=True)
sf.write(f"{out_dir}/voice.wav", y, SR); json.dump(segs, open(f"{out_dir}/segs.json", "w"), ensure_ascii=False)
print("total", round(len(y) / SR, 1), "s; speech", round(sum(s["end"] - s["start"] for s in segs), 1), "s")
