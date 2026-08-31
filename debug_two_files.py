import sys
import time

sys.stdout.reconfigure(line_buffering=True)

from chunkers import chunk_file

files = [
    "data/repos_source/huggingface_diffusers/setup.py",
    "data/repos_source/huggingface_diffusers/scripts/convert_original_stable_diffusion_to_diffusers.py",
]
for fp in files:
    print(f"debut: {fp}")
    code = open(fp, encoding="utf-8").read()
    t0 = time.time()
    chunks = chunk_file(fp, code, "cast_orig", 2000)
    print(f"fin: {fp} -> {len(chunks)} chunks en {time.time()-t0:.3f}s")

print("TERMINE")
