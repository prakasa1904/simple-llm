"""Bab 04 — Efek temperature dan top-k pada peluang kata berikutnya.

    python contoh/03_sampling_temperature.py
"""

import torch
import torch.nn.functional as F

kata = ["Saya", "Baik", "Kabar", "Luar", "Pisang"]
# Skor mentah (logits) dari model untuk kata setelah "Apa kabar?"
logits = torch.tensor([3.0, 2.5, 1.8, 1.6, -4.0])

print("Peluang tiap kata pada berbagai temperature\n")
print(f"{'kata':>8}" + "".join(f"{f'T={t}':>10}" for t in [0.2, 0.7, 1.0, 1.5]))
tabel = {t: F.softmax(logits / t, dim=-1) for t in [0.2, 0.7, 1.0, 1.5]}
for i, k in enumerate(kata):
    print(f"{k:>8}" + "".join(f"{tabel[t][i].item():10.1%}" for t in tabel))

print("\nT rendah -> fokus pada kata teratas. T tinggi -> peluang lebih rata (lebih kreatif).")

torch.manual_seed(0)
for t in [0.2, 1.0, 1.5]:
    pilihan = torch.multinomial(F.softmax(logits / t, dim=-1), num_samples=10, replacement=True)
    print(f"\n10 kali lempar dadu, T={t}: " + ", ".join(kata[i] for i in pilihan.tolist()))

# Top-k: hanya simpan k kata terbaik
k = 2
batas = torch.topk(logits, k).values[-1]
logits_topk = logits.masked_fill(logits < batas, float("-inf"))
print(f"\nDengan top-k={k}, hanya kata ini yang mungkin dipilih: "
      + ", ".join(f"{kata[i]} {p:.0%}" for i, p in enumerate(F.softmax(logits_topk, -1).tolist()) if p > 0))
