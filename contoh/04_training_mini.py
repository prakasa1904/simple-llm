"""Bab 05 — Melihat model belajar dalam hitungan detik.

Model super kecil belajar bahwa setelah "Ibu kota Indonesia adalah"
kata berikutnya adalah "Jakarta". Perhatikan loss turun.

    python contoh/04_training_mini.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

kosakata = ["Ibu", "kota", "Indonesia", "adalah", "Jakarta", "pisang", "biru"]
ke_id = {w: i for i, w in enumerate(kosakata)}


class MiniLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(len(kosakata), 8)   # tiap kata = vektor 8 angka
        self.keluar = nn.Linear(8, len(kosakata))     # skor untuk tiap kata

    def forward(self, x):
        return self.keluar(self.embed(x).mean(dim=0))


model = MiniLM()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.05)

masukan = torch.tensor([ke_id[w] for w in ["Ibu", "kota", "Indonesia", "adalah"]])
target = torch.tensor(ke_id["Jakarta"])

for langkah in range(31):
    logits = model(masukan)                                          # 1. TEBAK
    loss = F.cross_entropy(logits.unsqueeze(0), target.unsqueeze(0))  # 2-3. CEK & HITUNG
    optimizer.zero_grad()
    loss.backward()                                                  # 4a. BACKPROP
    optimizer.step()                                                 # 4b. PERBAIKI

    if langkah % 5 == 0:
        p = F.softmax(logits, -1)[ke_id["Jakarta"]].item()
        print(f"step {langkah:2d} | loss {loss.item():.3f} | peluang 'Jakarta' {p:6.1%}")

print("\nPeluang akhir untuk setiap kata:")
with torch.no_grad():
    for w, p in zip(kosakata, F.softmax(model(masukan), -1).tolist()):
        print(f"  {w:>10} {p:6.1%}")
