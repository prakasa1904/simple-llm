"""Bab 02 — Vektor dan cosine similarity.

Setiap buah digambarkan dengan 3 angka: [manis, ukuran, merah].
Buah yang mirip akan punya cosine similarity mendekati 1.

    python contoh/01_vektor_buah.py
"""

import torch
import torch.nn.functional as F

buah = {
    "apel":     [7.0, 5.0, 8.0],
    "ceri":     [8.0, 1.0, 9.0],
    "semangka": [6.0, 10.0, 3.0],
    "lemon":    [1.0, 4.0, 0.0],
}

nama = list(buah)
vektor = torch.tensor([buah[n] for n in nama])

print("Cosine similarity antar buah (1.0 = arah sama, 0 = tidak berhubungan)\n")
print(" " * 10 + "".join(f"{n:>10}" for n in nama))
for i, a in enumerate(nama):
    baris = [F.cosine_similarity(vektor[i], vektor[j], dim=0).item() for j in range(len(nama))]
    print(f"{a:>10}" + "".join(f"{v:10.2f}" for v in baris))

# Cari buah yang paling mirip dengan buah "baru"
baru = torch.tensor([8.0, 2.0, 8.5])  # manis, kecil, merah
skor = F.cosine_similarity(baru.unsqueeze(0), vektor)
terbaik = nama[skor.argmax()]
print(f"\nBuah misterius [manis 8, ukuran 2, merah 8.5] paling mirip dengan: {terbaik}")
print("Inilah ide dasar pencarian dengan vektor (dipakai di RAG dan rekomendasi).")
