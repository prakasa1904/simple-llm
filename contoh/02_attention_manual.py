"""Bab 03 — Attention dihitung manual, langkah demi langkah.

Kalimat: "Ular itu mengeluarkan bisa".
Kita buat vektor Q, K, V kecil (4 angka) secara manual supaya terlihat jelas
bahwa kata "bisa" memberi perhatian paling besar pada kata "Ular".

    python contoh/02_attention_manual.py
"""

import math

import torch
import torch.nn.functional as F

kata = ["Ular", "itu", "mengeluarkan", "bisa"]

# Di model sungguhan, Q/K/V = vektor kata x matriks bobot yang dipelajari.
# Di sini kita isi tangan. Dimensi pertama kira-kira berarti "hewan/racun".
K = torch.tensor([
    [3.0, 0.2, 0.1, 0.0],   # Ular          -> "saya hewan berbisa"
    [0.0, 0.1, 0.3, 0.1],   # itu           -> kata penunjuk, tidak penting
    [1.0, 0.5, 0.0, 0.2],   # mengeluarkan  -> kata kerja, agak relevan
    [0.5, 0.2, 0.2, 0.5],   # bisa          -> dirinya sendiri
])
V = torch.tensor([
    [1.0, 0.0, 0.0, 0.0],   # makna "hewan/racun"
    [0.0, 0.1, 0.0, 0.0],
    [0.2, 0.0, 1.0, 0.0],   # makna "aksi"
    [0.0, 0.0, 0.0, 1.0],   # makna "bisa" yang masih ambigu
])
Q = torch.tensor([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [1.0, 0.3, 0.1, 0.2],   # "bisa" bertanya: "apakah ada hewan/racun di sekitarku?"
])

d = Q.shape[-1]

# Langkah 1: skor = Q . K^T / sqrt(d)
skor = Q @ K.T / math.sqrt(d)

# Causal mask: kata hanya boleh melihat kata sebelumnya (dan dirinya sendiri)
mask = torch.tril(torch.ones(len(kata), len(kata)))
skor = skor.masked_fill(mask == 0, float("-inf"))

# Langkah 2: softmax -> persentase
perhatian = F.softmax(skor, dim=-1)

# Langkah 3: campur Value
hasil = perhatian @ V

print("Persentase perhatian (baris = kata yang sedang diperbarui, kolom = kata yang dilihat)\n")
print(" " * 14 + "".join(f"{k:>14}" for k in kata))
for i, k in enumerate(kata):
    print(f"{k:>14}" + "".join(
        f"{perhatian[i, j].item():13.0%} " if j <= i else f"{'-':>14}" for j in range(len(kata))))

print("\nTanda '-' = diblokir causal mask (tidak boleh mengintip ke depan).")
print(f"\nVektor baru 'bisa' = {[round(v, 2) for v in hasil[-1].tolist()]}")
print("Angka pertama (makna 'hewan/racun') paling besar -> 'bisa' dibaca sebagai racun.")
