"""Bab 02 — Hubungan tokenizer dan embedding, memakai GPT mini Anda sendiri.

Butuh model yang sudah dilatih:
    python mini_gpt/train.py
    python contoh/06_tokenizer_embedding.py
    python contoh/06_tokenizer_embedding.py --teks "Jakarta"
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

AKAR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(AKAR / "mini_gpt"))

from model import GPT, GPTConfig  # noqa: E402
from tokenizer import CharTokenizer  # noqa: E402


def tampil(c):
    return {"\n": "\\n", " ": "␣"}.get(c, c)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--teks", default="Kancil")
    p.add_argument("--checkpoint", default=str(AKAR / "mini_gpt" / "checkpoint.pt"))
    args = p.parse_args()

    if not Path(args.checkpoint).exists():
        raise SystemExit("Model belum dilatih. Jalankan dulu: python mini_gpt/train.py")

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    tok = CharTokenizer(ckpt["karakter"])
    model = GPT(GPTConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["model"])
    tabel = model.embed_token.weight.detach()   # (ukuran_kosakata, n_embd)

    # ---- Langkah 1: tokenizer ----
    print("LANGKAH 1 — TOKENIZER (kamus, tidak dilatih)")
    print(f"Kosakata {tok.ukuran_kosakata} token: " + " ".join(tampil(c) for c in tok.karakter))
    ids = tok.encode(args.teks)
    print(f"\n{args.teks!r} -> {[tampil(c) for c in args.teks if c in tok.ke_id]} -> {ids}")
    print("Nomor ini hanya urutan alfabet, belum punya makna.\n")

    # ---- Langkah 2: embedding ----
    print("LANGKAH 2 — EMBEDDING (tabel, dilatih)")
    print(f"Ukuran tabel embedding token: {tabel.shape[0]} baris x {tabel.shape[1]} kolom\n")
    for c, i in zip([c for c in args.teks if c in tok.ke_id], ids):
        angka = ", ".join(f"{v:5.2f}" for v in tabel[i][:6].tolist())
        print(f"  {tampil(c)!r:5} ID {i:2d} -> ambil baris {i:2d} -> [{angka}, ...]  ({tabel.shape[1]} angka)")

    hasil = model.embed_token(torch.tensor(ids))
    print(f"\nHasil untuk {args.teks!r}: matriks {tuple(hasil.shape)}  "
          f"(= {len(ids)} token x {tabel.shape[1]} angka)")
    sama = torch.equal(hasil[0].detach(), tabel[ids[0]])
    print(f"embed_token(ID) sama persis dengan baris tabel? {sama}\n")

    # ---- Token yang mirip ----
    print("TOKEN TERDEKAT (cosine similarity antar baris tabel)")
    normal = F.normalize(tabel, dim=1)
    kemiripan = normal @ normal.T
    for c in [".", "a", "K", " "]:
        if c not in tok.ke_id:
            continue
        i = tok.ke_id[c]
        skor = kemiripan[i].clone()
        skor[i] = -1  # abaikan dirinya sendiri
        top = torch.topk(skor, 3)
        daftar = ", ".join(f"'{tampil(tok.karakter[j])}' {v:.2f}"
                           for v, j in zip(top.values.tolist(), top.indices.tolist()))
        print(f"  {repr(tampil(c)):5} paling dekat dengan: {daftar}")
    print("\nModel kecil + data sedikit = sinyal lemah; sebagian pasangan masih acak.")

    # ---- Embedding posisi ----
    posisi = model.embed_posisi.weight.detach()
    print(f"\nEMBEDDING POSISI: tabel {posisi.shape[0]} x {posisi.shape[1]} "
          f"(satu baris per posisi 0..{posisi.shape[0] - 1})")
    print("Vektor yang masuk ke attention = baris tabel token + baris tabel posisi.")


if __name__ == "__main__":
    main()
