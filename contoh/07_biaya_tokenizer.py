"""Bab 01 — Token per huruf vs token per kata: mana yang lebih mahal?

Menghitung angka-angka di materi 01 dari teks latih GPT mini. Tanpa download.

    python contoh/07_biaya_tokenizer.py
    python contoh/07_biaya_tokenizer.py --data teks_saya.txt --n-embd 128
"""

import argparse
import re
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

AKAR = Path(__file__).resolve().parent.parent
UKURAN_BATCH = 32


def ukur_waktu(ukuran_kosakata, n_embd, konteks, ulang=3):
    """Ukur waktu 1 step training (maju + backprop) untuk embedding + kepala output saja."""
    torch.manual_seed(0)
    embed = nn.Embedding(ukuran_kosakata, n_embd)   # tabel embedding (awal)
    kepala = nn.Linear(n_embd, ukuran_kosakata)     # kepala output (akhir)
    x = torch.randint(ukuran_kosakata, (UKURAN_BATCH, konteks))
    y = torch.randint(ukuran_kosakata, (UKURAN_BATCH, konteks))

    def satu_step():
        skor = kepala(embed(x))                      # 1 skor untuk SETIAP token kosakata
        F.cross_entropy(skor.view(-1, ukuran_kosakata), y.view(-1)).backward()

    satu_step()  # pemanasan
    mulai = time.perf_counter()
    for _ in range(ulang):
        satu_step()
    return (time.perf_counter() - mulai) / ulang * 1000


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(AKAR / "mini_gpt" / "data" / "teks.txt"))
    p.add_argument("--n-embd", type=int, default=64, help="panjang vektor embedding")
    p.add_argument("--konteks", type=int, default=64, help="panjang konteks dalam token")
    args = p.parse_args()

    teks = Path(args.data).read_text(encoding="utf-8")
    huruf = list(teks)
    kata = re.findall(r"\w+|[^\w\s]", teks)   # kata dan tanda baca
    hitung_kata = Counter(kata)

    # ---- 1. Ukuran kosakata ----
    print("1. UKURAN KOSAKATA (jumlah token berbeda)")
    print(f"   per huruf : {len(set(huruf)):>7,}")
    print(f"   per kata  : {len(hitung_kata):>7,}")
    jarang = sum(1 for n in hitung_kata.values() if n == 1)
    print(f"   kata yang hanya muncul 1 kali: {jarang} dari {len(hitung_kata)} "
          f"-> barisnya hampir tidak bisa belajar")

    # ---- 2. Biaya tabel embedding + kepala output ----
    d = args.n_embd
    print(f"\n2. BIAYA TABEL EMBEDDING + KEPALA OUTPUT (n_embd = {d}, batch {UKURAN_BATCH} x "
          f"{args.konteks} token, diukur di CPU)")
    print(f"   {'':24} {'parameter':>11} {'file':>9} {'perkalian/token':>16} "
          f"{'memori skor':>12} {'waktu/step':>11}")
    for nama, v in [("per huruf", len(set(huruf))), ("per kata, teks ini", len(hitung_kata)),
                    ("per kata, bahasa nyata", 100_000)]:
        parameter = 2 * v * d
        file_mb = parameter * 4 / 1e6                       # 4 byte per angka (float32)
        skor_mb = UKURAN_BATCH * args.konteks * v * 4 / 1e6  # 1 skor per token kosakata, per posisi
        waktu = ukur_waktu(v, d, args.konteks)
        print(f"   {nama:24} {parameter:>11,} {file_mb:>7.2f}MB {d * v:>16,} "
              f"{skor_mb:>10.1f}MB {waktu:>9.1f}ms")
    print("   (seluruh GPT mini bawaan hanya ±110 ribu parameter;"
          " waktu berbeda di tiap komputer)")

    # ---- 3. Panjang teks ----
    rasio = len(huruf) / len(kata)
    print("\n3. PANJANG TEKS YANG SAMA (jumlah token)")
    print(f"   per huruf : {len(huruf):>7,} token")
    print(f"   per kata  : {len(kata):>7,} token")
    print(f"   per huruf ±{rasio:.1f}x lebih panjang -> ±{rasio:.1f}x lebih banyak langkah menulis")
    print(f"   attention membandingkan tiap token dengan semua token sebelumnya "
          f"-> ±{rasio ** 2:.0f}x lebih berat")
    print(f"   konteks {args.konteks} token ≈ {args.konteks / rasio:.0f} kata (per huruf) "
          f"vs {args.konteks} kata (per kata)")

    # ---- 4. Kata berimbuhan ----
    print("\n4. KATA BERIMBUHAN (per kata = baris terpisah yang tidak saling kenal)")
    kecil = {k.lower() for k in hitung_kata}
    for akar in ["makan", "hitung", "lari"]:
        keluarga = sorted(k for k in kecil if akar in k)
        if keluarga:
            print(f"   {akar:7} -> {', '.join(keluarga)}")

    # ---- 5. Kata yang tidak dikenal ----
    print("\n5. KATA BARU YANG TIDAK ADA DI DATA LATIH")
    for baru in ["dimakan", "berlarian", "Kancilnya"]:
        ada_kata = baru in hitung_kata
        ada_huruf = all(c in set(huruf) for c in baru)
        print(f"   {baru:10} per kata: {'dikenal' if ada_kata else 'TIDAK DIKENAL'}"
              f" | per huruf: {'bisa dieja' if ada_huruf else 'ada huruf asing'}")


if __name__ == "__main__":
    main()
