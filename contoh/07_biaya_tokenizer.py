"""Bab 01 — Token per huruf vs token per kata: mana yang lebih mahal?

Menghitung angka-angka di materi 01 dari teks latih GPT mini. Tanpa download.

    python contoh/07_biaya_tokenizer.py
    python contoh/07_biaya_tokenizer.py --data teks_saya.txt --n-embd 128
"""

import argparse
import re
from collections import Counter
from pathlib import Path

AKAR = Path(__file__).resolve().parent.parent


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

    # ---- 2. Ukuran model ----
    d = args.n_embd
    print(f"\n2. UKURAN MODEL: embedding + kepala = 2 x kosakata x n_embd ({d})")
    for nama, v in [("per huruf", len(set(huruf))), ("per kata, teks ini", len(hitung_kata)),
                    ("per kata, bahasa nyata", 100_000)]:
        print(f"   {nama:24} kosakata {v:>7,} -> {2 * v * d:>12,} parameter")
    print("   (seluruh GPT mini bawaan hanya ±110 ribu parameter)")

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
