"""Bab 01 — BPE (Byte Pair Encoding) sederhana: bagaimana potongan token ditentukan.

Tidak ada aturan bahasa. BPE hanya:
  1. mulai dari huruf satu per satu,
  2. menghitung pasangan yang berdampingan,
  3. menggabungkan pasangan yang PALING SERING,
  4. mengulangi langkah 2-3 sebanyak N kali.

    python contoh/08_bpe_sederhana.py
    python contoh/08_bpe_sederhana.py --merge 50
    python contoh/08_bpe_sederhana.py --kata makanan berlarian "Kancilnya"
    python contoh/08_bpe_sederhana.py --data teks_saya.txt --merge 1000
"""

import argparse
import re
from collections import Counter
from pathlib import Path

AKAR = Path(__file__).resolve().parent.parent
KATA_UJI = ["makan", "makanan", "memakan", "dimakan", "berlari",
            "menghitung", "kancilnya", "komputer"]


def hitung_pasangan(kosakata):
    """Hitung semua pasangan token berdampingan, dibobot jumlah kemunculan kata."""
    pasangan = Counter()
    for kata, jumlah in kosakata.items():
        for a, b in zip(kata, kata[1:]):
            pasangan[(a, b)] += jumlah
    return pasangan


def gabungkan(kata, pasangan):
    """Ganti setiap kemunculan pasangan (a, b) di dalam kata menjadi satu token a+b."""
    a, b = pasangan
    hasil, i = [], 0
    while i < len(kata):
        if i < len(kata) - 1 and kata[i] == a and kata[i + 1] == b:
            hasil.append(a + b)
            i += 2
        else:
            hasil.append(kata[i])
            i += 1
    return tuple(hasil)


def latih_bpe(teks, jumlah_merge):
    """Pelajari daftar penggabungan (merge) dari teks. Urutan merge penting."""
    frekuensi_kata = Counter(re.findall(r"[a-z]+", teks.lower()))
    # Setiap kata mulai sebagai deretan huruf: "makan" -> ('m', 'a', 'k', 'a', 'n')
    kosakata = {tuple(k): n for k, n in frekuensi_kata.items()}

    daftar_merge = []
    for _ in range(jumlah_merge):
        pasangan = hitung_pasangan(kosakata)
        if not pasangan:
            break
        terbaik, jumlah = pasangan.most_common(1)[0]
        if jumlah < 2:            # tidak ada pasangan yang muncul lebih dari sekali
            break
        daftar_merge.append((terbaik, jumlah))
        kosakata = {gabungkan(k, terbaik): n for k, n in kosakata.items()}
    return daftar_merge


def potong(kata, daftar_merge):
    """Potong kata baru dengan menerapkan merge sesuai urutan saat dilatih."""
    token = tuple(kata.lower())
    for pasangan, _ in daftar_merge:
        token = gabungkan(token, pasangan)
    return list(token)


def main():
    p = argparse.ArgumentParser(description="BPE sederhana")
    p.add_argument("--data", default=str(AKAR / "mini_gpt" / "data" / "teks.txt"))
    p.add_argument("--merge", type=int, default=300, help="jumlah penggabungan maksimal")
    p.add_argument("--kata", nargs="+", default=KATA_UJI, help="kata yang ingin dipotong")
    args = p.parse_args()

    teks = Path(args.data).read_text(encoding="utf-8")
    daftar_merge = latih_bpe(teks, args.merge)
    huruf_awal = len(set(re.sub(r"[^a-z]", "", teks.lower())))

    print(f"Dilatih pada {args.data}")
    print(f"Huruf awal: {huruf_awal}, merge dipelajari: {len(daftar_merge)}, "
          f"ukuran kosakata: {huruf_awal + len(daftar_merge)}\n")

    print("10 MERGE PERTAMA (pasangan paling sering di data)")
    for i, ((a, b), n) in enumerate(daftar_merge[:10], 1):
        print(f"  {i:2d}. {a + ' + ' + b:9} -> {a + b:6} ({n} kali)")

    titik = sorted({k for k in [10, 60, 120, len(daftar_merge)] if k <= len(daftar_merge)})
    for k in titik:
        print(f"\nSETELAH {k} MERGE")
        for kata in args.kata:
            print(f"  {kata:12} -> {' · '.join(potong(kata, daftar_merge[:k]))}")

    print("\nCatatan: tidak ada aturan imbuhan di kode ini. Potongan yang cocok dengan")
    print("imbuhan (makan · an, di · makan) muncul karena imbuhan memang sering muncul;")
    print("potongan yang 'salah' (mem · akan) muncul karena statistik, bukan tata bahasa.")


if __name__ == "__main__":
    main()
