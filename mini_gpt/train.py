"""Latih GPT mini dari nol pada sebuah file teks (pretraining).

Contoh:
    python mini_gpt/train.py
    python mini_gpt/train.py --langkah 3000 --n-layer 4 --n-embd 128 --data mini_gpt/data/teks.txt

Hasilnya disimpan ke mini_gpt/checkpoint.pt, lalu dipakai oleh generate.py.
"""

import argparse
import time
from dataclasses import asdict
from pathlib import Path

import torch

from model import GPT, GPTConfig
from tokenizer import CharTokenizer

FOLDER = Path(__file__).resolve().parent


def pilih_device(pilihan):
    if pilihan != "auto":
        return pilihan
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"  # GPU Apple Silicon
    return "cpu"


def ambil_batch(data, panjang_konteks, ukuran_batch, device):
    """Ambil potongan teks acak. Target = input yang digeser 1 posisi.

    data   : ... K  a  n  c  i  l     m  e ...
    x      :     K  a  n  c  i  l
    y      :        a  n  c  i  l  ␣
    """
    awal = torch.randint(len(data) - panjang_konteks - 1, (ukuran_batch,))
    x = torch.stack([data[i:i + panjang_konteks] for i in awal])
    y = torch.stack([data[i + 1:i + panjang_konteks + 1] for i in awal])
    return x.to(device), y.to(device)


@torch.no_grad()
def hitung_loss(model, data_latih, data_uji, args, device, n=50):
    """Rata-rata loss pada data latih dan data uji (tanpa memperbarui bobot)."""
    model.eval()
    hasil = {}
    for nama, data in [("latih", data_latih), ("uji", data_uji)]:
        total = 0.0
        for _ in range(n):
            x, y = ambil_batch(data, args.panjang_konteks, args.ukuran_batch, device)
            _, loss = model(x, y)
            total += loss.item()
        hasil[nama] = total / n
    model.train()
    return hasil


def main():
    p = argparse.ArgumentParser(description="Latih GPT mini dari nol")
    p.add_argument("--data", default=str(FOLDER / "data" / "teks.txt"))
    p.add_argument("--keluaran", default=str(FOLDER / "checkpoint.pt"))
    p.add_argument("--langkah", type=int, default=2000, help="jumlah step training")
    p.add_argument("--ukuran-batch", type=int, default=32)
    p.add_argument("--panjang-konteks", type=int, default=64)
    p.add_argument("--n-layer", type=int, default=2)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--n-embd", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--cek-setiap", type=int, default=100, help="cetak loss setiap N step")
    p.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = pilih_device(args.device)

    # 1. Data: baca teks, buat tokenizer, ubah ke ID token
    teks = Path(args.data).read_text(encoding="utf-8")
    tok = CharTokenizer.dari_teks(teks)
    data = torch.tensor(tok.encode(teks), dtype=torch.long)

    # 90% untuk latihan, 10% untuk uji (mendeteksi overfitting)
    batas = int(0.9 * len(data))
    data_latih, data_uji = data[:batas], data[batas:]
    if len(data_uji) <= args.panjang_konteks + 1:
        raise SystemExit("Data terlalu sedikit untuk panjang konteks ini. Tambah teks atau kecilkan --panjang-konteks.")

    # 2. Model
    cfg = GPTConfig(
        ukuran_kosakata=tok.ukuran_kosakata,
        panjang_konteks=args.panjang_konteks,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    )
    model = GPT(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    print(f"Device         : {device}")
    print(f"Teks           : {len(teks):,} karakter, kosakata {tok.ukuran_kosakata} karakter unik")
    print(f"Parameter model: {model.jumlah_parameter():,}")
    print(f"Loss awal ideal untuk tebakan acak: ln({tok.ukuran_kosakata}) = "
          f"{torch.log(torch.tensor(float(tok.ukuran_kosakata))).item():.2f}\n")

    # 3. Siklus training: tebak -> hitung loss -> backprop -> perbaiki bobot
    #    Model hanya disimpan ketika loss uji membaik (early stopping). Kalau loss latih
    #    terus turun tapi loss uji naik, model mulai menghafal (overfitting).
    mulai = time.time()
    loss_uji_terbaik, step_terbaik = float("inf"), 0
    for langkah in range(args.langkah + 1):
        if langkah % args.cek_setiap == 0:
            l = hitung_loss(model, data_latih, data_uji, args, device)
            tanda = ""
            if l["uji"] < loss_uji_terbaik:
                loss_uji_terbaik, step_terbaik = l["uji"], langkah
                torch.save({"config": asdict(cfg), "model": model.state_dict(),
                            "karakter": tok.karakter}, args.keluaran)
                tanda = "  <- terbaik, disimpan"
            print(f"step {langkah:5d} | loss latih {l['latih']:.3f} | loss uji {l['uji']:.3f} "
                  f"| {time.time() - mulai:5.1f} dtk{tanda}")
        if langkah == args.langkah:
            break

        x, y = ambil_batch(data_latih, args.panjang_konteks, args.ukuran_batch, device)
        _, loss = model(x, y)          # 1-3. tebak + hitung loss
        optimizer.zero_grad()
        loss.backward()                # 4a. backpropagation
        optimizer.step()               # 4b. perbaiki bobot

    # 4. Muat kembali versi terbaik
    print(f"\nModel terbaik (step {step_terbaik}, loss uji {loss_uji_terbaik:.3f}) "
          f"disimpan di {args.keluaran}")
    model.load_state_dict(torch.load(args.keluaran, map_location=device)["model"])

    # 5. Contoh hasil
    model.eval()
    awal = torch.tensor([tok.encode("Kancil ")], dtype=torch.long, device=device)
    hasil = model.generate(awal, token_baru=200, temperature=0.8, top_k=10)
    print("\nContoh tulisan model (awalan 'Kancil '):\n" + "-" * 50)
    print(tok.decode(hasil[0].tolist()))


if __name__ == "__main__":
    main()
