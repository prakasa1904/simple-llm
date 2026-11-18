"""Latih GPT mini: dari nol (pretraining) atau lanjutan dari checkpoint (continued pretraining).

Contoh — dari nol:
    python mini_gpt/train.py
    python mini_gpt/train.py --langkah 3000 --n-layer 4 --n-embd 128 --data mini_gpt/data/teks.txt

Contoh — lanjutan (continued pretraining, lihat materi/07-fine-tuning.md):
    python mini_gpt/train.py --lanjut-dari mini_gpt/checkpoint.pt \\
        --data mini_gpt/data/teks_tambahan.txt --pantau-data mini_gpt/data/teks.txt

    # dengan "replay": campur data baru dengan data lama agar tidak lupa
    python mini_gpt/train.py --lanjut-dari mini_gpt/checkpoint.pt \\
        --data mini_gpt/data/teks_tambahan.txt mini_gpt/data/teks.txt \\
        --pantau-data mini_gpt/data/teks.txt

Hasil dari nol disimpan ke mini_gpt/checkpoint.pt, hasil lanjutan ke mini_gpt/checkpoint_lanjut.pt.
"""

import argparse
import time
from dataclasses import asdict
from pathlib import Path

import torch

from model import GPT, GPTConfig
from tokenizer import CharTokenizer

FOLDER = Path(__file__).resolve().parent

# Default ukuran model untuk training dari nol
DEFAULT_MODEL = {"panjang_konteks": 64, "n_layer": 2, "n_head": 4, "n_embd": 64, "dropout": 0.2}


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
def hitung_loss(model, kumpulan_data, panjang_konteks, ukuran_batch, device, n=50):
    """Rata-rata loss untuk setiap data di kumpulan_data (tanpa memperbarui bobot)."""
    model.eval()
    hasil = {}
    for nama, data in kumpulan_data.items():
        total = 0.0
        for _ in range(n):
            x, y = ambil_batch(data, panjang_konteks, ukuran_batch, device)
            _, loss = model(x, y)
            total += loss.item()
        hasil[nama] = total / n
    model.train()
    return hasil


def periksa_karakter_asing(tok, teks, nama):
    """Tokenizer sudah dibekukan: karakter baru tidak bisa dipelajari, hanya dilewati."""
    asing = tok.karakter_asing(teks)
    if asing:
        jumlah = sum(teks.count(c) for c in asing)
        tampil = ", ".join(repr(c) for c in asing)
        print(f"PERINGATAN: {nama} berisi {jumlah} karakter yang tidak dikenal tokenizer "
              f"dan akan DILEWATI: {tampil}")


def main():
    p = argparse.ArgumentParser(description="Latih GPT mini (dari nol atau lanjutan)")
    p.add_argument("--data", nargs="+", default=[str(FOLDER / "data" / "teks.txt")],
                   help="satu atau beberapa file teks; tiap file dibagi 90%% latih / 10%% uji")
    p.add_argument("--lanjut-dari", default=None,
                   help="checkpoint untuk dilanjutkan (continued pretraining). "
                        "Tokenizer dan ukuran model diambil dari checkpoint ini.")
    p.add_argument("--pantau-data", default=None,
                   help="file teks lain yang loss-nya ikut dipantau (misalnya data lama, "
                        "untuk melihat apakah model mulai lupa)")
    p.add_argument("--keluaran", default=None,
                   help="default: checkpoint.pt (dari nol) atau checkpoint_lanjut.pt (lanjutan)")
    p.add_argument("--langkah", type=int, default=2000, help="jumlah step training")
    p.add_argument("--ukuran-batch", type=int, default=32)
    p.add_argument("--panjang-konteks", type=int, default=None)
    p.add_argument("--n-layer", type=int, default=None)
    p.add_argument("--n-head", type=int, default=None)
    p.add_argument("--n-embd", type=int, default=None)
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--learning-rate", type=float, default=None,
                   help="default: 1e-3 (dari nol) atau 3e-4 (lanjutan, lebih kecil agar tidak merusak)")
    p.add_argument("--cek-setiap", type=int, default=100, help="cetak loss setiap N step")
    p.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = pilih_device(args.device)
    lanjut = args.lanjut_dari is not None
    keluaran = args.keluaran or str(FOLDER / ("checkpoint_lanjut.pt" if lanjut else "checkpoint.pt"))
    lr = args.learning_rate or (3e-4 if lanjut else 1e-3)
    pilihan_model = {k: getattr(args, k) for k in DEFAULT_MODEL}

    # 1. Tokenizer + model
    if lanjut:
        if Path(keluaran).resolve() == Path(args.lanjut_dari).resolve():
            raise SystemExit("--keluaran sama dengan --lanjut-dari. Pakai nama file lain agar "
                             "checkpoint asli tidak tertimpa.")
        ckpt = torch.load(args.lanjut_dari, map_location="cpu")
        tok = CharTokenizer(ckpt["karakter"])            # tokenizer LAMA, dibekukan
        cfg = GPTConfig(**ckpt["config"])                # ukuran model LAMA
        diabaikan = [k for k, v in pilihan_model.items() if v is not None and v != getattr(cfg, k)]
        if diabaikan:
            print(f"Catatan: {', '.join('--' + k.replace('_', '-') for k in diabaikan)} diabaikan; "
                  f"ukuran model mengikuti checkpoint.")
        model = GPT(cfg)
        model.load_state_dict(ckpt["model"])
    else:
        teks_semua = "".join(Path(f).read_text(encoding="utf-8") for f in args.data)
        tok = CharTokenizer.dari_teks(teks_semua)       # tokenizer BARU dari data
        ukuran = {k: (v if v is not None else DEFAULT_MODEL[k]) for k, v in pilihan_model.items()}
        cfg = GPTConfig(ukuran_kosakata=tok.ukuran_kosakata, **ukuran)
        model = GPT(cfg)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # 2. Data: tiap file dibagi 90% latih / 10% uji, lalu digabung
    bagian_latih, bagian_uji, total_karakter = [], [], 0
    for f in args.data:
        teks = Path(f).read_text(encoding="utf-8")
        total_karakter += len(teks)
        periksa_karakter_asing(tok, teks, f)
        ids = torch.tensor(tok.encode(teks), dtype=torch.long)
        batas = int(0.9 * len(ids))
        bagian_latih.append(ids[:batas])
        bagian_uji.append(ids[batas:])
    data_latih, data_uji = torch.cat(bagian_latih), torch.cat(bagian_uji)
    if len(data_uji) <= cfg.panjang_konteks + 1:
        raise SystemExit("Data terlalu sedikit untuk panjang konteks ini. "
                         "Tambah teks atau kecilkan --panjang-konteks.")

    pantau = {"latih": data_latih, "uji": data_uji}
    if args.pantau_data:
        teks_pantau = Path(args.pantau_data).read_text(encoding="utf-8")
        data_pantau = torch.tensor(tok.encode(teks_pantau), dtype=torch.long)
        if len(data_pantau) > cfg.panjang_konteks + 1:
            pantau["pantau"] = data_pantau

    print(f"Mode           : {'LANJUTAN dari ' + args.lanjut_dari if lanjut else 'dari nol'}")
    print(f"Device         : {device}")
    print(f"Teks           : {total_karakter:,} karakter dari {len(args.data)} file, "
          f"kosakata {tok.ukuran_kosakata} karakter unik")
    print(f"Parameter model: {model.jumlah_parameter():,}")
    print(f"Learning rate  : {lr}")
    if "pantau" in pantau:
        print(f"Dipantau       : {args.pantau_data} (kolom 'pantau')")
    if not lanjut:
        print(f"Loss awal ideal untuk tebakan acak: ln({tok.ukuran_kosakata}) = "
              f"{torch.log(torch.tensor(float(tok.ukuran_kosakata))).item():.2f}")
    print()

    # 3. Siklus training: tebak -> hitung loss -> backprop -> perbaiki bobot
    #    Model hanya disimpan ketika loss uji membaik (early stopping). Kalau loss latih
    #    terus turun tapi loss uji naik, model mulai menghafal (overfitting).
    mulai = time.time()
    loss_uji_terbaik, step_terbaik = float("inf"), 0
    for langkah in range(args.langkah + 1):
        if langkah % args.cek_setiap == 0:
            l = hitung_loss(model, pantau, cfg.panjang_konteks, args.ukuran_batch, device)
            tanda = ""
            if l["uji"] < loss_uji_terbaik:
                loss_uji_terbaik, step_terbaik = l["uji"], langkah
                torch.save({"config": asdict(cfg), "model": model.state_dict(),
                            "karakter": tok.karakter}, keluaran)
                tanda = "  <- terbaik, disimpan"
            kolom_pantau = f" | pantau {l['pantau']:.3f}" if "pantau" in l else ""
            print(f"step {langkah:5d} | loss latih {l['latih']:.3f} | loss uji {l['uji']:.3f}"
                  f"{kolom_pantau} | {time.time() - mulai:5.1f} dtk{tanda}")
        if langkah == args.langkah:
            break

        x, y = ambil_batch(data_latih, cfg.panjang_konteks, args.ukuran_batch, device)
        _, loss = model(x, y)          # 1-3. tebak + hitung loss
        optimizer.zero_grad()
        loss.backward()                # 4a. backpropagation
        optimizer.step()               # 4b. perbaiki bobot

    # 4. Muat kembali versi terbaik
    print(f"\nModel terbaik (step {step_terbaik}, loss uji {loss_uji_terbaik:.3f}) "
          f"disimpan di {keluaran}")
    model.load_state_dict(torch.load(keluaran, map_location=device)["model"])

    # 5. Contoh hasil
    model.eval()
    awal = torch.tensor([tok.encode("Kancil ")], dtype=torch.long, device=device)
    hasil = model.generate(awal, token_baru=200, temperature=0.8, top_k=10)
    print("\nContoh tulisan model (awalan 'Kancil '):\n" + "-" * 50)
    print(tok.decode(hasil[0].tolist()))


if __name__ == "__main__":
    main()
