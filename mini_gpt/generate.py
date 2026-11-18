"""Pakai GPT mini yang sudah dilatih untuk menulis teks.

Contoh:
    python mini_gpt/generate.py --prompt "Kancil "
    python mini_gpt/generate.py --prompt "Kancil " --temperature 0.2
    python mini_gpt/generate.py --prompt "Ibu kota Indonesia adalah" --lihat-peluang
"""

import argparse
from pathlib import Path

import torch

from model import GPT, GPTConfig
from tokenizer import CharTokenizer

FOLDER = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description="Tulis teks dengan GPT mini")
    p.add_argument("--checkpoint", default=str(FOLDER / "checkpoint.pt"))
    p.add_argument("--prompt", default="Kancil ", help="teks awal")
    p.add_argument("--panjang", type=int, default=300, help="jumlah karakter baru")
    p.add_argument("--temperature", type=float, default=0.8, help="0 = greedy")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--jumlah", type=int, default=1, help="berapa kali menulis")
    p.add_argument("--lihat-peluang", action="store_true",
                   help="tampilkan 10 tebakan teratas untuk karakter berikutnya")
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    if not Path(args.checkpoint).exists():
        raise SystemExit(f"Checkpoint {args.checkpoint} belum ada. Jalankan dulu: python mini_gpt/train.py")
    if args.seed is not None:
        torch.manual_seed(args.seed)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    tok = CharTokenizer(ckpt["karakter"])
    model = GPT(GPTConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["model"])
    model.eval()

    asing = tok.karakter_asing(args.prompt)
    if asing:
        print(f"Catatan: karakter {asing} tidak ada di data latih dan dilewati.\n")
    ids = tok.encode(args.prompt)
    if not ids:
        ids = tok.encode("\n") or [0]
    idx = torch.tensor([ids], dtype=torch.long)

    if args.lihat_peluang:
        peluang = model.peluang_berikutnya(idx)
        nilai, indeks = torch.topk(peluang, 10)
        print(f"Tebakan karakter berikutnya setelah {args.prompt!r}:")
        for v, i in zip(nilai.tolist(), indeks.tolist()):
            print(f"  {tok.decode([i])!r:6} {v:6.1%}  " + "█" * int(v * 40))
        print()

    for n in range(args.jumlah):
        hasil = model.generate(idx, args.panjang, temperature=args.temperature, top_k=args.top_k)
        if args.jumlah > 1:
            print(f"--- hasil {n + 1} ---")
        print(tok.decode(hasil[0].tolist()))
        print()


if __name__ == "__main__":
    main()
