"""GPT mini: Transformer decoder kecil yang ditulis dari nol.

Strukturnya sama dengan LLM besar, hanya ukurannya jauh lebih kecil:

    ID token -> embedding token + embedding posisi
             -> [Block] x n_layer   (tiap Block = attention + feed-forward)
             -> LayerNorm -> Linear -> skor (logits) untuk setiap token kosakata

Lihat materi/03-attention.md untuk penjelasan attention.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    ukuran_kosakata: int        # jumlah token berbeda (di sini: jumlah karakter unik)
    panjang_konteks: int = 64   # maksimal token yang bisa "dilihat" sekaligus
    n_layer: int = 2            # jumlah Block yang ditumpuk
    n_head: int = 4             # jumlah attention head per Block
    n_embd: int = 64            # panjang vektor embedding
    dropout: float = 0.2        # mematikan sebagian neuron acak saat training (anti-hafalan)


class CausalSelfAttention(nn.Module):
    """Multi-head attention dengan causal mask (tidak boleh mengintip ke depan)."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0, "n_embd harus habis dibagi n_head"
        self.n_head = cfg.n_head
        self.dim_head = cfg.n_embd // cfg.n_head

        # Satu Linear menghasilkan Q, K, dan V sekaligus (3 x n_embd).
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.proyeksi = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.dropout = nn.Dropout(cfg.dropout)

        # Matriks segitiga bawah: posisi i hanya boleh melihat posisi 0..i
        mask = torch.tril(torch.ones(cfg.panjang_konteks, cfg.panjang_konteks))
        self.register_buffer("mask", mask.view(1, 1, cfg.panjang_konteks, cfg.panjang_konteks))

    def forward(self, x):
        B, T, C = x.shape  # batch, panjang urutan, panjang vektor

        # Buat kartu Query, Key, Value untuk setiap token
        q, k, v = self.qkv(x).split(C, dim=2)
        # Pecah menjadi beberapa head: (B, n_head, T, dim_head)
        q = q.view(B, T, self.n_head, self.dim_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.dim_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.dim_head).transpose(1, 2)

        # Langkah 1: bandingkan Query dengan semua Key -> skor (B, n_head, T, T)
        skor = (q @ k.transpose(-2, -1)) / math.sqrt(self.dim_head)
        # Causal mask: skor untuk token "masa depan" dibuat -inf (peluang 0)
        skor = skor.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))

        # Langkah 2: softmax -> persentase perhatian
        perhatian = F.softmax(skor, dim=-1)
        perhatian = self.dropout(perhatian)

        # Langkah 3: campur Value sesuai persentase
        y = perhatian @ v                                   # (B, n_head, T, dim_head)
        y = y.transpose(1, 2).contiguous().view(B, T, C)    # gabungkan semua head
        return self.dropout(self.proyeksi(y))


class FeedForward(nn.Module):
    """MLP sederhana yang mengolah tiap token secara terpisah."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.jaringan = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
            nn.GELU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x):
        return self.jaringan(x)


class Block(nn.Module):
    """Satu layer Transformer: attention lalu feed-forward, masing-masing dengan
    LayerNorm dan sambungan residual (x + ...) agar training stabil."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.ff = FeedForward(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))  # token saling bertukar informasi
        x = x + self.ff(self.ln2(x))    # tiap token mengolah informasinya
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.embed_token = nn.Embedding(cfg.ukuran_kosakata, cfg.n_embd)   # ID -> vektor makna
        self.embed_posisi = nn.Embedding(cfg.panjang_konteks, cfg.n_embd)  # posisi -> vektor posisi
        self.dropout = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_akhir = nn.LayerNorm(cfg.n_embd)
        self.kepala = nn.Linear(cfg.n_embd, cfg.ukuran_kosakata)          # vektor -> skor tiap token

    def jumlah_parameter(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, target=None):
        """idx: (B, T) ID token. target: (B, T) token yang benar (idx digeser 1).

        Mengembalikan logits (B, T, ukuran_kosakata) dan loss (jika target diberikan).
        """
        B, T = idx.shape
        assert T <= self.cfg.panjang_konteks, "teks lebih panjang dari panjang_konteks"

        posisi = torch.arange(T, device=idx.device)
        x = self.embed_token(idx) + self.embed_posisi(posisi)  # makna + posisi
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        logits = self.kepala(self.ln_akhir(x))

        loss = None
        if target is not None:
            # Cross-entropy = -log(peluang token yang benar), dirata-rata semua posisi
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1))
        return logits, loss

    @torch.no_grad()
    def peluang_berikutnya(self, idx):
        """Peluang (softmax) untuk token berikutnya setelah urutan idx (1, T)."""
        idx = idx[:, -self.cfg.panjang_konteks:]
        logits, _ = self(idx)
        return F.softmax(logits[0, -1], dim=-1)

    @torch.no_grad()
    def generate(self, idx, token_baru, temperature=1.0, top_k=None):
        """Tulis token_baru token, satu per satu (lihat materi/04-sampling.md).

        temperature 0  -> greedy (selalu ambil yang teratas)
        temperature <1 -> lebih fokus, >1 -> lebih kreatif
        top_k          -> hanya pertimbangkan k token terbaik
        """
        for _ in range(token_baru):
            idx_konteks = idx[:, -self.cfg.panjang_konteks:]  # potong ke panjang konteks
            logits, _ = self(idx_konteks)
            logits = logits[:, -1, :]                          # hanya posisi terakhir

            if temperature == 0:
                berikutnya = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    batas = torch.topk(logits, min(top_k, logits.size(-1))).values[:, [-1]]
                    logits[logits < batas] = float("-inf")
                peluang = F.softmax(logits, dim=-1)
                berikutnya = torch.multinomial(peluang, num_samples=1)  # lempar dadu berbobot

            idx = torch.cat([idx, berikutnya], dim=1)          # tambahkan, lalu ulangi
        return idx
