# simple-llm

Belajar cara kerja LLM (Large Language Model) dari dasar, dalam bahasa Indonesia,
sampai membangun dan melatih **GPT mini** sendiri di laptop.

Repo ini berisi:

- **`materi/`** — rangkuman materi, dari "apa itu LLM" sampai fine-tuning.
- **`contoh/`** — skrip kecil untuk mencoba tiap konsep (vektor, attention, sampling, training).
- **`mini_gpt/`** — LLM sungguhan (±250 baris) yang ditulis dari nol dengan PyTorch,
  dilatih dalam ±12 detik pada teks bahasa Indonesia.

---

## Mulai cepat

### 1. Persiapan

Butuh **Python 3.9+**. Disarankan memakai virtual environment:

```bash
git clone https://github.com/prakasa1904/simple-llm.git
cd simple-llm
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` hanya berisi PyTorch. Training otomatis memakai GPU jika ada
(NVIDIA CUDA atau Apple Silicon/MPS), kalau tidak memakai CPU — keduanya cukup cepat.

### 2. Latih GPT mini

```bash
python mini_gpt/train.py
```

Keluaran kira-kira:

```
Device         : mps
Teks           : 6,070 karakter, kosakata 49 karakter unik
Parameter model: 110,513
Loss awal ideal untuk tebakan acak: ln(49) = 3.89

step     0 | loss latih 4.108 | loss uji 4.095 |   0.3 dtk  <- terbaik, disimpan
...
step  1400 | loss latih 1.333 | loss uji 1.891 |   8.3 dtk  <- terbaik, disimpan
...
Model terbaik (step 1400, loss uji 1.891) disimpan di mini_gpt/checkpoint.pt
```

Loss turun dari ±3.9 (menebak acak) ke ±1.9. Model disimpan ke `mini_gpt/checkpoint.pt`.

### 3. Suruh model menulis

```bash
python mini_gpt/generate.py --prompt "Kancil "
```

Lihat tebakan model untuk huruf berikutnya:

```bash
python mini_gpt/generate.py --prompt "Jakarta adalah ibu kot" --lihat-peluang
```

```
Tebakan karakter berikutnya setelah 'Jakarta adalah ibu kot':
  'a'     89.0%  ███████████████████████████████████
  'e'      7.0%  ██
  'u'      2.0%
```

Bandingkan temperature rendah dan tinggi:

```bash
python mini_gpt/generate.py --prompt "Kancil " --temperature 0.2
python mini_gpt/generate.py --prompt "Kancil " --temperature 1.5 --top-k 40
```

> **Jangan berharap terlalu banyak:** dengan data hanya 6 ribu karakter, model belajar
> *bentuk* bahasa Indonesia (kata umum, spasi, tanda baca), tetapi kalimatnya belum
> bermakna. Itu normal — lihat [materi 08](materi/08-membangun-mini-gpt.md) untuk
> penjelasan dan cara memperbaikinya.

---

## Urutan belajar

Baca materi berurutan, lalu jalankan contoh yang menyertainya.

| # | Materi | Coba |
|---|---|---|
| 01 | [Cara kerja LLM](materi/01-cara-kerja-llm.md) — menebak token berikutnya, tokenizer per huruf vs per kata vs sub-word, cara kerja BPE, 3 tahap membuat LLM | `python contoh/07_biaya_tokenizer.py`<br>`python contoh/08_bpe_sederhana.py` |
| 02 | [Vektor dan embedding](materi/02-vektor-dan-embedding.md) — makna sebagai angka, hubungan tokenizer → tabel embedding | `python contoh/01_vektor_buah.py`<br>`python contoh/06_tokenizer_embedding.py` † |
| 03 | [Attention](materi/03-attention.md) — kata saling "melihat" (Q, K, V) | `python contoh/02_attention_manual.py` |
| 04 | [Sampling](materi/04-sampling.md) — temperature, top-k, top-p | `python contoh/03_sampling_temperature.py` |
| 05 | [Training](materi/05-training.md) — loss, gradient descent, backpropagation | `python contoh/04_training_mini.py` |
| 06 | [Bentuk data training](materi/06-data-training.md) — pretraining, SFT, preference | lihat `contoh/data_sft_contoh.jsonl` |
| 07 | [Fine-tuning](materi/07-fine-tuning.md) — LoRA/QLoRA dari model terbuka | `python contoh/05_fine_tune_lora.py` * |
| 08 | [Membangun GPT mini](materi/08-membangun-mini-gpt.md) — membedah `mini_gpt/` | `python mini_gpt/train.py` |
| 09 | ⭐ [Data yang cepat berubah](materi/09-data-cepat-berubah.md) — kapan fine-tuning, RAG, atau tool calling | — |

† Contoh 06 membaca GPT mini yang sudah dilatih, jadi jalankan `python mini_gpt/train.py` dulu.

\* Contoh 05 butuh paket tambahan dan mengunduh model ±1 GB dari Hugging Face:

```bash
pip install -r requirements-finetune.txt
python contoh/05_fine_tune_lora.py
```

Paling nyaman dijalankan di GPU NVIDIA (misalnya Google Colab). Di Mac/CPU bisa jalan
tapi lambat.

---

## Opsi perintah

### `mini_gpt/train.py`

| Opsi | Default | Arti |
|---|---|---|
| `--data` | `mini_gpt/data/teks.txt` | File teks untuk dilatih (UTF-8) |
| `--keluaran` | `mini_gpt/checkpoint.pt` | Lokasi menyimpan model |
| `--langkah` | `2000` | Jumlah step training |
| `--ukuran-batch` | `32` | Potongan teks per step |
| `--panjang-konteks` | `64` | Maksimal karakter yang dilihat sekaligus |
| `--n-layer` | `2` | Jumlah layer Transformer |
| `--n-head` | `4` | Jumlah attention head |
| `--n-embd` | `64` | Panjang vektor embedding (harus habis dibagi `--n-head`) |
| `--dropout` | `0.2` | Persentase neuron yang dimatikan saat training |
| `--learning-rate` | `0.001` | Ukuran langkah |
| `--cek-setiap` | `100` | Cetak loss setiap N step |
| `--device` | `auto` | `auto`, `cpu`, `cuda`, atau `mps` |

### `mini_gpt/generate.py`

| Opsi | Default | Arti |
|---|---|---|
| `--prompt` | `"Kancil "` | Teks awal |
| `--panjang` | `300` | Jumlah karakter baru |
| `--temperature` | `0.8` | `0` = greedy, rendah = fokus, tinggi = kreatif |
| `--top-k` | `10` | Hanya pilih dari k karakter terbaik |
| `--jumlah` | `1` | Berapa kali menulis |
| `--lihat-peluang` | — | Tampilkan 10 tebakan teratas untuk karakter berikutnya |
| `--seed` | — | Angka acak tetap agar hasil bisa diulang |
| `--checkpoint` | `mini_gpt/checkpoint.pt` | Model yang dipakai |

---

## Melatih dengan teks Anda sendiri

1. Siapkan file teks UTF-8, misalnya `teks_saya.txt`. Semakin banyak semakin bagus
   (idealnya ratusan ribu karakter atau lebih).
2. Latih:

   ```bash
   python mini_gpt/train.py --data teks_saya.txt --keluaran mini_gpt/model_saya.pt
   ```

3. Jika data Anda besar, perbesar model dan jumlah step:

   ```bash
   python mini_gpt/train.py --data teks_saya.txt --keluaran mini_gpt/model_saya.pt \
       --n-layer 4 --n-embd 128 --langkah 5000
   ```

4. Pakai:

   ```bash
   python mini_gpt/generate.py --checkpoint mini_gpt/model_saya.pt --prompt "Halo "
   ```

Perhatikan kolom **loss uji**. Jika loss latih terus turun tetapi loss uji naik, model
mulai menghafal (overfitting) — kecilkan model, naikkan dropout, atau tambah data.
Skrip otomatis hanya menyimpan versi dengan loss uji terbaik.

Model hanya mengenal karakter yang ada di data latih; karakter lain di prompt dilewati.

---

## Struktur repo

```
simple-llm/
├── README.md
├── requirements.txt              # PyTorch
├── requirements-finetune.txt     # paket tambahan untuk contoh 05
├── materi/                       # 01–09: rangkuman materi
├── contoh/
│   ├── 01_vektor_buah.py         # cosine similarity
│   ├── 02_attention_manual.py    # attention dihitung manual
│   ├── 03_sampling_temperature.py
│   ├── 04_training_mini.py       # model belajar "Jakarta"
│   ├── 05_fine_tune_lora.py      # fine-tuning LoRA (TRL + PEFT)
│   ├── 06_tokenizer_embedding.py # tokenizer -> tabel embedding GPT mini
│   ├── 07_biaya_tokenizer.py     # per huruf vs per kata: ukuran & biaya
│   ├── 08_bpe_sederhana.py       # BPE: bagaimana potongan token ditentukan
│   └── data_sft_contoh.jsonl     # contoh data SFT
└── mini_gpt/
    ├── tokenizer.py              # tokenizer per huruf
    ├── model.py                  # arsitektur GPT
    ├── train.py                  # training + early stopping
    ├── generate.py               # menulis teks
    └── data/teks.txt             # teks latih bahasa Indonesia
```

---

## Sumber belajar lanjutan

- **Andrej Karpathy — "Neural Networks: Zero to Hero"** (YouTube), terutama
  *"Let's build GPT: from scratch"*. `mini_gpt/` mengikuti pendekatan yang mirip dengan nanoGPT.
- **3Blue1Brown** — seri *Essence of Linear Algebra* dan *Neural Networks*.
- **Jay Alammar — *The Illustrated Transformer***.
- **Hugging Face LLM Course** (gratis) — transformers, datasets, fine-tuning.
- Buku: *Build a Large Language Model (From Scratch)* — Sebastian Raschka.
