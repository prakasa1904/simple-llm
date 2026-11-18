# 08 — Membangun GPT Mini dari Nol

Folder [`mini_gpt/`](../mini_gpt/) berisi LLM sungguhan dalam ±250 baris Python.
Strukturnya **sama dengan GPT/Claude/Llama**, hanya jauh lebih kecil:

| | GPT mini ini | LLM besar |
|---|---|---|
| Parameter | ±110 ribu | miliaran – triliunan |
| Data latih | 6 ribu karakter | triliunan token |
| Tokenizer | per huruf (49 token) | sub-kata (±100 ribu token) |
| Layer | 2 | puluhan – lebih dari seratus |
| Waktu training | ±12 detik di laptop | berminggu-minggu di ribuan GPU |

## Peta file

| File | Isi | Materi terkait |
|---|---|---|
| `tokenizer.py` | Teks ↔ ID token (per huruf) | [01](01-cara-kerja-llm.md) |
| `model.py` | Embedding, attention, feed-forward, generate | [02](02-vektor-dan-embedding.md), [03](03-attention.md), [04](04-sampling.md) |
| `train.py` | Siklus training + early stopping | [05](05-training.md), [06](06-data-training.md) |
| `generate.py` | Menulis teks dengan model yang sudah dilatih | [04](04-sampling.md) |
| `data/teks.txt` | Teks latih: cerita Kancil, kota, makanan, percakapan | [06](06-data-training.md) |

## Alur di dalam `model.py`

```
"Kancil"  → tokenizer → [23, 31, 44, 33, 39, 42]
          → embed_token (makna)  +  embed_posisi (urutan)       ← bab 02
          → Block 1: LayerNorm → CausalSelfAttention → + residual ← bab 03
                     LayerNorm → FeedForward          → + residual
          → Block 2: (sama)
          → LayerNorm → Linear "kepala" → 49 skor (logits)
          → softmax + temperature/top-k → huruf berikutnya         ← bab 04
```

**Residual (`x = x + ...`)**: setiap layer menambahkan informasi baru ke vektor, bukan
menggantinya. Ini membuat model dalam bisa dilatih dengan stabil.

**LayerNorm**: menormalkan angka-angka di vektor agar tidak terlalu besar atau kecil.

## Alur di dalam `train.py`

1. Baca `teks.txt`, buat tokenizer, ubah ke ID token.
2. Bagi 90% data latih, 10% data uji.
3. Ulangi: ambil potongan acak `x`, target `y` = `x` digeser 1 → tebak → loss → backprop → perbaiki.
4. Setiap 100 step, cek loss latih dan uji. Simpan model **hanya jika loss uji membaik**.

## Pelajaran nyata: overfitting

Ketika model besar (4 layer, 128 dimensi, ±810 ribu parameter) dilatih 3.000 step
pada data sekecil ini, inilah yang terjadi:

```
step     0 | loss latih 4.052 | loss uji 4.030
step   250 | loss latih 1.593 | loss uji 1.973   ← titik terbaik
step  1000 | loss latih 0.158 | loss uji 2.903   ← loss uji NAIK
step  3000 | loss latih 0.112 | loss uji 3.785
```

Loss latih hampir 0, tapi loss uji naik. Hasil tulisannya **menyalin kata per kata** dari
data latih — model **menghafal**, bukan belajar pola. Itulah *overfitting*.

Solusi yang dipakai di sini:
- **Model lebih kecil** (default: 2 layer, 64 dimensi) — tidak cukup "kapasitas" untuk menghafal.
- **Dropout 0.2** — mematikan 20% neuron acak saat training.
- **Early stopping** — hanya menyimpan model dengan loss uji terbaik.

Solusi terbaik sebenarnya: **lebih banyak data**. Itulah sebabnya LLM sungguhan dilatih
pada triliunan token.

## Seperti apa hasilnya?

Dengan setelan default, model belajar **bentuk** bahasa Indonesia: kata umum ("adalah",
"saya", "dan", "yang", "kota", "pulau"), spasi, titik, dan huruf kapital di awal kalimat,
tetapi isinya belum bermakna:

```
Jakarta adalah ibu kota puta menggoranng.
Selah kota ta di dann pulau penghitup saya butu tumput dango
```

Dan saat ditanya huruf apa setelah "ibu kot", model yakin 89% jawabannya `a`:

```bash
python mini_gpt/generate.py --prompt "Jakarta adalah ibu kot" --lihat-peluang
```

Ini persis yang terjadi pada LLM besar, hanya skalanya jauh lebih kecil.

## Eksperimen yang disarankan

| Eksperimen | Perintah | Yang diamati |
|---|---|---|
| Temperature rendah vs tinggi | `generate.py --temperature 0.2` lalu `1.5` | 0.2 mengulang-ulang, 1.5 kacau |
| Model besar → overfitting | `train.py --n-layer 4 --n-embd 128 --dropout 0.1 --langkah 3000` | loss uji naik, teks menyalin data |
| Tanpa dropout | `train.py --dropout 0` | loss uji memburuk lebih cepat |
| Data Anda sendiri | `train.py --data teks_saya.txt` | model meniru gaya teks Anda |
| Konteks lebih pendek | `train.py --panjang-konteks 16` | kalimat lebih cepat "lupa" awalnya |

**Tips data sendiri:** semakin banyak teks semakin bagus. Coba gabungkan beberapa artikel
atau buku domain publik (ratusan ribu karakter), lalu naikkan ukuran model dan jumlah step.

## Setelah ini

Model dari nol bagus untuk **memahami**. Untuk hasil yang **berguna**, mulai dari model
terbuka yang sudah dilatih lalu fine-tune — lanjut ke [07 — Fine-tuning](07-fine-tuning.md).
