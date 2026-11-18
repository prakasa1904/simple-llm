# 07 — Fine-tuning dari Base LLM

Melatih model dari nol (seperti [`mini_gpt/`](../mini_gpt/)) bagus untuk **memahami**,
tetapi untuk **hasil yang berguna**, kita mulai dari model terbuka yang sudah dilatih
(keluarga Qwen, Llama, Gemma, Mistral, OLMo) lalu fine-tune.

Fine-tuning = **melanjutkan turun gunung dari posisi yang sudah dekat lembah**,
ke lembah yang sedikit berbeda (gaya bahasa atau domain Anda). Mesinnya sama persis
dengan training biasa ([05](05-training.md)); yang beda hanya titik awal dan datanya.

## Full fine-tuning vs LoRA vs QLoRA

| Metode | Yang dilatih | Kebutuhan GPU (model 7B) |
|---|---|---|
| **Full fine-tuning** | Semua bobot | Beberapa GPU 80 GB |
| **LoRA** | Adapter kecil (<1% parameter); bobot asli dibekukan | 1 GPU besar |
| **QLoRA** | LoRA di atas model yang dikompres ke 4-bit | 1 GPU 24 GB |

**Mulailah dengan LoRA/QLoRA.**

## Contoh kode (Hugging Face TRL + PEFT)

Lihat [`contoh/05_fine_tune_lora.py`](../contoh/05_fine_tune_lora.py). Intinya:

```python
trainer = SFTTrainer(
    model="Qwen/Qwen2.5-0.5B-Instruct",       # mulai dari model kecil
    train_dataset=dataset,                     # format {"messages": [...]}
    args=SFTConfig(output_dir="hasil", learning_rate=2e-4, num_train_epochs=2),
    peft_config=LoraConfig(r=16, lora_alpha=32, target_modules="all-linear",
                           task_type="CAUSAL_LM"),
)
trainer.train()
```

Setelah SFT berhasil, coba:
- **DPO** (`DPOTrainer`) dengan data `prompt/chosen/rejected`;
- **GRPO** (`GRPOTrainer`) — reinforcement learning dengan fungsi reward
  (misalnya memberi hadiah untuk jawaban matematika yang benar).

## Bisakah menambah data (pengetahuan) lewat fine-tuning?

Bisa, tetapi **hasilnya tergantung jenis data dan cara fine-tuning-nya**:

| Cara | Bentuk data | Yang bertambah | Cocok untuk |
|---|---|---|---|
| **Continued pretraining** (pretraining lanjutan) | Teks mentah `{"text": ...}` dalam jumlah besar | **Pengetahuan dan bahasa** domain baru | Domain baru (hukum, medis), bahasa daerah |
| **SFT** | Percakapan `{"messages": [...]}` | **Perilaku**: cara menjawab, gaya, format | Asisten khusus, gaya perusahaan |
| **RAG** (bukan fine-tuning) | Dokumen di database | **Fakta** yang dicari saat menjawab | Fakta spesifik, data yang sering berubah ([09](09-data-cepat-berubah.md)) |

### Continued pretraining: cara menambah pengetahuan

Secara teknis ini **fine-tuning dengan cara pretraining**: base model dilanjutkan dengan **teks
mentah** domain baru, memakai tujuan yang sama — menebak token berikutnya ([05](05-training.md)).

```
Base model (pengetahuan umum)
     ↓ + jutaan–miliaran token teks hukum Indonesia (undang-undang, putusan, jurnal)
Model yang paham istilah dan gaya bahasa hukum
     ↓ + SFT percakapan
Asisten hukum
```

Syarat agar berhasil:

1. **Datanya harus banyak** — biasanya jutaan sampai miliaran token. Seratus dokumen tidak cukup.
2. **Campur dengan data umum (*replay*)** — kalau hanya diberi data baru, model mulai **lupa**
   kemampuan lamanya (*catastrophic forgetting*). Lihat eksperimen di bawah.
3. **Learning rate lebih kecil** daripada pretraining awal, agar tidak merusak yang sudah dipelajari.
4. **Tokenizer tetap sama** — tokenizer sudah dibekukan ([01](01-cara-kerja-llm.md)). Kalau domain
   baru punya banyak istilah yang dipecah jadi banyak token, kadang tim **menambah token baru**,
   dan itu membuat training lebih mahal.

### SFT: bisa menambah fakta, tetapi lemah dan berisiko

Data SFT seperti ini memang bisa dibuat:

```json
{"messages": [
  {"role": "user", "content": "Kapan Toko Sinar berdiri?"},
  {"role": "assistant", "content": "Toko Sinar berdiri tahun 2015 di Bandung."}
]}
```

Tetapi ada tiga masalah:

1. **Satu contoh tidak cukup.** Model jarang mengingat fakta yang dilihat sekali-dua kali. Ia butuh
   fakta itu dalam **banyak variasi kalimat**, sama seperti di pretraining.
2. **Bisa memicu halusinasi.** Penelitian menunjukkan bahwa melatih model dengan fakta yang
   **benar-benar baru** baginya lewat SFT cenderung membuatnya lebih sering **mengarang**: model
   belajar pola *"jawab dengan yakin walaupun tidak tahu"*, bukan faktanya.
3. **LoRA kapasitasnya terbatas.** Bagus untuk mengubah perilaku, kurang cocok untuk "menanam"
   banyak pengetahuan baru.

Jadi SFT sebaiknya dipakai untuk mengajarkan **cara memakai pengetahuan** ("jawab singkat, sebut
sumbernya, bilang tidak tahu kalau tidak yakin"), bukan untuk menanam fakta.

### Panduan memilih

```
Apa yang ingin ditambahkan?
 ├─ Pengetahuan/bahasa domain baru, datanya BANYAK dan STABIL
 │     → CONTINUED PRETRAINING (teks mentah), lalu SFT
 ├─ Fakta spesifik (produk, kebijakan, dokumen) atau datanya SEDIKIT
 │     → RAG
 ├─ Data yang SERING BERUBAH
 │     → RAG / TOOL CALLING (materi 09)
 └─ Gaya, format, cara menjawab
       → SFT (LoRA cukup)
```

### Eksperimen: continued pretraining di GPT mini

GPT mini bisa dilanjutkan dari checkpoint dengan opsi `--lanjut-dari`. Tokenizer dan ukuran model
diambil dari checkpoint (dibekukan), learning rate otomatis lebih kecil (`3e-4`), dan hasilnya
disimpan ke `checkpoint_lanjut.pt` agar model asli tidak tertimpa.

Data barunya [`mini_gpt/data/teks_tambahan.txt`](../mini_gpt/data/teks_tambahan.txt), tentang
**tata surya** — topik yang sama sekali tidak ada di data awal. Opsi `--pantau-data` ikut
mengukur loss pada **data lama**, untuk melihat apakah model mulai lupa.

**Langkah 0 — latih model awal** (jika belum):

```bash
python mini_gpt/train.py
```

**Percobaan A — hanya data baru:**

```bash
python mini_gpt/train.py --lanjut-dari mini_gpt/checkpoint.pt \
    --data mini_gpt/data/teks_tambahan.txt \
    --pantau-data mini_gpt/data/teks.txt --langkah 1000
```

```
PERINGATAN: mini_gpt/data/teks_tambahan.txt berisi 8 karakter yang tidak dikenal tokenizer
            dan akan DILEWATI: '3', '5', '6', '8', 'C', 'V', 'Z'

step     0 | loss latih 2.267 | loss uji 2.162 | pantau 1.379   <- terbaik, disimpan
step   300 | loss latih 1.250 | loss uji 1.885 | pantau 1.953   <- terbaik, disimpan
step  1000 | loss latih 0.810 | loss uji 2.124 | pantau 2.486
```

- **Loss uji** (data baru) turun 2.16 → 1.89: model **belajar topik baru**.
- **Pantau** (data lama) naik 1.38 → 1.95 → 2.49: model **lupa** cerita Kancil (*catastrophic forgetting*).
  Diberi awalan "Kancil ", model kini menulis tentang "bumi" dan "matahari".
- **Peringatan karakter**: angka (8 planet, 365 hari), "V" (Venus), "C" (Cahaya), dan "Z" (Zuhal)
  tidak ada di data awal, sehingga **tidak bisa dipelajari** — persis batasan tokenizer yang dibekukan.

**Percobaan B — data baru + data lama (*replay*):**

```bash
python mini_gpt/train.py --lanjut-dari mini_gpt/checkpoint.pt \
    --data mini_gpt/data/teks_tambahan.txt mini_gpt/data/teks.txt \
    --pantau-data mini_gpt/data/teks.txt --langkah 1000
```

```
step     0 | loss latih 1.605 | loss uji 2.003 | pantau 1.379   <- terbaik, disimpan
step   500 | loss latih 1.410 | loss uji 1.902 | pantau 1.365   <- terbaik, disimpan
step  1000 | loss latih 1.321 | loss uji 1.908 | pantau 1.328
```

- **Pantau** tetap ±1.37: model **tidak lupa**, karena data lama ikut dilatih.
- Tulisannya kini mencampur keduanya: "Kancil buayang... bumi...".

(Di percobaan B, "loss uji" adalah gabungan data baru dan lama, jadi angkanya tidak bisa
dibandingkan langsung dengan percobaan A. Yang penting dibandingkan adalah kolom **pantau**.)

Coba hasilnya:

```bash
python mini_gpt/generate.py --checkpoint mini_gpt/checkpoint_lanjut.pt --prompt "Matahari "
```

> **Ringkasan:** data bisa ditambahkan setelah pretraining, tetapi caranya menentukan hasilnya.
> **Continued pretraining** (teks mentah, data banyak, dicampur data lama, learning rate kecil)
> menambah **pengetahuan dan bahasa** — tanpa *replay*, model melupakan yang lama. **SFT**
> terutama menambah **perilaku**; memakainya untuk menanam fakta baru lemah dan bisa membuat
> model lebih sering mengarang. Untuk fakta spesifik atau yang sering berubah, **RAG** tetap
> pilihan terbaik.

## Alat yang memudahkan

- **Unsloth** — LoRA lebih cepat & hemat memori, banyak notebook Colab siap pakai.
- **Axolotl / LLaMA-Factory** — fine-tuning cukup dengan file konfigurasi.
- **lm-evaluation-harness** — evaluasi standar; selalu bandingkan sebelum vs sesudah.
- **vLLM / Ollama** — menjalankan (serving) model hasil fine-tune.

## Nasihat praktis

1. **Mulai kecil** (0.5B – 1.5B). Iterasi dalam menit, bukan jam.
2. **Fine-tuning mengajarkan perilaku, bukan fakta.** Untuk menambah pengetahuan
   (misalnya dokumen perusahaan), **RAG** biasanya lebih baik dan lebih murah.
   Untuk data yang sering berubah, **jangan** fine-tuning ulang setiap ada data baru —
   lihat ⭐ [09 — Data yang Cepat Berubah](09-data-cepat-berubah.md).
3. **Chat template harus benar** — template yang salah sering membuat model rusak.
4. **Selalu evaluasi sebelum dan sesudah** — waspadai *catastrophic forgetting*.
5. **Kerjakan satu proyek nyata dari awal sampai akhir**: data → training → evaluasi → deploy.
