# 01 — Cara Kerja LLM

## Ide utama

LLM (Large Language Model) hanya melakukan **satu hal**: diberi sebuah teks, ia
**menebak token berikutnya**. Mengobrol, menulis kode, menjawab soal — semuanya
muncul dari kemampuan menebak kata berikutnya dengan sangat baik, satu token
setiap kali.

## Alur dari teks ke jawaban

```
Teks → Tokenizer → ID token → Embedding (vektor) → [Layer Transformer × N] → Peluang token berikutnya
```

| Bagian | Fungsi | Dibahas di |
|---|---|---|
| **Tokenizer** | Memecah teks jadi potongan (token), tiap potongan diberi nomor | bab ini |
| **Embedding** | Mengubah tiap nomor token menjadi vektor (deretan angka) | [02](02-vektor-dan-embedding.md) |
| **Attention** | Tiap token "melihat" token sebelumnya untuk memahami konteks | [03](03-attention.md) |
| **Feed-forward (MLP)** | Mengolah informasi tiap token; banyak "pengetahuan" tersimpan di sini | [03](03-attention.md) |
| **Output + sampling** | Memberi peluang untuk setiap kata, lalu memilih satu | [04](04-sampling.md) |

## Tokenizer

```
"how are you?"  →  ["how", " are", " you", "?"]  →  [4438, 527, 499, 30]
```

- Model besar memakai potongan kata (sub-word), misalnya `"belajar"` → `["bel", "ajar"]`.
- Model mini di folder [`mini_gpt/`](../mini_gpt/) memakai **tokenizer per huruf**
  (setiap karakter = 1 token) karena paling mudah dipahami.

## Model menulis jawaban satu token setiap kali

```
Input: Apa kabar?                        → tebak "Saya"
Input: Apa kabar? Saya                   → tebak "baik"
Input: Apa kabar? Saya baik              → tebak ","
Input: Apa kabar? Saya baik,             → tebak "terima"
...                                      → tebak [SELESAI] → berhenti
```

Model **tidak mencari jawaban di database**. Ia menulis jawabannya kata demi kata,
setiap kali bertanya: *"dengan semua teks sejauh ini, kata apa yang paling mungkin berikutnya?"*
Itulah sebabnya chatbot menampilkan jawaban seperti sedang diketik.

## Tiga tahap membuat LLM

| Tahap | Yang terjadi | Biaya | Siapa |
|---|---|---|---|
| **Pretraining** | Menebak kata berikutnya pada triliunan token teks internet → **base model** | Sangat mahal (ribuan GPU) | Lab besar |
| **Fine-tuning (SFT)** | Dilatih dengan contoh tanya-jawab → bisa mengikuti instruksi | Murah (1 GPU, beberapa jam) | **Anda** |
| **Preference (DPO/RLHF/GRPO)** | Diajari mana jawaban yang lebih baik | Sedang | **Anda** |

- **Base model** hanya melanjutkan teks. Ditanya "Apa ibu kota Indonesia?", ia bisa
  melanjutkan dengan soal-soal lain, seperti daftar soal ujian di internet.
- **Instruct/chat model** sudah melewati SFT dan preference, sehingga menjawab.

Detail data tiap tahap ada di [06 — Data Training](06-data-training.md).
