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

## Per huruf, per kata, atau potongan kata?

Mana yang lebih "mahal"? Jawabannya tergantung **apa yang diukur**, dan hasilnya berlawanan:

| Diukur dari | Lebih mahal | Kenapa |
|---|---|---|
| **Ukuran model** (per token) | **Per kata** | Kosakatanya sangat besar, jadi tabelnya besar |
| **Biaya memproses teks** (per kalimat) | **Per huruf** | Teks yang sama menjadi jauh lebih banyak token |

> Angka di bagian ini dihitung dari teks latih GPT mini (`mini_gpt/data/teks.txt`).
> Jalankan `python contoh/07_biaya_tokenizer.py` untuk melihatnya sendiri.

### Kenapa per kata membuat model lebih mahal

**1. Kosakatanya meledak.** Huruf jumlahnya tetap, tetapi kata baru selalu muncul
(nama, istilah, salah ketik):

| Tokenizer | Jumlah token berbeda (kosakata) |
|---|---|
| Per huruf | **49** |
| Per kata, hanya dari teks latih kecil ini | **426** |
| Per kata, untuk bahasa nyata | **±100.000 atau lebih** |

**2. Setiap token butuh dua baris di model.** Satu baris di **tabel embedding** (input,
lihat [02](02-vektor-dan-embedding.md)) dan satu baris di **kepala output**, karena di
akhir model harus memberi skor untuk **setiap** token di kosakata:

```
parameter embedding + kepala = 2 × ukuran_kosakata × n_embd
```

Dengan `n_embd = 64` seperti GPT mini:

| Tokenizer | Embedding + kepala |
|---|---|
| Per huruf (49) | **6.272** parameter |
| Per kata, teks ini (426) | **54.528** parameter |
| Per kata, bahasa nyata (100.000) | **12.800.000** parameter |

Seluruh GPT mini hanya **±110 ribu parameter**. Dengan kosakata 100.000 kata, dua tabel ini
saja sudah **100 kali lebih besar dari seluruh modelnya**. Setiap kali menebak token
berikutnya, model juga harus menghitung skor untuk 100.000 kemungkinan, bukan 49.

**3. Data untuk tiap kata terlalu sedikit.** Dari 426 kata unik, **259 hanya muncul sekali**.
Barisnya hampir tidak bisa belajar apa pun. Sebaliknya, huruf `a` muncul ratusan kali.

**4. Kata berimbuhan jadi token yang tidak saling kenal.** Masalah besar untuk bahasa Indonesia:

```
makan  → makan, makanan, memakan
hitung → berhitung, menghitung
lari   → lari, berlari, larinya, larimu
```

Dengan token per kata, `makan`, `makanan`, dan `memakan` adalah **baris yang terpisah sama
sekali**; model harus belajar dari nol bahwa ketiganya berhubungan.

**5. Kata baru tidak bisa diproses.** `dimakan`, `berlarian`, dan `Kancilnya` tidak ada di
data latih, sehingga **tidak punya nomor sama sekali** (*out-of-vocabulary*). Tokenizer per
huruf tidak punya masalah ini: kata baru tetap bisa dieja dengan huruf yang sudah dikenal.

### Kenapa per huruf membuat pemrosesan teks lebih mahal

Untuk teks latih yang sama:

| | Per huruf | Per kata |
|---|---|---|
| Jumlah token | **6.070** | **1.139** |

Per huruf **±5,3 kali lebih panjang**. Akibatnya:

1. **Lebih banyak langkah.** Model menulis satu token per langkah → ±5 kali lebih banyak langkah.
2. **Attention jauh lebih berat.** Setiap token membandingkan dirinya dengan **semua** token
   sebelumnya ([03](03-attention.md)), jadi biayanya naik sekitar kuadrat: **5,3² ≈ 28 kali**.
3. **Konteks cepat habis.** `panjang_konteks = 64` di GPT mini hanya **±12 kata** dengan token
   per huruf, sehingga model cepat "lupa" awal kalimat.
4. **Biaya API.** Layanan LLM menagih **per token**; tokenizer per huruf akan membuat tagihan
   untuk teks yang sama naik berkali-kali lipat.

### Jalan tengah: potongan kata (sub-word)

LLM sungguhan memakai **potongan kata** (dengan algoritma seperti BPE):

```
"makanan"  → ["makan", "an"]
"dimakan"  → ["di", "makan"]   ← kata baru tetap bisa diproses
```

(Potongan persisnya tergantung tokenizer masing-masing model.)

| | Per huruf | Per kata | **Sub-word** |
|---|---|---|---|
| Ukuran kosakata | sangat kecil | meledak | **terkendali (±100–250 ribu)** |
| Panjang teks | sangat panjang | pendek | **pendek** (kata umum = 1 token) |
| Kata baru | bisa dieja | tidak dikenal | **dipecah jadi potongan yang dikenal** |
| Imbuhan | — | tidak saling kenal | **potongan `makan` dipakai bersama** |

Catatan: kebanyakan tokenizer dilatih dengan lebih banyak teks bahasa Inggris, sehingga kalimat
bahasa Indonesia biasanya dipecah menjadi **lebih banyak token** daripada kalimat bahasa Inggris
yang sepanjang itu — dan biayanya di API pun bisa sedikit lebih mahal.

### Kenapa GPT mini memakai per huruf?

Untuk **belajar** dengan **data sangat kecil**, per huruf paling cocok: tabelnya kecil
(49 baris), setiap huruf terlatih karena muncul ratusan kali, tidak ada kata yang "tidak
dikenal", dan kodenya sangat sederhana ([`mini_gpt/tokenizer.py`](../mini_gpt/tokenizer.py)).
Kekurangannya terlihat di hasilnya: model cepat lupa konteks dan sering membuat kata yang tidak ada.

> **Ringkasan:** token **per kata** membuat **model** lebih mahal (kosakata besar, tabel besar,
> banyak kata jarang, kata baru tidak dikenal). Token **per huruf** membuat **pemrosesan teks**
> lebih mahal (±5 kali lebih panjang, attention ±28 kali lebih berat, konteks cepat habis).
> LLM sungguhan memakai **potongan kata** sebagai jalan tengah.

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
