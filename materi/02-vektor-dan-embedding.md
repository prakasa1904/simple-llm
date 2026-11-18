# 02 — Vektor dan Embedding

## Apa itu vektor?

Vektor hanyalah **deretan angka yang menggambarkan sesuatu**.

Contoh: gambarkan buah dengan 3 angka **[manis, ukuran, merah]**

| Buah | Vektor |
|---|---|
| 🍎 Apel | [7, 5, 8] |
| 🍒 Ceri | [8, 1, 9] |
| 🍉 Semangka | [6, 10, 3] |
| 🍋 Lemon | [1, 4, 0] |

Vektor adalah **alamat** sebuah benda di "ruang makna". Benda yang mirip letaknya **berdekatan**.

## Embedding = vektor untuk kata

AI melakukan hal yang sama untuk kata, tetapi:
- tiap kata punya **ratusan sampai ribuan angka**, bukan 3;
- angka-angka itu **dipelajari sendiri** oleh model saat training (lihat [05](05-training.md)).

Hasilnya: "kucing" dekat dengan "anjing" (sama-sama hewan peliharaan), tetapi jauh
dari "kunci", walaupun ejaannya mirip. Yang menentukan jarak adalah **makna**, bukan ejaan.

---

## Hubungan tokenizer dan embedding

Tokenizer dan embedding adalah **dua langkah berbeda** yang bekerja berurutan:

```
Teks  ──(1) TOKENIZER──►  Nomor ID  ──(2) EMBEDDING──►  Vektor (deretan angka)
"a"                         26                          [0.03, -0.15, 0.02, 0.58, ...]
```

| | Tokenizer | Embedding |
|---|---|---|
| **Tugas** | Memotong teks dan memberi **nomor** | Mengubah nomor menjadi **vektor bermakna** |
| **Bentuk** | Kamus: token ↔ nomor | **Tabel angka** (matriks) |
| **Dipelajari?** | ❌ Tidak, dibuat sekali di awal | ✅ Ya, angkanya berubah saat training |
| **Di kode** | [`mini_gpt/tokenizer.py`](../mini_gpt/tokenizer.py) | `self.embed_token` di [`mini_gpt/model.py`](../mini_gpt/model.py) |

> Angka-angka di bagian ini diambil dari GPT mini di repo ini setelah dilatih dengan
> `python mini_gpt/train.py`. Jalankan `python contoh/06_tokenizer_embedding.py` untuk
> melihatnya sendiri (hasil Anda bisa sedikit berbeda).

### Langkah 1 — Tokenizer memberi nomor

Tokenizer GPT mini mengenal **49 karakter** dari data latihnya:

```
\n ␣ " , - . ? A B D E G H I J K L M N O P R S T U Y a b c d e f g h i j k l m n o p r s t u v w y
```

Setiap karakter diberi nomor sesuai urutannya:

```
"Kancil" → ['K', 'a', 'n', 'c', 'i', 'l'] → [15, 26, 39, 28, 34, 37]
```

**Penting: nomor ini tidak punya makna.** Ia seperti **nomor absen murid**: murid nomor 15
dan 16 tidak lebih mirip daripada murid nomor 15 dan 40, karena nomornya hanya urutan alfabet.

Karena itu model **tidak bisa langsung memakai nomor ID**. Kalau dipakai langsung, model akan
mengira `K` (15) "dekat" dengan `L` (16) hanya karena nomornya berdekatan.

### Langkah 2 — Embedding adalah tabel, ID adalah nomor barisnya

Embedding GPT mini adalah **tabel berukuran 49 × 64**:
- **49 baris** — satu baris per token di kosakata;
- **64 kolom** — panjang vektornya (`n_embd = 64`).

```
              kolom 1  kolom 2  kolom 3  kolom 4  ...  kolom 64
baris  0 (\n) [ ...                                          ]
baris  1 (␣)  [ ...                                          ]
...
baris 15 (K)  [ ...                                          ]
...
baris 26 (a)  [ 0.03    -0.15    0.02     0.58    ...        ]   ← vektor untuk 'a'
...
baris 48 (y)  [ ...                                          ]
```

Cara kerjanya hanya **mengambil baris** — tidak ada perhitungan rumit:

```
ID 26  →  ambil baris ke-26  →  [0.03, -0.15, 0.02, 0.58, -0.81, 0.88, 0.77, -0.85, ...]
```

Analogi: tokenizer memberi **nomor loker**, embedding adalah **lemari loker** tempat isinya diambil.

Untuk "Kancil":

```
Tokenizer:   K    a    n    c    i    l
ID       :  15   26   39   28   34   37
             ↓    ↓    ↓    ↓    ↓    ↓   ambil baris dari tabel 49×64
Embedding: [64] [64] [64] [64] [64] [64]   ← 6 vektor, masing-masing 64 angka
```

Hasilnya matriks **6 × 64** — inilah yang masuk ke attention ([03](03-attention.md)).

### Isi tabel dipelajari saat training

Saat model baru dibuat, **isi tabel ini acak**. Setiap langkah training, backpropagation
([05](05-training.md)) juga **mengubah sedikit baris-baris tabel ini** agar tebakan makin benar.

Lama-lama, token yang **berperan mirip** mendapat vektor yang **mirip**. Ini terlihat di
GPT mini setelah ±12 detik training:

| Token | Paling dekat dengan | Kenapa masuk akal |
|---|---|---|
| `.` (titik) | `,` (koma), `"` (kutip), `\n` (baris baru) | Semuanya **tanda baca / akhir kalimat** |
| `a` | `A` | Huruf yang sama, besar dan kecil |
| `K` | `J` | Sama-sama **huruf kapital di awal nama** (**K**ancil, **J**akarta) |

Tidak ada yang memprogram "titik mirip dengan koma". Model **menemukannya sendiri** karena
keduanya muncul di posisi yang mirip: setelah kata dan sebelum spasi.

Sinyalnya masih lemah karena modelnya sangat kecil dan datanya hanya 6 ribu karakter;
sebagian pasangan "terdekat" lain masih acak. Di model besar yang dilatih dengan triliunan
token, pola seperti ini menjadi sangat kuat.

### Ditambah embedding posisi

Token yang sama **selalu** mengambil baris yang sama. Di "Jakarta", huruf `a` muncul tiga
kali, dan ketiganya mendapat vektor yang **persis sama** dari baris 26.

Artinya tabel token tidak tahu **urutan**: tanpa informasi posisi, "Kancil" dan "licnaK" berisi
vektor yang sama, hanya urutannya berbeda. Karena itu ada **tabel kedua** untuk posisi
(`embed_posisi`, ukuran 64 × 64, juga dipelajari):

```python
# mini_gpt/model.py
x = self.embed_token(idx) + self.embed_posisi(posisi)
#   ↑ "token apa"              ↑ "di urutan ke berapa"
```

```
vektor akhir 'a' di posisi 2 = baris 26 tabel token + baris 2 tabel posisi
```

### Alur lengkap

```
"Kancil"
   │  tokenizer  (kamus, tidak dilatih)
   ▼
[15, 26, 39, 28, 34, 37]                 ← 6 nomor
   │  embed_token  (tabel 49×64, DILATIH)
   │  + embed_posisi (tabel 64×64, DILATIH)
   ▼
matriks 6 × 64                           ← 6 vektor bermakna
   │  attention + feed-forward (materi 03)
   ▼
matriks 6 × 64 (sudah paham konteks)
   │  kepala Linear → 49 skor
   ▼
peluang token berikutnya → pilih satu → tokenizer.decode → teks
```

Tokenizer dipakai **di awal** (teks → ID) dan **di akhir** (ID → teks). Embedding hanya
dipakai **di awal**, untuk mengubah ID menjadi vektor.

### Pada model besar: mekanismenya sama, ukurannya berbeda

| | GPT mini | LLM besar |
|---|---|---|
| Jenis token | 1 karakter | potongan kata, misalnya `"makan"`, `"an"` |
| Jumlah token di kosakata | 49 | ±100.000 – 250.000 |
| Panjang vektor | 64 | ±2.000 – 12.000 |
| Ukuran tabel embedding | 49 × 64 ≈ 3 ribu angka | 100.000 × 4.096 ≈ **400 juta angka** |

Karena tokenizer model besar memakai potongan kata, `"makanan"` bisa menjadi
`["makan", "an"]` — **2 baris** dari tabel. Model belajar bahwa baris `"makan"` berhubungan
dengan makanan, dan `"an"` sering membentuk kata benda.

---

## Hati-hati: "embedding" punya dua arti

| | Embedding **token** (di dalam LLM) | Embedding **kalimat** (untuk RAG/pencarian) |
|---|---|---|
| Satuan | Per token | **Seluruh** kalimat atau dokumen |
| Hasil untuk "Apa kabar?" | Beberapa vektor, satu per token | **Satu** vektor saja |
| Cara membuat | Ambil baris dari tabel | Teks melewati seluruh model, lalu semua vektor tokennya diringkas jadi satu |
| Dipakai untuk | Input awal LLM | Mencari dokumen yang mirip ([RAG, materi 09](09-data-cepat-berubah.md)) |
| Contoh | `embed_token` di `mini_gpt` | model `sentence-transformers` |

Bagian di bawah ini (kemiripan dan cosine similarity) berlaku untuk **keduanya**.

---

## Matematika pada vektor = matematika pada makna

```
Raja − Pria + Wanita ≈ Ratu
```

Arah dari "pria" ke "wanita" kira-kira sama dengan arah dari "raja" ke "ratu".
Jadi **arah** di ruang vektor bisa mewakili sebuah konsep.

## Mengukur kemiripan: cosine similarity

Dua vektor dianggap mirip jika **menunjuk ke arah yang sama**:

| Nilai | Arti |
|---|---|
| **1.0** | arah sama → sangat mirip |
| **0** | tegak lurus → tidak berhubungan |
| **−1** | berlawanan |

Operasi sederhana ini dipakai di mana-mana:

| Aplikasi | Cara memakai vektor |
|---|---|
| Pencarian / RAG | Ubah pertanyaan jadi vektor, cari dokumen dengan vektor terdekat |
| Rekomendasi | "Kamu suka lagu ini" → cari lagu dengan vektor berdekatan |
| Pencarian gambar | Teks dan gambar berada di ruang yang sama |
| Deteksi spam | Email spam berkumpul di satu wilayah ruang vektor |

## Jaringan saraf = mesin pengubah vektor

```
vektor input → [kalikan dengan bobot + tambah] → vektor baru → ... → jawaban
```

Setiap layer mengalikan vektor dengan **matriks bobot** (weights).
**Training** = menyesuaikan bobot itu sampai vektor keluaran memberi jawaban yang benar.

Itulah sebabnya **aljabar linear** (matematika vektor dan matriks) adalah matematika
terpenting untuk AI.

## Ringkasan

> **Tokenizer** memberi setiap potongan teks sebuah **nomor** (tanpa makna), lalu
> **embedding** memakai nomor itu sebagai **nomor baris** untuk mengambil **vektor** dari
> sebuah tabel. Isi tabel itulah yang dipelajari saat training, sehingga token yang berperan
> mirip mendapat vektor yang mirip.

## Coba sendiri

```bash
python contoh/01_vektor_buah.py          # cosine similarity antar buah, tanpa download
python contoh/06_tokenizer_embedding.py  # tokenizer → tabel embedding GPT mini Anda
```

`06_tokenizer_embedding.py` membutuhkan model yang sudah dilatih (`python mini_gpt/train.py`).
