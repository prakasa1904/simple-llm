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

## Coba sendiri

```bash
python contoh/01_vektor_buah.py
```

Skrip ini menghitung cosine similarity antar buah memakai tabel di atas, tanpa download apa pun.
