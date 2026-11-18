# 03 — Attention

## Masalahnya

> "Ular itu mengeluarkan **bisa**." 🐍 (racun)
> "Adik saya sudah **bisa**." 🙂 (mampu)

Kata "bisa" punya vektor awal yang **sama** di kedua kalimat, padahal maknanya beda.
Model butuh cara agar **kata-kata di sekitarnya bisa mengubah makna sebuah kata**.
Itulah tugas attention.

## Analogi: ruang kelas

Setiap kata adalah murid yang memegang 3 kartu (masing-masing berupa vektor, dibuat
dengan mengalikan vektor kata dengan matriks bobot yang dipelajari):

| Kartu | Arti | Contoh |
|---|---|---|
| **Query (Q)** | "Apa yang saya cari?" | "bisa": *"Saya butuh konteks: racun atau mampu?"* |
| **Key (K)** | "Apa yang saya tawarkan?" | "Ular" mengangkat kartu: *"Saya hewan berbisa"* |
| **Value (V)** | "Ini informasi saya" | "Ular" memberikan maknanya |

## Langkah demi langkah

Kalimat: **"Ular itu mengeluarkan bisa"**. Kita perbarui kata **"bisa"**. Ia hanya
boleh melihat dirinya sendiri dan kata-kata **sebelumnya**.

**1. Bandingkan Query "bisa" dengan Key setiap kata** (cek kemiripan vektor)
```
skor:  Ular 3.5   itu 0.1   mengeluarkan 1.5   bisa 1.0
```

**2. Ubah skor jadi persentase (softmax)**
```
Ular 70%   itu 2%   mengeluarkan 18%   bisa 10%
```

**3. Campur Value sesuai persentase**
```
vektor baru "bisa" = 70%·V(Ular) + 2%·V(itu) + 18%·V(mengeluarkan) + 10%·V(bisa)
```

Sekarang vektor "bisa" mengandung banyak informasi "Ular" → maknanya condong ke **racun**.
Di kalimat "Adik saya sudah bisa", kata "sudah" yang akan mendapat skor tinggi →
maknanya condong ke **mampu**.

## Rumusnya (sekarang pasti masuk akal)

```
Attention(Q, K, V) = softmax( Q · Kᵀ / √d ) · V
                      └─ langkah 1 ─┘ └ lgkh 2 ┘  └ lgkh 3 ┘
```

`√d` hanya menjaga angka agar tidak terlalu besar.

## Tiga detail penting

1. **Multi-head**: attention dilakukan beberapa kali secara paralel ("head").
   Tiap head bisa belajar fokus berbeda: tata bahasa, rujukan kata ganti, topik, dll.
2. **Banyak layer**: attention + feed-forward = 1 layer. Model menumpuk puluhan layer.
   Layer awal menangkap pola sederhana, layer akhir menangkap makna yang lebih dalam.
3. **Tidak boleh mengintip ke depan (causal mask)**: sebuah token hanya boleh melihat token
   **sebelumnya**. Masuk akal — saat menulis kata berikutnya, masa depan belum ada.

## Token terakhir = ringkasan seluruh kalimat

Untuk "Apa kabar?", setelah melewati semua layer, vektor token terakhir `?` sudah
menyerap makna seluruh kalimat: *"seseorang menyapa dan menanyakan kabar saya"*.
Vektor inilah yang dipakai untuk menebak kata berikutnya.

## Di mana ini ada di kode?

Kelas `CausalSelfAttention` di [`mini_gpt/model.py`](../mini_gpt/model.py) menulis
ketiga langkah di atas secara eksplisit, lengkap dengan causal mask.

## Coba sendiri

```bash
python contoh/02_attention_manual.py
```

Skrip ini menjalankan ketiga langkah di atas dengan angka kecil, lalu mencetak tabel
persentase perhatian tiap kata.
