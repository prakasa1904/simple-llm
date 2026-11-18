# 09 — Data yang Cepat Berubah: Fine-tuning, RAG, atau Tool Calling?

> **Pertanyaan:** Kalau data untuk fine-tuning berubah terlalu cepat, apakah harus
> fine-tuning ulang setiap ada data baru?
>
> **Jawaban singkat:** Tidak. Data yang sering berubah sebaiknya **dibaca model saat
> menjawab**, bukan **ditanamkan ke bobotnya**.

## Prinsip utama

> **Fine-tuning mengajarkan perilaku, bukan fakta.**
> (lihat juga [07 — Fine-tuning](07-fine-tuning.md))

## Kenapa fine-tuning tidak cocok untuk data yang cepat berubah

| Masalah | Penjelasan |
|---|---|
| **Selalu ketinggalan** | Data berubah jam 10, training selesai jam 14. Selama 4 jam itu model memberi jawaban lama |
| **Fakta lama tidak benar-benar hilang** | Harga lama Rp50.000 dan harga baru Rp45.000 bisa sama-sama "teringat", lalu model bingung atau mencampurnya |
| **Bisa mengarang** | Model hafal *pola* jawaban, bukan isinya. Ia bisa menyebut harga yang terdengar meyakinkan tapi salah |
| **Mahal dan berisiko** | Tiap training butuh GPU, evaluasi ulang, dan bisa merusak kemampuan lain (*catastrophic forgetting*) |
| **Tidak bisa dilacak** | Tidak jelas jawaban model berasal dari data versi mana |

## Solusinya: pisahkan "cara menjawab" dan "isi jawaban"

```
┌──────────────────────────────┐     ┌────────────────────────────────┐
│  PERILAKU (jarang berubah)   │     │  FAKTA (sering berubah)        │
│  gaya bahasa, format,        │     │  harga, stok, jadwal, promo,   │
│  aturan, cara berpikir       │     │  dokumen, status pesanan       │
│                              │     │                                │
│  → FINE-TUNING               │     │  → RAG / TOOL CALLING          │
│    (sesekali)                │     │    (dibaca saat menjawab)      │
└──────────────────────────────┘     └────────────────────────────────┘
```

### Cara 1 — RAG (Retrieval-Augmented Generation)

Model **mencari dokumen yang relevan dulu**, lalu menjawab berdasarkan dokumen itu.
Pencariannya memakai vektor dan cosine similarity dari [02 — Vektor dan Embedding](02-vektor-dan-embedding.md).

```
Pertanyaan: "Berapa harga sepatu X?"
     ↓ ubah jadi vektor
Cari dokumen terdekat di database vektor
     ↓
Dokumen: "Sepatu X — Rp450.000 (diperbarui hari ini)"
     ↓ tempel ke prompt
Model: "Harga sepatu X saat ini Rp450.000, Kak."
```

Data baru? **Cukup perbarui dokumennya.** Tidak ada training, langsung berlaku saat itu juga.

Cocok untuk: dokumen perusahaan, FAQ, katalog produk, kebijakan, artikel.

### Cara 2 — Tool calling (memanggil database/API)

Untuk data yang **sangat dinamis** atau berbeda per pengguna, model memanggil fungsi
untuk mengambil data **langsung dari sumbernya**:

```
User  : "Pesanan saya #1234 sudah sampai mana?"
Model : panggil cek_pesanan("1234")
Sistem: {"status": "dikirim", "kurir": "JNE", "estimasi": "besok"}
Model : "Pesanan #1234 sedang dikirim lewat JNE, estimasi tiba besok, Kak."
```

Cocok untuk: stok, status pesanan, saldo, kurs, cuaca, jadwal — apa pun yang berubah tiap menit.

## Contoh nyata: chatbot toko online

| Kebutuhan | Seberapa cepat berubah | Solusi |
|---|---|---|
| Gaya bicara ramah, panggil "Kak" | Hampir tidak pernah | **Fine-tuning** (atau cukup system prompt) |
| Format jawaban singkat + emoji | Jarang | **Fine-tuning** |
| Kebijakan retur, FAQ | Bulanan | **RAG** |
| Katalog dan deskripsi produk | Mingguan | **RAG** |
| Harga dan promo | Harian | **RAG** atau **tool** |
| Stok barang | Per menit | **Tool** (query database) |
| Status pesanan pelanggan | Real-time | **Tool** (API pesanan) |

Kombinasi yang paling umum di produk nyata:

> **Fine-tuning untuk perilaku + RAG/tool untuk fakta.**

Fine-tuning bahkan bisa dipakai untuk melatih model **agar pandai memakai** hasil RAG,
misalnya: "jawab hanya dari dokumen yang diberikan, dan bilang tidak tahu kalau tidak ada".

## Kalau memang harus fine-tuning ulang

Ada kasus yang tetap butuh fine-tuning berkala, misalnya istilah atau gaya baru di domain Anda.
Praktik yang baik:

1. **Jadwalkan, jangan setiap ada data baru.** Kumpulkan dulu, lalu latih **mingguan atau bulanan**.
2. **Selalu latih ulang dari base model**, dengan **data lama + data baru** (buang data lama
   yang sudah tidak berlaku). Jangan menumpuk fine-tune di atas hasil fine-tune terus-menerus —
   kesalahan ikut menumpuk dan model makin "lupa".
3. **Pakai LoRA** — training murah dan cepat, hasilnya hanya adapter kecil yang mudah diganti.
4. **Beri versi pada adapter** (`adapter-v1`, `adapter-v2`, ...) supaya bisa rollback.
5. **Evaluasi otomatis sebelum dipakai** — siapkan set pertanyaan uji yang tetap, bandingkan
   versi lama dan baru.

```
Data baru terkumpul → (tiap minggu/bulan) → training LoRA dari base
     → evaluasi otomatis → lulus? → deploy adapter-vN
                         → gagal? → tetap pakai adapter-v(N-1)
```

## Panduan memilih

```
Apakah yang ingin diubah adalah FAKTA/DATA?
 ├─ Ya → apakah berubah sering atau per pengguna?
 │        ├─ Ya, real-time / per pengguna → TOOL CALLING
 │        └─ Berupa dokumen / teks        → RAG
 └─ Tidak, yang ingin diubah adalah PERILAKU (gaya, format, keahlian)
          ├─ Bisa dijelaskan di prompt?   → SYSTEM PROMPT dulu (paling murah)
          └─ Tidak cukup                  → FINE-TUNING (sesekali)
```

## Ringkasan

> Jangan fine-tuning setiap ada data baru. **Data yang sering berubah diambil saat menjawab**
> lewat RAG (dokumen) atau tool calling (database/API), sehingga langsung up to date tanpa
> training. **Fine-tuning dipakai untuk perilaku yang jarang berubah**, dan kalau perlu diulang,
> lakukan terjadwal: dari base model, dengan gabungan data lama dan baru, lalu dievaluasi
> dulu sebelum dipakai.
