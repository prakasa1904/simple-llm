# 06 — Bentuk Data untuk Melatih LLM

Bentuk data **berbeda di setiap tahap**:

```
Pretraining      → teks mentah biasa         (triliunan token)
Fine-tuning SFT  → contoh percakapan         (ribuan contoh)
Preference (DPO) → jawaban bagus vs jelek    (ribuan pasang)
```

## 1. Pretraining — teks biasa

```json
{"text": "Jakarta adalah ibu kota Indonesia. Kota ini terletak di pulau Jawa..."}
{"text": "Nasi goreng adalah makanan khas Indonesia..."}
{"text": "def tambah(a, b):\n    return a + b"}
```

Tidak ada label. **Teks itu sendiri adalah soal sekaligus kunci jawabannya** — caranya
dengan **menggeser 1 posisi**:

```
Token :  [Jakarta] [adalah] [ibu]  [kota] [Indonesia] [.]
Input :  [Jakarta] [adalah] [ibu]  [kota] [Indonesia]
Target:  [adalah]  [ibu]    [kota] [Indonesia] [.]
```

Satu dokumen 1.000 token = 1.000 soal latihan gratis.

➡️ Model mini kita dilatih persis seperti ini, pada [`mini_gpt/data/teks.txt`](../mini_gpt/data/teks.txt).
Lihat fungsi `ambil_batch()` di [`mini_gpt/train.py`](../mini_gpt/train.py): `y` adalah `x` yang digeser 1.

Hasil pretraining: model pandai **melanjutkan teks**, tapi belum bisa diajak ngobrol.

### Data apa yang paling baik untuk pretraining?

Fakta, berita, ilmu pengetahuan, novel, atau percakapan sehari-hari? Jawabannya: **bukan salah
satunya, tetapi campurannya.** Setiap jenis data mengajarkan kemampuan yang berbeda. Campuran ini
disebut **data mix**, dan menentukannya adalah salah satu "resep rahasia" terpenting pembuat model.

| Jenis data | Mengajarkan | Kelemahan kalau terlalu banyak |
|---|---|---|
| **Ensiklopedia/fakta** (Wikipedia) | Pengetahuan dunia, bahasa rapi | Gaya kaku; jumlahnya relatif kecil |
| **Berita** | Peristiwa, bahasa formal, nama orang dan tempat | Cepat basi; bisa bias sumbernya |
| **Ilmu pengetahuan** (makalah, buku pelajaran) | Penalaran, penjelasan bertahap, istilah teknis | Sulit dan bahasanya padat |
| **Novel/cerita** | Teks panjang yang runtut, alur, emosi, dialog, kreativitas | Isinya fiksi, bukan fakta |
| **Percakapan sehari-hari** (forum, obrolan) | Bahasa santai, tanya-jawab, bahasa gaul | Pendek, berantakan, banyak salah ketik dan informasi keliru |
| **Kode program** | Logika, struktur, langkah demi langkah | Bukan bahasa manusia |
| **Matematika** | Penalaran bertahap, ketelitian | Sempit |
| **Web umum** (hasil crawl internet) | Keragaman luar biasa, jumlahnya sangat besar | Banyak sampah, iklan, duplikat, dan spam |

Model yang hanya membaca satu jenis data akan "berkepribadian" sesuai data itu:
- hanya berita → tahu banyak peristiwa, tetapi kaku dan pengetahuannya berhenti di tanggal tertentu;
- hanya novel → pandai bercerita, tetapi suka mengarang fakta;
- hanya percakapan → santai, tetapi dangkal dan tidak bisa menjelaskan hal rumit.

#### Contoh nyata: campuran data LLaMA (Meta, 2023)

Makalah LLaMA generasi pertama mempublikasikan campurannya:

| Sumber | Porsi |
|---|---|
| Web umum (CommonCrawl, sudah difilter) | ±67% |
| Web yang dibersihkan lebih ketat (C4) | ±15% |
| Kode (GitHub) | ±4,5% |
| Wikipedia | ±4,5% |
| Buku | ±4,5% |
| Makalah ilmiah (ArXiv) | ±2,5% |
| Tanya-jawab (StackExchange) | ±2% |

**Mayoritas adalah web umum**, karena hanya web yang jumlahnya cukup besar (triliunan token).
Sumber berkualitas tinggi seperti buku, Wikipedia, dan makalah jumlahnya terbatas, jadi porsinya
kecil — dan sering diulang 2–3 kali agar pengaruhnya lebih besar.

#### Lebih penting dari jenisnya: kualitas

**Cara menyaring data sering lebih berpengaruh daripada jenis datanya.**

1. **Filter kualitas** — buang spam, iklan, teks rusak, halaman yang isinya hanya menu navigasi,
   dan konten berbahaya.
2. **Hapus duplikat (deduplication)** — internet penuh salinan teks yang sama. Kalau tidak dihapus,
   model cenderung **menghafal** (lihat overfitting di [08](08-membangun-mini-gpt.md)).
3. **Utamakan teks "bernilai edukasi"** — contohnya dataset **FineWeb-Edu** dari Hugging Face:
   teks web dinilai oleh model pengklasifikasi ("seberapa mendidik teks ini?") dan hanya yang
   bernilai tinggi disimpan. Model yang dilatih dengan data hasil saringan ini jauh lebih baik di
   tes pengetahuan dan penalaran, padahal datanya lebih sedikit.
4. **Data sintetis** — beberapa model kecil (misalnya keluarga **Phi** dari Microsoft) dilatih
   dengan banyak teks "bergaya buku pelajaran" yang **ditulis oleh LLM lain**, dan menghasilkan
   kemampuan penalaran yang mengejutkan untuk ukurannya.

Temuan menarik: **kode** ternyata membantu kemampuan penalaran secara umum, bahkan untuk tugas
yang bukan pemrograman. Karena itu hampir semua LLM modern memasukkan banyak kode.

#### Urutan juga berpengaruh

```
Fase awal  : data besar dan beragam (web umum dominan)
             → model belajar bahasa dan pengetahuan luas
Fase akhir : porsi data berkualitas tinggi dinaikkan
             (buku pelajaran, matematika, kode, sains)
             → "memoles" kemampuan penalaran
```

Fase akhir ini sering disebut **annealing**, karena learning rate juga diturunkan pelan-pelan.

#### Lalu, percakapan sehari-hari masuk ke mana?

Percakapan memang ada di pretraining (dari forum dan media sosial), tetapi porsinya tidak besar.
Kemampuan **mengobrol dengan baik** justru dibentuk di tahap **SFT** (bagian 2 di bawah) dengan
data percakapan yang dikurasi:

- **Pretraining** → pengetahuan dan kemampuan bahasa (campuran semua jenis data).
- **SFT** → cara menjawab sebagai asisten (percakapan berkualitas).

#### Untuk bahasa Indonesia

- **Datanya jauh lebih sedikit** daripada bahasa Inggris, jadi setiap sumber berkualitas sangat
  berharga: Wikipedia Indonesia, berita, buku domain publik, dokumen pemerintah, skripsi dan jurnal
  terbuka.
- **Campur dengan bahasa Inggris tetap membantu.** Pengetahuan dari teks Inggris sebagian bisa
  "pindah" ke bahasa Indonesia, sehingga model multibahasa sering lebih baik daripada model yang
  hanya dilatih dengan sedikit data bahasa Indonesia.
- **Perhatikan hak cipta dan privasi.** Novel dan berita modern biasanya berhak cipta, dan data
  percakapan bisa berisi informasi pribadi. Banyak tim kini memakai data berlisensi terbuka atau
  data yang izinnya jelas.

#### Lihat langsung di GPT mini

[`mini_gpt/data/teks.txt`](../mini_gpt/data/teks.txt) sebenarnya juga sebuah **data mix** kecil:
cerita Kancil (novel), kota dan makanan (fakta), hewan (ilmu pengetahuan), kegiatan sehari-hari,
dan percakapan. Karena itu tulisan model mencampur semuanya.

Eksperimen: buat dua file, satu **hanya percakapan** dan satu **hanya fakta**, lalu latih masing-masing:

```bash
python mini_gpt/train.py --data percakapan.txt --keluaran mini_gpt/model_percakapan.pt
python mini_gpt/train.py --data fakta.txt --keluaran mini_gpt/model_fakta.pt
```

Tulisan kedua model akan punya "gaya" yang sangat berbeda — bukti langsung bahwa **data
menentukan karakter model**.

> **Ringkasan:** tidak ada satu jenis data terbaik. Pretraining memakai **campuran**: web umum
> dalam jumlah besar untuk keragaman, ditambah buku, ensiklopedia, sains, kode, dan matematika
> untuk kualitas dan penalaran. Yang paling menentukan adalah **kualitas data**: penyaringan,
> penghapusan duplikat, dan mengutamakan teks bernilai edukasi. Kemampuan bercakap terutama
> dibentuk di tahap **SFT**, bukan pretraining.

## 2. Fine-tuning (SFT) — contoh percakapan

```json
{"messages": [
  {"role": "system", "content": "Kamu asisten customer service yang ramah."},
  {"role": "user", "content": "Pesanan saya belum sampai."},
  {"role": "assistant", "content": "Mohon maaf, Kak. Boleh minta nomor pesanannya?"}
]}
```

| Role | Isi |
|---|---|
| `system` | Instruksi umum (kepribadian, aturan) |
| `user` | Pertanyaan |
| `assistant` | Jawaban yang **ingin ditiru** model |

Di dalam, JSON ini diubah menjadi satu teks dengan **chat template**:

```
<|im_start|>user
Pesanan saya belum sampai.<|im_end|>
<|im_start|>assistant
Mohon maaf, Kak. Boleh minta nomor pesanannya?<|im_end|>
```

(Penandanya beda-beda per keluarga model; contoh di atas gaya Qwen.)

**Loss masking**: hanya bagian jawaban `assistant` yang dinilai. Model tidak perlu
belajar menulis pertanyaan user.

Contoh file: [`contoh/data_sft_contoh.jsonl`](../contoh/data_sft_contoh.jsonl).

## 3. Preference (DPO) — mana yang lebih baik

```json
{
  "prompt": "Jelaskan apa itu inflasi.",
  "chosen": "Inflasi adalah kenaikan harga barang secara umum. Contoh: nasi goreng tahun lalu Rp15.000, sekarang Rp18.000.",
  "rejected": "Inflasi adalah fenomena makroekonomi yang mengindikasikan depresiasi daya beli akibat ekspansi agregat moneter."
}
```

Model belajar: buat jawaban seperti `chosen` lebih mungkin, seperti `rejected` lebih jarang.

## Perbandingan

| | Pretraining | SFT | Preference |
|---|---|---|---|
| Format | `{"text"}` | `{"messages"}` | `{"prompt","chosen","rejected"}` |
| Jumlah | Triliunan token | 1.000 – 100.000 contoh | 1.000 – 50.000 pasang |
| Mengajarkan | Pengetahuan & bahasa | Cara menjawab | Selera & kualitas |
| Analogi | Membaca perpustakaan 📚 | Magang dengan mentor 👨‍🏫 | Diberi feedback 👍👎 |

## Tips data SFT yang bagus

1. **Kualitas > kuantitas** — 1.000 contoh bersih > 100.000 contoh berantakan.
2. **Konsisten** — gaya, panjang, dan format jawaban seragam.
3. **Beragam** — variasikan cara bertanya.
4. **Sisihkan ±10% sebagai data uji** — jangan ikut dilatih.

## Lihat dataset sungguhan

Di **huggingface.co/datasets**, cari: `fineweb` (pretraining), `ultrachat` (SFT),
`ultrafeedback` (preference), `indonesian`.
