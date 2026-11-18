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

## Bagaimana potongan kata ditentukan? (BPE)

Tidak ada ahli bahasa yang menentukan bahwa "makanan" dipotong menjadi `["makan", "an"]`, dan
tidak ada aturan imbuhan di dalam tokenizer. Pemotongan ditentukan oleh **statistik**: potongan
huruf mana yang **paling sering muncul berdampingan** di data. Algoritma yang paling umum
disebut **BPE (Byte Pair Encoding)**.

> Hasil di bagian ini nyata, dari BPE sederhana yang dilatih pada teks latih GPT mini.
> Jalankan `python contoh/08_bpe_sederhana.py` untuk melihatnya sendiri.

### Cara kerja BPE

Idenya mirip orang yang malas menulis: *pasangan huruf yang paling sering muncul, gabungkan jadi satu token.*

```
1. Mulai dari huruf satu per satu.
2. Hitung semua pasangan yang berdampingan di seluruh teks.
3. Gabungkan pasangan yang PALING SERING menjadi token baru.
4. Ulangi langkah 2–3 sebanyak N kali.
```

Tidak ada kamus, tidak ada aturan tata bahasa. Hanya **menghitung dan menggabung**.

10 penggabungan (merge) pertama pada teks latih GPT mini:

```
 1. a + n   → an    (254 kali)
 2. a + r   → ar    (95 kali)
 3. l + a   → la    (78 kali)
 4. n + g   → ng    (77 kali)
 5. e + r   → er    (77 kali)
 6. an + g  → ang   (70 kali)   ← token hasil gabungan bisa digabung lagi
 7. m + e   → me    (64 kali)
 8. a + k   → ak    (56 kali)
 9. a + t   → at    (50 kali)
10. d + i   → di    (50 kali)
```

`an` menang karena pasangan ini muncul di mana-mana dalam bahasa Indonesia: makan**an**,
k**an**cil, **an**jing, dan seterusnya. Algoritma tidak tahu bahwa "-an" adalah akhiran; ia hanya
tahu bahwa `a` dan `n` **sering bersebelahan**.

### Melihat potongan terbentuk sedikit demi sedikit

| Setelah | makanan | dimakan | memakan | kancilnya | menghitung |
|---|---|---|---|---|---|
| 10 merge | m · ak · an · an | di · m · ak · an | me · m · ak · an | k · an · c · i · l · n · y · a | me · ng · h · i · t · u · ng |
| 60 merge | m · ak · an · an | di · m · ak · an | mem · ak · an | kancil · n · ya | me · ng · h · i · tu · ng |
| 120 merge | m · akan · an | di · m · akan | mem · akan | kancil · nya | meng · h · **itu** · ng |
| 300 merge | **makan · an** ✅ | **di · makan** ✅ | mem · akan ❌ | **kancil · nya** ✅ | menghitung |

- `makanan → makan · an`, `dimakan → di · makan`, dan `kancilnya → kancil · nya` **kebetulan cocok**
  dengan imbuhan yang benar.
- `memakan → mem · akan` ❌ — secara bahasa seharusnya "me + makan". Tetapi `akan` sudah terbentuk
  lebih dulu (dari "makan" dan "memakan"), jadi statistik memotongnya di tempat yang "salah".
- `menghitung` sempat dipotong `meng · h · itu · ng`, karena kata "itu" sangat sering muncul.
  Jelas bukan pemotongan yang masuk akal secara bahasa.
- Setelah 300 merge, `menghitung` menjadi **satu token utuh** karena cukup sering muncul di data.

### Kenapa hasilnya sering cocok dengan imbuhan?

Karena **imbuhan memang sering muncul**. "-an", "di-", "me-", "ber-", dan "-nya" ditempelkan ke
ratusan kata berbeda, jadi statistik menemukannya. Itu **efek samping**, bukan aturan.

### Apakah potongan yang "salah" merusak model?

Hampir tidak:

1. **Model tidak butuh potongan yang benar secara bahasa, hanya yang konsisten.** "memakan" akan
   **selalu** dipotong `mem · akan`, dan embedding ([02](02-vektor-dan-embedding.md)) serta
   attention ([03](03-attention.md)) belajar sendiri artinya dari konteks.
2. **Kata yang sering muncul akhirnya menjadi satu token utuh**, seperti `menghitung`.
3. **Tidak ada kata yang tidak dikenal**: paling buruk, sebuah kata dipecah sampai ke huruf.

### Berapa banyak penggabungan?

Jumlah merge **dipilih oleh pembuat model** — sebuah *hyperparameter*, sama seperti `n_layer`
atau `n_embd`:

```
ukuran kosakata = jumlah huruf/byte awal + jumlah merge
```

| Merge sedikit | Merge banyak |
|---|---|
| Kosakata kecil, model kecil | Kosakata besar, tabel embedding besar |
| Teks menjadi banyak token | Teks menjadi sedikit token |
| Mendekati tokenizer per huruf | Mendekati tokenizer per kata |

Jadi jumlah merge adalah tombol pengatur "jalan tengah" di atas. LLM besar biasanya berhenti di
sekitar **100.000 – 250.000 token**.

### Hal penting lainnya

1. **Tokenizer dilatih sekali, sebelum modelnya.** Daftar merge disimpan, lalu dipakai untuk
   memotong semua teks dengan **urutan merge yang sama**. Setelah model dilatih, tokenizer tidak
   boleh diubah, karena setiap nomor token sudah terikat ke baris tabel embedding.
2. **Data latih tokenizer menentukan siapa yang "diuntungkan".** Jika tokenizer dilatih sebagian
   besar dengan teks bahasa Inggris, merge-nya berisi potongan bahasa Inggris, dan kata bahasa
   Indonesia dipecah lebih kecil (lebih banyak token, lebih mahal di API).
3. **Ada beberapa variasi algoritma, tetapi idenya sama: statistik.**

| Algoritma | Ide | Dipakai di (contoh) |
|---|---|---|
| **BPE** | Gabungkan pasangan paling sering | Keluarga GPT, Llama, Qwen |
| **Byte-level BPE** | BPE yang mulai dari byte, bukan huruf, sehingga emoji dan aksara apa pun bisa diproses | GPT dan banyak LLM modern |
| **WordPiece** | Mirip BPE, tetapi memilih gabungan yang paling menaikkan peluang data | BERT |
| **Unigram (SentencePiece)** | Mulai dari kosakata besar, lalu buang potongan yang paling tidak berguna | T5, beberapa model multibahasa |

> **Ringkasan:** pemotongan token **tidak memakai aturan bahasa**. BPE mulai dari huruf, lalu
> berulang kali **menggabungkan pasangan yang paling sering muncul** di data. "makanan" menjadi
> `makan · an` karena "makan" dan "an" sering muncul, bukan karena algoritma tahu "-an" adalah
> akhiran. Kadang hasilnya "salah" secara bahasa (`mem · akan`), tetapi tidak masalah selama
> pemotongannya **konsisten**: model belajar arti setiap potongan dari konteks.

## Kapan tokenizer dilatih? Sekali, sebelum pretraining

"Menghitung dan menggabung" **tidak** membuat model terus berubah. Proses itu hanya terjadi
**sekali, sebelum pretraining**, lalu hasilnya **dibekukan selamanya**.

```
0. Kumpulkan data teks
        │
1. LATIH TOKENIZER (BPE)            ← "menghitung dan menggabung" terjadi DI SINI
   - di CPU, hitungan menit/jam
   - tanpa GPU, tanpa loss, tanpa backpropagation
   - hasil: daftar merge + kosakata → disimpan ke file
   - SETELAH INI DIBEKUKAN 🔒
        │
2. PRETRAINING                      ← bobot model dilatih (GPU, berminggu-minggu)
   - tokenizer hanya DIPAKAI untuk mengubah teks → ID
   - yang berubah: tabel embedding, attention, feed-forward
        │
3. FINE-TUNING (SFT, DPO, LoRA)     ← tokenizer yang SAMA
        │
4. DIPAKAI (inference)              ← tokenizer yang SAMA
```

Ada **dua jenis "pelatihan" yang sangat berbeda**:

| | Melatih tokenizer (BPE) | Melatih model (pretraining) |
|---|---|---|
| Kapan | Sekali, **sebelum** pretraining | Setelah tokenizer jadi |
| Caranya | Menghitung pasangan yang sering muncul | Tebak → loss → backprop → perbaiki bobot ([05](05-training.md)) |
| Punya bobot/loss? | ❌ Tidak | ✅ Ya |
| Butuh GPU? | ❌ Tidak | ✅ Ya, banyak |
| Hasil | Daftar merge + kosakata | Bobot model (miliaran angka) |
| Berubah lagi setelahnya? | ❌ Dibekukan | ✅ Bisa di-fine-tune |

### Kenapa tokenizer harus dibekukan?

**Nomor token = nomor baris di tabel embedding** ([02](02-vektor-dan-embedding.md)). Selama
pretraining, baris 26 belajar menjadi "makna `makan`". Kalau tokenizer dilatih ulang dan nomor 26
tiba-tiba berarti `ber`, model membaca baris yang salah untuk setiap token — **model rusak total**,
seperti mengganti nomor semua loker tanpa memindahkan isinya.

Karena itu, model di Hugging Face selalu dibagikan **bersama file tokenizernya**
(`tokenizer.json`, atau `vocab` + `merges`). `AutoTokenizer.from_pretrained(...)` mengunduh
tokenizer **persis sama** dengan yang dipakai saat pretraining. Ini juga alasan hasil fine-tuning
disebut **model turunan**: ia **harus** memakai tokenizer base model-nya ([07](07-fine-tuning.md)).

### Hal yang sama di GPT mini

1. **Latih tokenizer** — di [`mini_gpt/train.py`](../mini_gpt/train.py), `CharTokenizer.dari_teks(teks)`
   mengumpulkan semua karakter unik. Instan, tanpa GPU.
2. **Dibekukan dan disimpan** — daftar karakternya disimpan **ke dalam checkpoint**
   (`"karakter": tok.karakter`).
3. **Pretraining** — loop training mengubah bobot; daftar karakternya tidak pernah berubah.
4. **Dipakai** — [`mini_gpt/generate.py`](../mini_gpt/generate.py) **tidak membuat tokenizer baru**,
   tetapi memuatnya dari checkpoint (`CharTokenizer(ckpt["karakter"])`) agar nomornya cocok
   dengan tabel embedding.

Konsekuensinya: melatih dengan `--data teks_saya.txt` menghasilkan kosakata berbeda, jadi itu
**model baru dari nol** dengan tokenizer baru. Checkpoint lama dan baru tidak bisa saling ditukar.

### Pengecualian: menambah token setelah pretraining

Kadang tim tetap **menambah** token baru, misalnya:
- **token khusus** seperti `<|im_start|>` untuk chat template saat fine-tuning;
- **token bahasa Indonesia** agar teks Indonesia lebih hemat token.

Caranya **menambah, bukan mengganti**: token lama tetap dengan nomor yang sama, token baru
mendapat nomor di ujung. Tabel embedding diperbesar dengan baris baru yang **masih acak**, jadi
model perlu dilatih lagi agar baris-baris itu bermakna. Untuk beberapa token khusus, fine-tuning
sudah cukup; untuk banyak token bahasa baru, biasanya perlu *continued pretraining* dengan banyak
data — dan itu mahal.

> **Ringkasan:** BPE adalah **langkah terpisah yang terjadi sekali, sebelum pretraining**, tanpa
> bobot, loss, atau backpropagation. Hasilnya **dibekukan**, lalu dipakai tanpa perubahan saat
> pretraining, fine-tuning, dan saat model dipakai, karena setiap nomor token sudah terikat ke
> satu baris tabel embedding. Yang berubah saat training adalah **bobot model**, bukan tokenizernya.

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

- **Sebelum tahap pertama** ada langkah persiapan: **melatih tokenizer**, yang dilakukan sekali
  lalu dibekukan (lihat [Kapan tokenizer dilatih?](#kapan-tokenizer-dilatih-sekali-sebelum-pretraining)).
- **Base model** hanya melanjutkan teks. Ditanya "Apa ibu kota Indonesia?", ia bisa
  melanjutkan dengan soal-soal lain, seperti daftar soal ujian di internet.
- **Instruct/chat model** sudah melewati SFT dan preference, sehingga menjawab.

Detail data tiap tahap ada di [06 — Data Training](06-data-training.md).
