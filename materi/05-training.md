# 05 — Bagaimana AI Belajar (Training)

## Awalnya model "bodoh"

Saat baru dibuat, semua bobot (termasuk matriks Q, K, V) berisi **angka acak**.
Ditanya "Ibu kota Indonesia adalah ...", tebakannya pun acak.

## Siklus belajar: Tebak → Cek → Hitung → Perbaiki

```
┌──────────────────────────────────────────────────┐
│ 1. TEBAK    : model memprediksi token berikutnya │
│ 2. CEK      : bandingkan dengan jawaban asli     │
│ 3. HITUNG   : seberapa salah? (loss)             │
│ 4. PERBAIKI : ubah bobot sedikit agar lebih benar│
└──────────────────────────────────────────────────┘
                    ↺ ulangi jutaan kali
```

## Loss: skor kesalahan

```
loss = −log(peluang jawaban benar)

peluang "Jakarta" = 0.001%  →  loss = 11.5   😱
peluang "Jakarta" = 50%     →  loss = 0.69   🙂
peluang "Jakarta" = 99%     →  loss = 0.01   😎
```

Tujuan training: **membuat loss sekecil mungkin.**

> Patokan berguna: untuk model yang menebak acak dari `V` token, loss awalnya sekitar
> `ln(V)`. Model mini kita punya 49 karakter unik → loss awal ±3.9.

## Gradient descent: turun gunung dalam kabut ⛰️

Anda di gunung berkabut dan ingin turun ke lembah (loss terendah). Anda tidak bisa
melihat lembahnya, tapi bisa merasakan **kemiringan tanah**:

1. Rasakan ke arah mana tanah menurun → **gradient**
2. Melangkah sedikit ke arah itu → **learning rate** = ukuran langkah
3. Ulangi

- Learning rate terlalu besar → melompat-lompat, tidak pernah sampai.
- Terlalu kecil → sangat lambat.

## Backpropagation: mencari siapa yang salah

Kalau tebakan salah, bobot mana yang perlu diubah? Backpropagation menelusuri kesalahan
**dari layer terakhir ke layer pertama**, dan memberi tahu tiap bobot seberapa besar ia
ikut menyebabkan kesalahan. Seperti pelatih bola yang memutar ulang video gol kebobolan
dan memberi koreksi ke kiper, bek, dan gelandang sesuai kesalahan masing-masing.

## Q, K, V menjadi pintar dengan sendirinya

Tidak ada yang memprogram "kata *bank* harus memperhatikan *ikan*". Backprop berulang
kali memberi tahu Q dan K: *"kalau kamu fokus ke kata ini, tebakannya lebih benar"*.
Kepintaran attention adalah **efek samping** dari satu tujuan sederhana: menebak kata
berikutnya dengan benar.

## Istilah penting

| Istilah | Arti sederhana |
|---|---|
| **Loss** | Skor kesalahan (makin kecil makin bagus) |
| **Gradient** | Arah dan besarnya perubahan bobot |
| **Learning rate** | Ukuran langkah |
| **Batch** | Sekelompok contoh yang diproses sekaligus |
| **Step** | Satu kali memperbarui bobot |
| **Epoch** | Satu kali membaca seluruh dataset |
| **Optimizer (AdamW)** | Pemandu langkah yang lebih pintar dari gradient descent biasa |
| **Overfitting** | Model menghafal data latihan; bagus di data latihan, jelek di data baru |
| **Validation loss** | Loss di data yang tidak dipakai training — alat untuk mendeteksi overfitting |

## Inti training: 4 baris

Semua training AI — dari model mini sampai model terbesar — intinya 4 baris ini:

```python
logits, loss = model(x, y)     # 1-3. tebak + hitung loss
optimizer.zero_grad()
loss.backward()                # 4a. backpropagation
optimizer.step()               # 4b. perbaiki bobot
```

Lihat [`mini_gpt/train.py`](../mini_gpt/train.py) untuk versi lengkapnya.

## Coba sendiri

```bash
python contoh/04_training_mini.py
```

Model 7 kata belajar bahwa setelah "Ibu kota Indonesia adalah" datang "Jakarta" —
lihat loss turun dan peluang "Jakarta" naik ke ±99%.
