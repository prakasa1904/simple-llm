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
