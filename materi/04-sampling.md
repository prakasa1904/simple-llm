# 04 — Sampling: Memilih Kata Berikutnya

Setelah semua layer, model memberi **peluang untuk setiap token** di kosakatanya:

```
"Apa kabar?" → kata berikutnya:
   "Saya"    40%
   "Baik"    25%
   "Kabar"   12%
   "Luar"    10%
   "Pisang"   0.01%
   ...
```

Bagaimana memilih satu?

## Strategi 1 — Greedy (selalu ambil yang teratas)

- ✅ Bisa ditebak
- ❌ Membosankan, jawaban selalu sama, bisa terjebak mengulang ("Saya baik. Saya baik. Saya baik...")

## Strategi 2 — Sampling (lempar dadu berbobot)

Pilih secara acak dengan peluang sesuai persentase. "Saya" menang 40% dari waktu.
Itulah sebabnya **pertanyaan yang sama bisa mendapat jawaban berbeda**.

## Kontrol 1 — Temperature 🌡️

Mempertajam atau meratakan peluang **sebelum** melempar dadu:

```
              rendah (0.2)   normal (1.0)   tinggi (1.5)
"Saya"           85%            40%            28%
"Baik"           12%            25%            22%
"Kabar"           2%            12%            16%
"Pisang"         ~0%          0.01%             1%
                fokus        seimbang       kreatif/kacau
```

| Temperature | Cocok untuk |
|---|---|
| 0 – 0.3 | Kode, matematika, fakta |
| 0.7 – 1.0 | Obrolan biasa |
| 1.2 + | Brainstorming, puisi (risiko ngawur) |

Temperature 0 = greedy.

## Kontrol 2 — Top-k dan Top-p

Buang kata yang terlalu tidak mungkin **sebelum** melempar dadu:

- **Top-k = 3**: simpan hanya 3 kata terbaik.
- **Top-p = 0.9**: simpan kata terbaik sampai jumlah peluangnya mencapai 90%.

Kebanyakan chatbot memakai **temperature + top-p**: jawaban bervariasi tapi tidak ngawur.

## Di mana ini ada di kode?

Fungsi `GPT.generate()` di [`mini_gpt/model.py`](../mini_gpt/model.py) menerapkan
temperature dan top-k. Coba bandingkan:

```bash
python mini_gpt/generate.py --prompt "Kancil " --temperature 0.2
python mini_gpt/generate.py --prompt "Kancil " --temperature 1.5
```

Atau lihat efek temperature pada angka saja:

```bash
python contoh/03_sampling_temperature.py
```
