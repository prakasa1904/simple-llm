# 07 — Fine-tuning dari Base LLM

Melatih model dari nol (seperti [`mini_gpt/`](../mini_gpt/)) bagus untuk **memahami**,
tetapi untuk **hasil yang berguna**, kita mulai dari model terbuka yang sudah dilatih
(keluarga Qwen, Llama, Gemma, Mistral, OLMo) lalu fine-tune.

Fine-tuning = **melanjutkan turun gunung dari posisi yang sudah dekat lembah**,
ke lembah yang sedikit berbeda (gaya bahasa atau domain Anda). Mesinnya sama persis
dengan training biasa ([05](05-training.md)); yang beda hanya titik awal dan datanya.

## Full fine-tuning vs LoRA vs QLoRA

| Metode | Yang dilatih | Kebutuhan GPU (model 7B) |
|---|---|---|
| **Full fine-tuning** | Semua bobot | Beberapa GPU 80 GB |
| **LoRA** | Adapter kecil (<1% parameter); bobot asli dibekukan | 1 GPU besar |
| **QLoRA** | LoRA di atas model yang dikompres ke 4-bit | 1 GPU 24 GB |

**Mulailah dengan LoRA/QLoRA.**

## Contoh kode (Hugging Face TRL + PEFT)

Lihat [`contoh/05_fine_tune_lora.py`](../contoh/05_fine_tune_lora.py). Intinya:

```python
trainer = SFTTrainer(
    model="Qwen/Qwen2.5-0.5B-Instruct",       # mulai dari model kecil
    train_dataset=dataset,                     # format {"messages": [...]}
    args=SFTConfig(output_dir="hasil", learning_rate=2e-4, num_train_epochs=2),
    peft_config=LoraConfig(r=16, lora_alpha=32, target_modules="all-linear",
                           task_type="CAUSAL_LM"),
)
trainer.train()
```

Setelah SFT berhasil, coba:
- **DPO** (`DPOTrainer`) dengan data `prompt/chosen/rejected`;
- **GRPO** (`GRPOTrainer`) — reinforcement learning dengan fungsi reward
  (misalnya memberi hadiah untuk jawaban matematika yang benar).

## Alat yang memudahkan

- **Unsloth** — LoRA lebih cepat & hemat memori, banyak notebook Colab siap pakai.
- **Axolotl / LLaMA-Factory** — fine-tuning cukup dengan file konfigurasi.
- **lm-evaluation-harness** — evaluasi standar; selalu bandingkan sebelum vs sesudah.
- **vLLM / Ollama** — menjalankan (serving) model hasil fine-tune.

## Nasihat praktis

1. **Mulai kecil** (0.5B – 1.5B). Iterasi dalam menit, bukan jam.
2. **Fine-tuning mengajarkan perilaku, bukan fakta.** Untuk menambah pengetahuan
   (misalnya dokumen perusahaan), **RAG** biasanya lebih baik dan lebih murah.
   Untuk data yang sering berubah, **jangan** fine-tuning ulang setiap ada data baru —
   lihat ⭐ [09 — Data yang Cepat Berubah](09-data-cepat-berubah.md).
3. **Chat template harus benar** — template yang salah sering membuat model rusak.
4. **Selalu evaluasi sebelum dan sesudah** — waspadai *catastrophic forgetting*.
5. **Kerjakan satu proyek nyata dari awal sampai akhir**: data → training → evaluasi → deploy.
