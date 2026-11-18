"""Bab 07 — Fine-tuning model terbuka dengan LoRA (Hugging Face TRL + PEFT).

Butuh paket tambahan dan akan mengunduh model ±1 GB dari Hugging Face:

    pip install -r requirements-finetune.txt
    python contoh/05_fine_tune_lora.py

Data: contoh/data_sft_contoh.jsonl (format {"messages": [...]}).
Untuk hasil yang berguna, ganti dengan ratusan sampai ribuan contoh milik Anda.
"""

from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import pipeline
from trl import SFTConfig, SFTTrainer

FOLDER = Path(__file__).resolve().parent
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"   # model kecil supaya cepat dicoba
KELUARAN = FOLDER.parent / "hasil-lora"

# 1. Data
dataset = load_dataset("json", data_files=str(FOLDER / "data_sft_contoh.jsonl"), split="train")

# 2. LoRA: latih adapter kecil, bobot asli dibekukan
peft_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules="all-linear", task_type="CAUSAL_LM",
)

# 3. Konfigurasi training
args = SFTConfig(
    output_dir=str(KELUARAN),
    num_train_epochs=5,                  # data contoh sangat sedikit, jadi beberapa epoch
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    logging_steps=1,
    save_strategy="no",
    bf16=torch.cuda.is_available(),
    report_to="none",
)

# 4. Training
trainer = SFTTrainer(model=MODEL, train_dataset=dataset, args=args, peft_config=peft_config)
trainer.train()
trainer.save_model()
print(f"\nAdapter LoRA disimpan ke {KELUARAN}")

# 5. Coba hasilnya
chat = pipeline("text-generation", model=trainer.model, tokenizer=trainer.processing_class)
for tanya in ["Jam berapa toko buka?", "Bisa bayar di tempat?"]:
    jawab = chat([{"role": "user", "content": tanya}], max_new_tokens=60)
    print(f"\nUser     : {tanya}\nAssistant: {jawab[0]['generated_text'][-1]['content']}")
