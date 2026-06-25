#!/usr/bin/env python
"""
webSLM — fine-tune a small base model on YOUR domain data with LoRA, then merge and
push a full HF checkpoint that the webSLM build pipeline can quantize to WebGPU.

  domain data (JSONL chat)
        │  train_lora.py   (LoRA SFT → merge → push)
        ▼
  a merged model on Hugging Face (e.g. yourname/WebSLM-Medical-0.5B)
        │  webSLM Actions → Build domain SLM (Domain preset = Custom)
        ▼
  WebGPU .wasm + quantized weights, runnable in the browser

DATA FORMAT — one JSON object per line ("conversational"/chat):
  {"messages":[{"role":"system","content":"..."},
               {"role":"user","content":"..."},
               {"role":"assistant","content":"..."}]}
See finetune/data/{medical,legal,insurance}.jsonl for small starter sets — REPLACE
them with your own, larger data for a model that actually learns the domain.

USAGE (Colab T4 or any CUDA GPU):
  pip install -r finetune/requirements.txt
  python finetune/train_lora.py \
      --base Qwen/Qwen2.5-0.5B-Instruct \
      --data finetune/data/medical.jsonl \
      --push-merged yourname/WebSLM-Medical-0.5B \
      --epochs 3
  # bigger base on a 16 GB GPU? add  --bits 4  (QLoRA, needs bitsandbytes)

THEN build it: webSLM → Actions → "Build domain SLM" → Domain preset = Custom,
  custom_model_hf = yourname/WebSLM-Medical-0.5B, custom_arch = qwen2, custom_conv = qwen2.

NOTE: working scaffold pinned to finetune/requirements.txt. TRL's SFT API drifts across
releases; if you bump versions and hit an argument error, check the SFTConfig/SFTTrainer
signature for your installed trl (this script handles the tokenizer/processing_class rename).
"""
import argparse
import sys


def parse_args():
    ap = argparse.ArgumentParser(description="LoRA fine-tune → merge → push, for the webSLM pipeline.")
    ap.add_argument("--base", default="Qwen/Qwen2.5-0.5B-Instruct",
                    help="Base model HF id or local dir (small Instruct model recommended)")
    ap.add_argument("--data", required=True, help="Training data: a .jsonl (chat 'messages') file or a glob")
    ap.add_argument("--adapter-out", default="./lora-adapter", help="Where to save the trained LoRA adapter")
    ap.add_argument("--merge-out", default="./merged", help="Where to save the merged full checkpoint")
    ap.add_argument("--push-merged", default=None, metavar="REPO_ID",
                    help="Push the merged checkpoint to this HF repo id (needs `huggingface-cli login`)")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-seq-len", type=int, default=1024)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--bits", type=int, default=16, choices=[16, 4],
                    help="16 = fp16 LoRA (default); 4 = QLoRA (needs bitsandbytes, for larger bases)")
    ap.add_argument("--no-merge", action="store_true", help="Train only; skip the merge/push step")
    return ap.parse_args()


def main():
    args = parse_args()
    try:
        import torch
        from datasets import load_dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, PeftModel
        from trl import SFTTrainer, SFTConfig
    except ImportError as e:
        sys.exit(f"Missing dependency: {e}\n  pip install -r finetune/requirements.txt")

    # ── Tokenizer + dataset (render chat messages to text via the model's template) ──
    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    ds = load_dataset("json", data_files=args.data, split="train")
    if "messages" not in ds.column_names:
        sys.exit("Data must have a 'messages' field per line (chat format). See finetune/data/*.jsonl")

    def render(ex):
        return {"text": tok.apply_chat_template(ex["messages"], tokenize=False)}
    ds = ds.map(render, remove_columns=ds.column_names)
    print(f"Loaded {len(ds)} examples from {args.data}")

    # ── Base model (optionally 4-bit for QLoRA) ──────────────────────────────────
    quant_cfg = None
    if args.bits == 4:
        from transformers import BitsAndBytesConfig
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.float16, device_map="auto",
        quantization_config=quant_cfg, trust_remote_code=True,
    )

    lora = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    sft_cfg = SFTConfig(
        output_dir=args.adapter_out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=5,
        save_strategy="epoch",
        max_seq_length=args.max_seq_len,
        dataset_text_field="text",
        packing=False,
        fp16=True,
        report_to="none",
    )

    # TRL renamed the `tokenizer` kwarg to `processing_class`; support both.
    try:
        trainer = SFTTrainer(model=model, train_dataset=ds, args=sft_cfg,
                             peft_config=lora, processing_class=tok)
    except TypeError:
        trainer = SFTTrainer(model=model, train_dataset=ds, args=sft_cfg,
                             peft_config=lora, tokenizer=tok)

    print("==> Training…")
    trainer.train()
    trainer.save_model(args.adapter_out)
    tok.save_pretrained(args.adapter_out)
    print(f"✓ Adapter saved -> {args.adapter_out}")

    if args.no_merge:
        print("--no-merge set; stopping after adapter. Merge later with merge_lora.py.")
        return

    # ── Merge: reload the base in fp16 (NOT 4-bit), fuse the adapter, save/push ──
    del model, trainer
    if 'torch' in dir() and torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("==> Merging adapter into a full fp16 checkpoint…")
    base = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.float16, trust_remote_code=True)
    merged = PeftModel.from_pretrained(base, args.adapter_out).merge_and_unload()
    merged.save_pretrained(args.merge_out, safe_serialization=True)
    tok.save_pretrained(args.merge_out)
    print(f"✓ Merged checkpoint -> {args.merge_out}")

    if args.push_merged:
        print(f"==> Pushing merged checkpoint to HF: {args.push_merged}")
        merged.push_to_hub(args.push_merged)
        tok.push_to_hub(args.push_merged)
        print(f"✓ Pushed. Build it via webSLM Actions (Domain preset = Custom, custom_model_hf = {args.push_merged}).")
    else:
        print("No --push-merged given. Push it yourself, or build the local dir with build.sh:")
        print(f"  MODEL_HF={args.merge_out} ARCH=qwen2 CONV=qwen2 NAME=WebSLM-Custom ./build.sh")


if __name__ == "__main__":
    main()
