#!/usr/bin/env python
"""
webSLM — fine-tuning integration: merge a LoRA/PEFT adapter into its base model
and write a plain HF checkpoint that the compile pipeline can consume directly.

WHY THIS EXISTS
  The browser build pipeline (build.sh / build-slm.yml) takes a *standard* HF model
  directory and quantizes + compiles it to WebGPU. A LoRA adapter is NOT a standard
  model — it's a small set of delta weights. MLC-LLM's convert_weight expects the
  full fused model. So the domain-adaptation flow is:

      base model  +  your LoRA adapter (trained on domain data)
                       │  merge_lora.py
                       ▼
      a merged HF checkpoint  ──►  build.sh / build-slm.yml  ──►  WebGPU .wasm

  Train the LoRA however you like (PEFT, axolotl, Unsloth, ...). This script only
  does the merge + export step, so the result drops straight into the pipeline.

USAGE
  pip install torch transformers peft
  # merge to a local dir, then feed it to build.sh:
  python merge_lora.py --base Qwen/Qwen2.5-0.5B-Instruct \
      --adapter ./my-medical-lora --out ./merged/WebSLM-Medical-0.5B
  MODEL_HF=./merged/WebSLM-Medical-0.5B ARCH=qwen2 CONV=qwen2 \
      NAME=WebSLM-Medical-0.5B ./build.sh

  # or push the merged checkpoint to HF and build it via the 'Custom' workflow path:
  python merge_lora.py --base Qwen/Qwen2.5-0.5B-Instruct --adapter ./my-medical-lora \
      --out ./merged/WebSLM-Medical-0.5B --push-to-hub yourname/WebSLM-Medical-0.5B

NOTE
  If you fine-tuned WITHOUT LoRA (a full fine-tune) you already have a standard HF
  checkpoint — skip this script and point the pipeline straight at that directory.
"""
import argparse
import sys


def main():
    ap = argparse.ArgumentParser(description="Merge a LoRA/PEFT adapter into its base model for the webSLM pipeline.")
    ap.add_argument("--base", required=True, help="Base model HF id or local dir (must match what the adapter was trained on)")
    ap.add_argument("--adapter", required=True, help="LoRA/PEFT adapter dir or HF id")
    ap.add_argument("--out", required=True, help="Output dir for the merged HF checkpoint")
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"],
                    help="Merge/compute dtype (float16 is fine; the pipeline re-quantizes anyway)")
    ap.add_argument("--push-to-hub", default=None, metavar="REPO_ID",
                    help="Optionally push the merged checkpoint to this HF repo id")
    args = ap.parse_args()

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
    except ImportError as e:
        sys.exit(f"Missing dependency: {e}\n  pip install torch transformers peft")

    dtype = getattr(torch, args.dtype)

    print(f"==> Loading base model: {args.base}")
    base = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=dtype, trust_remote_code=True)

    print(f"==> Applying adapter: {args.adapter}")
    model = PeftModel.from_pretrained(base, args.adapter)

    print("==> Merging adapter weights into the base (merge_and_unload)")
    model = model.merge_and_unload()   # fuse LoRA deltas -> a plain causal-LM

    print(f"==> Saving merged checkpoint -> {args.out}")
    model.save_pretrained(args.out, safe_serialization=True)

    # Carry the tokenizer across so the merged dir is self-contained for gen_config.
    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    tok.save_pretrained(args.out)

    if args.push_to_hub:
        print(f"==> Pushing to HF: {args.push_to_hub}")
        model.push_to_hub(args.push_to_hub)
        tok.push_to_hub(args.push_to_hub)

    print(f"""
✓ Merged checkpoint ready: {args.out}
  Feed it to the compile pipeline:
    MODEL_HF={args.out} ARCH=<qwen2|llama|...> CONV=<qwen2|llama-3|...> NAME=<your-name> ./build.sh
  (or push it to HF and build via the 'Custom' path in .github/workflows/build-slm.yml)
""")


if __name__ == "__main__":
    main()
