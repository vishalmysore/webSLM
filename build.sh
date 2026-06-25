#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# webSLM — compile a domain-specific Small Language Model to a WebGPU/WebLLM
# model library (.wasm) + quantized weights, runnable entirely in the browser.
#
# This is the PLAIN MLC-LLM compile path: convert_weight -> gen_config -> compile.
# There is NO model-definition patching here (no exposed hidden states, no latent
# loop) — webSLM ships ordinary chat SLMs. The pipeline is MODEL-AGNOSTIC: any base
# supported by MLC-LLM works (Qwen2/2.5, Llama, Gemma2, Phi-3/3.5, Mistral, ...).
#
# RUN ON LINUX OR WSL2 (the WebGPU target needs emscripten; not Windows-native).
# The reliable, reproducible path is the from-source build in CI / Colab — see
# .github/workflows/build-slm.yml and colab/. This script assumes a WORKING mlc_llm
# (built from source, v0.19.0) + emscripten are already on PATH.
#
# Quick start (after a from-source mlc-llm/TVM build, see colab/README.md):
#   ./build.sh                                   # builds the default (Qwen2.5-0.5B)
#   MODEL_HF=Qwen/Qwen2.5-Coder-1.5B-Instruct ARCH=qwen2 CONV=qwen2 \
#     NAME=WebSLM-Code-1.5B ./build.sh           # a code SLM
#   MODEL_HF=./merged/my-medical-slm ARCH=qwen2 CONV=qwen2 \
#     NAME=WebSLM-Medical-0.5B ./build.sh        # a fine-tuned checkpoint (see merge_lora.py)
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Config (override via env) ────────────────────────────────────────────────
MODEL_HF="${MODEL_HF:-Qwen/Qwen2.5-0.5B-Instruct}"  # HF repo id OR a local checkpoint dir
ARCH="${ARCH:-qwen2}"                                # mlc_llm model arch: qwen2 | llama | gemma2 | phi3 ...
CONV="${CONV:-qwen2}"                                # mlc_llm conversation template
QUANT="${QUANT:-q4f16_1}"                            # quantization (q4f16_1 small; q4f32_1 if f16 overflows)
NAME="${NAME:-WebSLM-0.5B}"                           # output name
OUT="${OUT:-./dist-model}"                            # output root
PREFILL_CHUNK="${PREFILL_CHUNK:-1024}"

WEIGHTS="$OUT/$NAME-MLC"                              # converted weights dir
LIBS="$OUT/libs"                                     # compiled wasm dir
mkdir -p "$WEIGHTS" "$LIBS"

echo "==> 0. Sanity: emscripten + mlc_llm present"
command -v emcc >/dev/null || { echo "emcc not found — 'source path/to/emsdk_env.sh' first"; exit 1; }
python -c "import mlc_llm; print('mlc_llm at', mlc_llm.__file__)"

echo "==> 1. Resolve base weights ($MODEL_HF)"
if [ -d "$MODEL_HF" ]; then
  HF_DIR="$MODEL_HF"                                 # already a local checkpoint (e.g. a merged fine-tune)
  echo "    using local checkpoint: $HF_DIR"
else
  HF_DIR="./hf/$NAME"
  hf download "$MODEL_HF" --local-dir "$HF_DIR"      # huggingface_hub 1.x CLI
fi

echo "==> 1b. Normalize config for mlc-llm v0.19.0 (newer-transformers compat)"
python "$(dirname "$0")/normalize_config.py" "$HF_DIR"

echo "==> 2. convert_weight  (HF -> MLC quantized params)"
mlc_llm convert_weight "$HF_DIR" \
  --quantization "$QUANT" \
  --model-type "$ARCH" \
  -o "$WEIGHTS"

echo "==> 3. gen_config  (chat template, tokenizer, model metadata)"
mlc_llm gen_config "$HF_DIR" \
  --quantization "$QUANT" \
  --model-type "$ARCH" \
  --conv-template "$CONV" \
  --prefill-chunk-size "$PREFILL_CHUNK" \
  -o "$WEIGHTS"

echo "==> 4. compile  ->  WebGPU wasm model library"
mlc_llm compile "$WEIGHTS/mlc-chat-config.json" \
  --device webgpu \
  -o "$LIBS/$NAME-$QUANT-webgpu.wasm"

cat <<EOF

==> DONE
   Weights : $WEIGHTS
   Wasm    : $LIBS/$NAME-$QUANT-webgpu.wasm

Next:
  • Upload $WEIGHTS to a Hugging Face model repo (e.g. you/$NAME-MLC).
  • Host $LIBS/$NAME-$QUANT-webgpu.wasm on a static CDN (GitHub Release / HF).
  • Point WebLLM at both URLs — see demo/index.html and model-card/README.md.
  • The .wasm is tied to the runtime: load it with the @mlc-ai/web-llm build that
    matches the mlc-llm you compiled with (v0.19.0 -> web-llm 0.2.79).
EOF
