# webSLM — domain-specific Small Language Models, compiled for the browser

A **customizable compilation pipeline** that turns a domain-specific base model (or your own
fine-tune) into a **WebGPU/WebLLM** model library that runs **entirely in the browser** — no
server, no API key, fully on-device and private.

This is the same MLC-LLM → WebGPU build mechanism as
[recursiveMASWebLLM](https://github.com/vishalmysore/recursiveMASWebLLM), **with the recursive /
latent part removed**. No model-definition patching, no exposed hidden states, no RecursiveLink —
just a clean, model-agnostic pipeline that ships ordinary chat SLMs to the browser. The value
here is the **pipeline**, not any one model.

```
 a domain base model (or your fine-tune)
        │   convert_weight → gen_config → compile --device webgpu
        ▼
 *.wasm  +  quantized weights        ──host on a GitHub Release + Hugging Face──▶
        │
        ▼
 WebLLM in the browser  ── loads by URL ──▶  on-device, private domain inference
```

## Model-agnostic

Any base supported by **MLC-LLM** works — Qwen2 / Qwen2.5, Llama, Gemma2, Phi-3 / 3.5, Mistral,
and their derivatives. You only supply three things: the HF model id, the mlc **arch**, and the
**conv-template**. Presets for common choices are wired into the CI workflow's dropdown.

## Pick a domain base

| Domain | Suggested base | Notes |
|---|---|---|
| **Code** | `Qwen/Qwen2.5-Coder-1.5B-Instruct` | strong tiny code model; preset included |
| **Math / reasoning** | `Qwen/Qwen2.5-Math-1.5B-Instruct` | preset included |
| **Medical** | `Qwen/Qwen2.5-0.5B` or `microsoft/Phi-3.5-mini` **fine-tuned** on PubMed/MedQA | fine-tune, then build (see below) |
| **Legal** | a small legal fine-tune of Llama-3.x / Mistral (stay small) | fine-tune, then build |
| **Other** | any strong small base + domain fine-tune (LoRA is efficient) | fine-tune, then build |

**Keep it small.** This targets in-browser WebGPU: 0.5B–2B is the sweet spot; ~3–4B (e.g.
Phi-3.5-mini) is the practical ceiling. 7B+ won't load comfortably in a browser tab.

## Two use cases

### 1. Pure domain SLM (no fine-tuning)
If a domain base already exists (code, math), you don't need to train anything. Just compile it
and use it normally in WebLLM for faster/private/offline domain inference in the browser.

### 2. Fine-tuning integration (medical / legal / insurance / your own)
Train a small base on your domain data first, then feed the result into this pipeline. The
**[`finetune/`](finetune/)** folder is a complete, clone-and-run workflow for this — starter
datasets, a LoRA training script, and a Colab:

```
your data (JSONL) ─finetune/─► merged model on HF ─build-slm.yml (Custom)─► WebGPU .wasm
```

- **Easiest:** open [`finetune/finetune_webslm_colab.ipynb`](finetune/finetune_webslm_colab.ipynb)
  in Colab (T4 GPU). It clones the repo, trains on the domain you pick (medical/legal/insurance
  or your own JSONL), and **pushes a finished model to your HF account**; then build it via the
  Action's **Custom** path. See [`finetune/README.md`](finetune/README.md).

  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vishalmysore/webSLM/blob/main/finetune/finetune_webslm_colab.ipynb)

- **Script:** `python finetune/train_lora.py --base Qwen/Qwen2.5-0.5B-Instruct
  --data finetune/data/medical.jsonl --push-merged yourname/WebSLM-Medical-0.5B`
- **Already trained?** A full fine-tune is a standard HF checkpoint — point the pipeline straight
  at it. A standalone LoRA adapter → merge with [`merge_lora.py`](merge_lora.py) first.

> Training (GPU) and quantization (CPU) are deliberately separate: `finetune/` makes the model,
> the build Action turns it into a browser SLM.

## Build it

`mlc_llm compile` is **code generation, not GPU execution** — it emits the WebGPU wasm on a plain
CPU. `convert_weight` / `gen_config` also run on CPU for small models. Three ways to build:

### A) GitHub Actions (no local Linux/GPU needed)
Run [`.github/workflows/build-slm.yml`](.github/workflows/build-slm.yml) from the **Actions** tab
→ *Run workflow*. Pick a **domain preset** (or **Custom** + your HF id / arch / conv / name),
choose quantization, optionally tick `upload_hf`. It builds mlc-llm **v0.19.0 from source**, then
`convert_weight` + `gen_config` + `compile --device webgpu`, and uploads the `.wasm` to a Release
(plus weights + wasm to HF when enabled).

Host the outputs so a browser can load them by URL:
- **weights** (many shard files) → **Hugging Face**: set repo **secret** `HF_TOKEN` + **variable**
  `HF_NAMESPACE` (e.g. `yourname`) and tick `upload_hf`.
- **wasm** (single file) → the Release asset URL (or the HF repo, which the workflow also pushes).

### B) Colab (interactive)
[`colab/build_webslm_colab.ipynb`](colab/build_webslm_colab.ipynb) — the same from-source recipe,
interactive, with editable model variables and an optional LoRA-merge cell. See
[`colab/README.md`](colab/README.md).

### C) Locally (Linux / WSL2)
After a from-source mlc-llm/TVM build + emscripten on PATH, run [`build.sh`](build.sh) (configure
via env vars: `MODEL_HF`, `ARCH`, `CONV`, `QUANT`, `NAME`).

## Use it in the browser

Point WebLLM at the two URLs via a CUSTOM model config — a ready-to-run page is in
[`demo/index.html`](demo/index.html):

```js
import * as webllm from "@mlc-ai/web-llm";  // pin to 0.2.79 (matches mlc-llm v0.19.0)
const appConfig = { model_list: [{
  model:     "https://huggingface.co/yourname/WebSLM-Code-1.5B-MLC",
  model_id:  "webslm-code-1.5b",
  model_lib: "https://huggingface.co/yourname/WebSLM-Code-1.5B-MLC/resolve/main/libs/WebSLM-Code-1.5B-q4f16_1-webgpu.wasm",
}]};
const engine = await webllm.CreateMLCEngine("webslm-code-1.5b", { appConfig });
```

## Files

| file | what |
|---|---|
| `.github/workflows/build-slm.yml` | the CI pipeline (CPU Ubuntu): domain presets + Custom override |
| `build.sh` | local `convert_weight` → `gen_config` → `compile --device webgpu` |
| `finetune/` | train a domain base on your data (datasets + LoRA script + clone-and-run Colab) |
| `merge_lora.py` | merge a separately-trained LoRA adapter → a checkpoint the pipeline builds |
| `normalize_config.py` | make newer-transformers `config.json` readable by the pinned mlc-llm v0.19.0 |
| `colab/` | interactive build notebook + notes |
| `demo/index.html` | minimal WebLLM browser page to load + test a compiled SLM |
| `model-card/README.md` | Hugging Face model-card template for a published SLM |

## Honest limits

- ⚠️ **Build from source, pinned to mlc-llm `v0.19.0`** — the current nightly wheels are broken by
  an in-progress `apache-tvm-ffi` migration. The from-source recipe (CI/Colab) is the reliable path.
- **Quantization:** `q4f16_1` is smallest. Some bases overflow f16 and produce NaN hidden states —
  switch to `q4f32_1` if you see garbage/NaNs (this bit TinyLlama-1.1B).
- **Small models only.** 0.5–2B comfortable in-browser; ~3–4B is the ceiling; 7B+ no.
- **Version pin:** the `.wasm` is tied to the runtime — load it with the matching `@mlc-ai/web-llm`
  (v0.19.0 → `0.2.79`), or you get a "model lib version" error.
- **Newer-transformers configs:** models saved by recent transformers nest the RoPE base in a
  `rope_parameters` dict instead of a top-level `rope_theta`, which mlc v0.19.0 rejects
  (`QWen2Config ... missing ... 'rope_theta'`). The build runs [`normalize_config.py`](normalize_config.py)
  to fix this automatically — no need to re-train or re-export your model.
- **Gated bases** (Llama, Gemma) need an `HF_TOKEN` with access at download time.

## License

MIT for the pipeline. Compiled models inherit their base model's license.
