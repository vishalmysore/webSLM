# webSLM: Fine-tuning, Compiling, and Running Domain-Specific Small Language Models Entirely in the Browser

**A technical whitepaper**

*Project: [vishalmysore/webSLM](https://github.com/vishalmysore/webSLM) · Demo: [vishalmysore/webSLMDemo](https://github.com/vishalmysore/webSLMDemo) · Last updated: 2026-06-29*

---

## Abstract

webSLM is an end-to-end pipeline for turning a general-purpose Small Language Model (SLM) into a **domain-specialized assistant that runs 100% in the browser** — no server, no API key, no inference cost, full offline capability after first load. This paper documents the complete lifecycle of a worked example, `WebSLM-Medical-0.5B`: (1) **LoRA fine-tuning** of `Qwen2.5-0.5B-Instruct` on a small domain dataset using a free Colab T4; (2) **compilation and 4-bit quantization** to a WebGPU model library via a reproducible, GPU-free GitHub Actions workflow built on MLC-LLM v0.19.0; and (3) **in-browser execution** through WebLLM, including the non-obvious runtime-integration details that determine whether the model loads at all.

We also report a controlled A/B validation showing what fine-tuning on a small dataset actually changes — and what it does not — and document two decoding pitfalls (greedy repetition loops and penalty-induced language drift on bilingual base models) that materially affect output quality on sub-1B models.

---

## 1. Introduction

The dominant narrative in language modeling has been scale. But a parallel track — Small Language Models in the 0.5B–7B range, *designed* to be efficient rather than pruned down — has matured to the point where a properly fine-tuned 0.5B–1.5B model delivers genuinely useful behavior in a focused domain. At the same time, **WebGPU** has made it possible to run these models directly inside a browser tab, on the user's own hardware.

The gap webSLM closes is not a research problem; it is an **engineering and tooling problem**. Getting from a Hugging Face checkpoint to a working in-browser, domain-specialized chatbot requires stitching together three independently fiddly stages — fine-tuning, MLC compilation/quantization, and WebLLM runtime wiring — each with version-sensitive environments and undocumented failure modes. This paper is the field manual for that path.

### 1.1 Design principles

- **Model-agnostic.** Any MLC-LLM-supported base works (Qwen2/2.5, Llama-3.x, Gemma-2, Phi-3.5, Mistral). Nothing in the pipeline is specific to one architecture.
- **Training and compilation are separate steps, on purpose.** Fine-tuning needs a GPU (Colab); compilation is CPU-only codegen (CI). The two never run in the same environment, so neither inherits the other's dependency constraints.
- **Reproducible, GPU-free builds.** The expensive, error-prone compilation runs in GitHub Actions with a pinned, from-source toolchain — no local Linux GPU box required.
- **Browser-first deployment.** The artifact is static files (weight shards + a `.wasm`) on a CDN; the client is one HTML page.

---

## 2. Background

### 2.1 SLM vs. quantized LLM

These are frequently conflated but are different answers to the same "large models are expensive" problem. An **SLM** is small *by design* — its efficiency comes from architecture and curated training data. A **quantized LLM** is a large model compressed *after* training; it keeps the full parameter structure and capability profile of the original, just at lower numeric precision. For the browser, only the SLM path is viable: a 1.5B model uses ~1–2 GB of GPU memory, while an INT4-quantized 70B model still needs 30–40 GB. They are not in the same deployment category.

Note that webSLM uses quantization **on top of** an SLM: the 0.5B model is itself quantized to 4-bit (q4f16_1) for the browser. Quantization here is a deployment compression, not the source of "smallness."

### 2.2 Fine-tuned SLM vs. RAG

RAG (Retrieval-Augmented Generation) is a *system architecture*, not a model type: it injects retrieved documents into the prompt at query time. It excels at large, dynamic knowledge bases but requires a retrieval layer (embeddings, vector store, ingestion) — infrastructure that does not exist in a pure browser deployment. Fine-tuning encodes **behavior, style, and domain patterns into the weights** instead. The two are complementary; for the serverless browser case, fine-tuning is the only specialization mechanism available. (The companion demo includes an optional in-browser TF-IDF RAG path for comparison, but the model itself carries no retrieval.)

### 2.3 The runtime stack: WebLLM + MLC-LLM

- **WebLLM** ([github.com/mlc-ai/web-llm](https://github.com/mlc-ai/web-llm)) is the browser runtime. It executes models on **WebGPU** and exposes an OpenAI-compatible JS API (`engine.chat.completions.create()`), with streaming, fully local. It **cannot** load a raw Hugging Face checkpoint.
- **MLC-LLM** (part of the Apache TVM ecosystem) is the compiler. For a browser target it produces two things WebLLM consumes:
  1. **Quantized weight shards** (`params_shard_*.bin`) plus a manifest (`ndarray-cache.json`) and chat/sampling config (`mlc-chat-config.json`).
  2. A **`.wasm` model library** containing the compiled compute kernels for that *specific* architecture.

The critical, easily-missed consequence: **the `.wasm` is tied to the runtime ABI.** A model library compiled with MLC-LLM v0.19.0 must be loaded by the matching `@mlc-ai/web-llm` build (0.2.79). Mismatched versions fail to instantiate the wasm.

---

## 3. System architecture

```
  domain data (JSONL chat)
        │
        │   STAGE 1 — Colab T4 (GPU)
        │   finetune/train_lora.py:  LoRA SFT → merge_and_unload → push
        ▼
  merged HF checkpoint                       e.g. VishalMysore/WebSLM-Medical-0.5B
  (standard fp16 safetensors)                (qwen2, 0.5B, full weights)
        │
        │   STAGE 2 — GitHub Actions (CPU only)
        │   normalize_config → convert_weight (q4f16_1) → gen_config → compile --device webgpu
        ▼
  MLC artifacts                              VishalMysore/WebSLM-Custom-MLC
  (params_shard_*.bin + .wasm + configs)     (8 shards ~278 MB + 3.7 MB wasm)
        │
        │   STAGE 3 — Any WebGPU browser (client GPU)
        │   @mlc-ai/web-llm@0.2.79: MLCEngine(appConfig) → reload → chat.completions
        ▼
  in-browser domain assistant               webSLMDemo (GitHub Pages)
```

Three environments, three hardware profiles, zero shared dependencies. The handoff between stages is always a plain artifact (an HF repo), never a live process.

---

## 4. Stage 1 — Fine-tuning on Colab (LoRA SFT)

### 4.1 Data format

Training data is **chat/conversational JSONL** — one JSON object per line:

```json
{"messages":[
  {"role":"system","content":"You are a careful medical information assistant. Provide general, educational health information in plain language, and always recommend consulting a licensed healthcare professional for diagnosis or treatment."},
  {"role":"user","content":"What are common signs of dehydration?"},
  {"role":"assistant","content":"Common signs include increased thirst, dry mouth, dark yellow urine, reduced urination, fatigue, dizziness, and headache. Severe dehydration can cause confusion or a rapid heartbeat and needs urgent care. Drink fluids and seek medical help if symptoms are severe or persist."}
]}
```

Two design rules carry most of the signal:

1. **Keep the `system` message identical across the dataset.** The model learns a stable persona from it. (This system prompt becomes the one you should use at inference — see §6.4 and §7.)
2. **The `assistant` turns are exactly what the model imitates.** Their length, register, and structure *are* the behavior you are training. The worked example's answers are short, plain-language, and always close with a referral to a professional — and that is precisely the signature the fine-tune learns.

The repository ships three **illustrative starter sets** in `finetune/data/` (`medical.jsonl` ≈ 34 examples, plus `legal` and `insurance`). These are deliberately tiny: they prove the pipeline, they do **not** fully specialize a model. Real domain quality needs hundreds to thousands of examples.

### 4.2 The training recipe

Fine-tuning uses **LoRA** (Low-Rank Adaptation) via TRL's `SFTTrainer`, rendering each example through the base model's chat template before training. The merged result is a standard HF checkpoint the build stage can consume directly.

| Hyperparameter | Value | Notes |
|---|---|---|
| Base model | `Qwen/Qwen2.5-0.5B-Instruct` | any Instruct SLM; `arch=qwen2`, `conv=qwen2` flow straight to build |
| LoRA rank `r` | 16 | |
| LoRA `alpha` | 32 | |
| LoRA dropout | 0.05 | |
| Target modules | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` | all attention + MLP projections |
| Epochs | 3 | |
| Learning rate | 2e-4 | |
| Per-device batch | 2 | |
| Grad accumulation | 8 | effective batch ≈ 16 |
| Max sequence length | 1024 | |
| Precision | fp16 | QLoRA (`--bits 4`, bitsandbytes) available for larger bases on 16 GB GPUs |
| Packing | off | |

The core of `train_lora.py`:

```python
tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
ds  = load_dataset("json", data_files=args.data, split="train")
ds  = ds.map(lambda ex: {"text": tok.apply_chat_template(ex["messages"], tokenize=False)},
             remove_columns=ds.column_names)

lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                  task_type="CAUSAL_LM",
                  target_modules=["q_proj","k_proj","v_proj","o_proj",
                                  "gate_proj","up_proj","down_proj"])

trainer = SFTTrainer(model=model, train_dataset=ds, args=sft_cfg,
                     peft_config=lora, processing_class=tok)   # falls back to tokenizer= on older TRL
trainer.train()
```

After training, the adapter is **merged** back into the base in fp16 and saved as a self-contained checkpoint (this is mandatory — MLC-LLM's `convert_weight` expects a fused model, not LoRA deltas):

```python
base   = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.float16)
merged = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()
merged.save_pretrained(out, safe_serialization=True)
tok.save_pretrained(out)               # carry the tokenizer so the dir is self-contained
merged.push_to_hub("yourname/WebSLM-Medical-0.5B")
```

> If you trained a LoRA adapter elsewhere (axolotl, Unsloth, raw PEFT), `merge_lora.py` does just the merge+export step. If you did a *full* fine-tune (no LoRA) you already have a standard checkpoint — skip merging entirely.

### 4.3 Running it on Colab

The fastest path is the clone-and-run notebook `finetune/finetune_webslm_colab.ipynb` (Runtime → **T4 GPU**). It clones the repo, installs `finetune/requirements.txt`, logs into Hugging Face, trains on the chosen domain, and **pushes the merged model to your HF account**. On a T4, a few hundred examples train in well under an hour. The final cell prints exactly what to enter in the build Action.

Equivalent CLI:

```bash
pip install -r finetune/requirements.txt
python finetune/train_lora.py \
    --base Qwen/Qwen2.5-0.5B-Instruct \
    --data finetune/data/medical.jsonl \
    --push-merged yourname/WebSLM-Medical-0.5B \
    --epochs 3
```

The output of Stage 1 is a plain fp16 HF model repo — e.g. `VishalMysore/WebSLM-Medical-0.5B` (qwen2 architecture, single `model.safetensors`, tokenizer, configs). Nothing browser-specific yet.

---

## 5. Stage 2 — Compiling to WebGPU via GitHub Actions

This is the stage that, done by hand, costs newcomers a day or more. webSLM encodes the entire toolchain build and the three-command MLC pipeline into `.github/workflows/build-slm.yml`, triggered manually (`workflow_dispatch`) with a domain preset **or** a Custom path pointing at your fine-tune.

### 5.1 Why build the toolchain from source

The workflow builds **MLC-LLM v0.19.0 and TVM from source**, because the MLC nightly wheels are broken by the in-progress `apache-tvm-ffi` migration. The compile step is **CPU-only**: `mlc_llm compile --device webgpu` is code generation via emscripten — it emits WebAssembly kernels and never needs a GPU. A full toolchain build is ~35–45 minutes; the workflow has a 350-minute ceiling and frees disk space first.

Build environment (the parts that matter):

| Component | Pin / setting | Reason |
|---|---|---|
| MLC-LLM | v0.19.0, from source | nightly wheels broken (tvm-ffi migration) |
| TVM | bundled `3rdparty/tvm`, from source | `USE_LLVM "llvm-config --link-static"`, `HIDE_PRIVATE_SYMBOLS ON`, all GPU backends OFF |
| LLVM | system `llvm-dev` + matching `libpolly-*-dev` | TVM static-links LLVM incl. Polly; `llvm-dev` doesn't ship `libPolly.a` |
| emscripten | 3.1.56 | wasm toolchain matching MLC-LLM's web runtime |
| Rust | latest stable (≥1.85) | modern deps; `tokenizers-cpp` needs `--cap-lints=allow` on new Rust |

A subtle linking detail handled by the workflow: `mlc_llm compile --device webgpu` links several wasm bitcode runtimes — `mlc_wasm_runtime.bc` (from `mlc-llm/web`) and `wasm_runtime.bc` / `tvmjs_support.bc` / `webgpu_runtime.bc` (from `tvm/web`). The latter three are built and copied into TVM's `build/` so its `find_lib_path` discovers them.

### 5.2 The MLC compile pipeline

Once the toolchain exists, the actual conversion is three commands (mirrored in `build.sh` for local/WSL2 runs), preceded by a config-normalization step:

```bash
# 1b. Make a freshly-merged (newer-transformers) config readable by mlc-llm v0.19.0
python normalize_config.py "hf/$NAME"

# 2. Quantize + shard the weights  (HF -> MLC params)
mlc_llm convert_weight "hf/$NAME" --quantization q4f16_1 --model-type qwen2 -o "$W"

# 3. Emit chat template + tokenizer + model metadata
mlc_llm gen_config "hf/$NAME" --quantization q4f16_1 --model-type qwen2 \
    --conv-template qwen2 --prefill-chunk-size 1024 -o "$W"

# 4. Codegen the WebGPU model library
mlc_llm compile "$W/mlc-chat-config.json" --device webgpu \
    -o "$L/$NAME-q4f16_1-webgpu.wasm"
```

#### 5.2.1 The config-normalization gotcha

Recent `transformers` releases changed the config schema: the RoPE base moved from a top-level `rope_theta` into a nested `rope_parameters`/`rope_scaling` dict, and `torch_dtype` became `dtype`. MLC-LLM v0.19.0 still expects the old top-level keys, so `convert_weight` on a freshly fine-tuned model fails with:

```
TypeError: QWen2Config.__init__() missing 1 required positional argument: 'rope_theta'
```

`normalize_config.py` hoists `rope_theta` back to the top level and restores `torch_dtype` — decoupling your *training* transformers version from the *pinned, old* build compiler. It is idempotent and safe on already-old configs:

```python
if "rope_theta" not in cfg:
    for key in ("rope_parameters", "rope_scaling"):
        rp = cfg.get(key)
        if isinstance(rp, dict) and rp.get("rope_theta") is not None:
            cfg["rope_theta"] = rp["rope_theta"]; break
if "torch_dtype" not in cfg and "dtype" in cfg:
    cfg["torch_dtype"] = cfg["dtype"]
```

This is the kind of failure that, undiagnosed, produces a silent crash or garbage output with no upstream documentation.

#### 5.2.2 Quantization choice

Default is **`q4f16_1`** (4-bit weights, fp16 activations) — the smallest practical format. Some models **overflow fp16 to NaN** during inference (observed with TinyLlama-1.1B); for those, fall back to **`q4f32_1`** (fp32 activations), trading size for numerical stability. The selected format is part of the wasm filename and the `model_id`, so it must match between the compiled artifact and the browser config.

### 5.3 Inputs and outputs

The workflow exposes domain presets (Qwen2.5-Coder-1.5B for code, Qwen2.5-Math-1.5B for math, general Qwen/Llama/Gemma/Phi bases) and a **Custom** path. For a fine-tune you select **Custom** and pass:

```
domain          = Custom
custom_model_hf = VishalMysore/WebSLM-Medical-0.5B
custom_arch     = qwen2
custom_conv     = qwen2
quant           = q4f16_1
custom_name     = WebSLM-Custom        (becomes the output repo/lib name)
upload_hf       = true                 (push artifacts to HF; needs HF_TOKEN + HF_NAMESPACE)
```

Outputs are saved **unconditionally** (so a bad HF token never loses a 45-minute build): a downloadable **Actions artifact**, a **GitHub Release** carrying the `.wasm`, and — when `upload_hf` is set — a self-contained **Hugging Face model repo**. The worked example produced `VishalMysore/WebSLM-Custom-MLC`:

| File | Size | Role |
|---|---|---|
| `mlc-chat-config.json` | 2 KB | chat template, sampling defaults, context window |
| `ndarray-cache.json` | 102 KB | weight-shard manifest |
| `params_shard_0…7.bin` | ~278 MB total | 4-bit quantized weights (8 shards) |
| `tokenizer.json`, `tokenizer_config.json` | ~11 MB | tokenizer |
| `libs/WebSLM-Custom-q4f16_1-webgpu.wasm` | 3.7 MB | WebGPU model library |

Compiled metadata of note: `model_type: qwen2`, `quantization: q4f16_1`, `context_window_size: 32768`, `vocab_size: 151936`, `conv_template: qwen2`, default sampling `temperature: 0.7`, `top_p: 0.8`.

---

## 6. Stage 3 — Running in the browser with WebLLM

The client is a static page importing `@mlc-ai/web-llm`. The model loads from its HF URLs, caches in the browser (Cache API / IndexedDB), and runs on the client GPU. Three integration details determine whether it works at all — each one produced a distinct, opaque error during bring-up.

### 6.1 Registering a custom model — `appConfig` goes in the constructor

WebLLM only knows its built-in (prebuilt) models unless you give it an `appConfig` describing yours. The **`appConfig` must be passed to the `MLCEngine` constructor**, not to `reload()`:

```js
import * as webllm from "https://esm.run/@mlc-ai/web-llm@0.2.79";

const appConfig = {
  model_list: [{
    model:     "https://huggingface.co/VishalMysore/WebSLM-Custom-MLC",          // FULL HF URL
    model_id:  "WebSLM-Custom-q4f16_1-webgpu",                                   // arbitrary local name
    model_lib: "https://huggingface.co/VishalMysore/WebSLM-Custom-MLC/resolve/main/libs/WebSLM-Custom-q4f16_1-webgpu.wasm",
  }],
};

const engine = new webllm.MLCEngine({ appConfig, initProgressCallback });
await engine.reload("WebSLM-Custom-q4f16_1-webgpu");
```

`reload(modelId, chatOpts?)`'s second argument is `ChatOptions`, which has **no `appConfig` field** — passing `appConfig` there is silently dropped, the engine falls back to its prebuilt list, and you get:

```
Cannot find model record in appConfig for WebSLM-Custom-q4f16_1-webgpu.
```

### 6.2 `model` must be a full URL

The `model_list[].model` field must be a complete `https://huggingface.co/{USER}/{REPO}` URL (the four accepted forms all start with `https://`). A bare repo id makes WebLLM's internal `new URL(...)` throw:

```
Failed to construct 'URL': Invalid URL
```

`model_lib` is the full `/resolve/main/.../*.wasm` URL; `model_id` is a free-form local handle.

### 6.3 Pin the runtime to the wasm's ABI

The import must be **version-pinned** to the `web-llm` build matching the MLC-LLM that produced the wasm:

```js
import * as webllm from "https://esm.run/@mlc-ai/web-llm@0.2.79";   // NOT unpinned (=latest)
```

`esm.run/@mlc-ai/web-llm` with no version resolves to *latest*, whose wasm ABI can differ from a v0.19.0-compiled library — producing instantiation failures at load. The pin table:

| Built with | Runtime |
|---|---|
| MLC-LLM v0.19.0 | `@mlc-ai/web-llm@0.2.79` |

### 6.4 Inference and decoding

WebLLM's API is OpenAI-shaped and streams:

```js
const stream = await engine.chat.completions.create({
  messages: [
    { role: "system", content: TRAINING_SYSTEM_PROMPT },   // match the prompt the model was TRAINED with
    { role: "user",   content: query },
  ],
  stream: true,
  temperature: 0.7, top_p: 0.8, max_tokens: 256,            // Qwen2.5's recommended sampling
});
for await (const chunk of stream) { /* chunk.choices[0].delta.content */ }
```

Two points are decisive for output quality on a 0.5B model (see §7.3):

- **Use the training system prompt at inference.** The fine-tune's behavior is conditioned on the persona it was trained with; a different system prompt pulls it back toward generic base behavior.
- **Use the model's recommended sampling** (`temperature 0.7`, `top_p 0.8`). Greedy decoding and aggressive penalties both degrade small-model output in characteristic ways.

### 6.5 The demo application

The companion demo ([webSLMDemo](https://github.com/vishalmysore/webSLMDemo), deployed to GitHub Pages) has two modes:

- **Product demo** — *Base + RAG* (left) vs. *Fine-tuned SLM* (right). The base panel runs a general model with an in-browser **TF-IDF retriever** injecting document context; the SLM panel runs the fine-tune with a domain system prompt and no retrieval. This contrasts the two specialization strategies.
- **Fine-tuning proof** — a controlled A/B (next section). It loads the **exact base** the fine-tune started from (`Qwen2.5-0.5B-Instruct-q4f16_1-MLC`, a WebLLM prebuilt at the same quantization) on the left and the fine-tune on the right, feeds **both** the identical training system prompt with no retrieval, and uses **identical decoding** — so the only variable is the LoRA training.

---

## 7. Validation: what fine-tuning actually changed

Because the proof mode holds base, prompt, and decoding identical across both panels, any difference is attributable to the LoRA fine-tuning. The following are verbatim in-browser outputs.

### 7.1 In-distribution question (trained topic)

**Prompt:** *"What are common signs of dehydration?"* (a question whose topic is in the training set.)

> **Base — Qwen2.5-0.5B, no fine-tune:** "1. Sunken Eyes… 6. Confusion… 9. Dry Skin: Not having much moisture in your skin. 10. Dry Skin… 11. Dry Skin…" — rambling, and falls into a repeat loop to the token limit.
>
> **Fine-tune — WebSLM-Medical-0.5B:** "Common signs of dehydration include: 1. Sunken eyes 2. Dry mouth and lips … 4. Urine that is dark or less than normal … 7. Not sweating heavily or feeling weak. These symptoms can be caused by low fluid intake, heatstroke… If you notice any of these signs, it's important to seek medical attention right away."

The fine-tune reproduces the **trained content** (thirst, dark urine, reduced urination, rapid heartbeat — closely mirroring its `medical.jsonl` answer), is **concise**, **stops cleanly**, and **closes with a referral** — exactly the trained signature. The base is verbose and unstable. Fine-tuning here improved both **style** and **generation stability**.

### 7.2 Held-out question (untrained topic)

**Prompt:** *"What is long covid?"* (a topic the 34-example dataset never covered.)

> **Base:** invents an alias ("also known as Long-Term Exposure") but gets the chronic/months-to-years framing roughly right.
> **Fine-tune:** cleaner structure and plausible symptom list, but states symptoms last "an average of two to three weeks" — which is wrong (that is *acute* COVID; long COVID lasts months by definition).

**Honest finding:** on a held-out topic the fine-tuning signal is weak and **both models hallucinate**. A 34-example LoRA imparts *style and in-distribution phrasing*, not reliable new *knowledge*, and certainly not factual reliability on topics outside the training distribution. This is expected and is the central caveat for small-data fine-tuning.

### 7.3 Decoding pitfalls (general to sub-1B WebLLM models)

Decoding choice changes outputs as much as fine-tuning does on these models:

| Decoding | Effect on the 0.5B models |
|---|---|
| Greedy (`temperature: 0`) | Degenerate **repetition loops** ("Confusion. Confusion…") that bury the trained style |
| Low temp + strong `frequency_penalty`/`presence_penalty` | **Language drift**: penalizing repeated English tokens pushes a *bilingual* Qwen model into Chinese tokens → gibberish loops ("答? 答答?") |
| **`temperature 0.7`, `top_p 0.8`** (Qwen's own recommended) | Stable, coherent, no loops, no drift — the chosen setting |

The takeaway: small models are decoding-sensitive. Fairness in the A/B is preserved by applying identical decoding to both panels; quality is preserved by using the model's recommended sampling rather than greedy or penalty-heavy schemes.

---

## 8. Reproducibility

Everything required to reproduce `WebSLM-Medical-0.5B` is public and pinned:

- **Data & training:** `finetune/data/medical.jsonl`, `finetune/train_lora.py`, `finetune/finetune_webslm_colab.ipynb` (Colab T4).
- **Build:** `.github/workflows/build-slm.yml` (CI) or `build.sh` (Linux/WSL2), both invoking `normalize_config.py` then the three MLC commands.
- **Artifacts:** [`VishalMysore/WebSLM-Medical-0.5B`](https://huggingface.co/VishalMysore/WebSLM-Medical-0.5B) (merged fp16) and [`VishalMysore/WebSLM-Custom-MLC`](https://huggingface.co/VishalMysore/WebSLM-Custom-MLC) (compiled, 4-bit, + wasm).
- **Client:** the demo's `app.js`, pinned to `@mlc-ai/web-llm@0.2.79`.

### Version matrix (the pins that matter)

| Layer | Pin |
|---|---|
| Base model | `Qwen/Qwen2.5-0.5B-Instruct` |
| Fine-tuning | LoRA (r=16, α=32) via TRL `SFTTrainer`, 3 epochs, fp16 |
| Compiler | MLC-LLM **v0.19.0**, built from source (+ bundled TVM) |
| wasm toolchain | emscripten **3.1.56** |
| Quantization | **q4f16_1** (fallback `q4f32_1` on fp16 NaN) |
| Browser runtime | `@mlc-ai/web-llm` **0.2.79** (must match the compiler) |
| Inference sampling | `temperature 0.7`, `top_p 0.8` |

### Repository components

| Path | Purpose |
|---|---|
| `finetune/train_lora.py` | LoRA SFT → merge → push |
| `finetune/finetune_webslm_colab.ipynb` | clone-and-run Colab trainer |
| `finetune/data/*.jsonl` | starter domain datasets (illustrative) |
| `merge_lora.py` | merge an externally-trained adapter into its base |
| `normalize_config.py` | newer-transformers → mlc-llm v0.19.0 config fix |
| `build.sh` | local/WSL2 convert → gen_config → compile |
| `.github/workflows/build-slm.yml` | from-source toolchain + full compile/release/upload |
| `demo/index.html` (and webSLMDemo) | self-contained WebLLM client |

---

## 9. Limitations and responsible use

- **Browser favors small models.** 0.5B–3.5B is the practical sweet spot; 7B+ is slow and memory-pressured on consumer GPUs because WebGPU memory is shared with the browser process.
- **Small-data fine-tuning transfers style, not knowledge.** As §7.2 shows, expect on-distribution behavioral alignment, not factual reliability — and expect hallucination off-distribution.
- **Quant/version coupling is brittle.** The wasm ↔ runtime pin is mandatory; fp16 quantization can NaN on some architectures.
- **Sensitive domains require a human in the loop.** The medical/legal/insurance examples exist to demonstrate the pipeline. Outputs can be confidently wrong; keep disclaimers (the sample data trains one in) and validate before relying on any output.

---

## 10. Conclusion

The distance between a capable small language model and a useful, private, offline, domain-specific browser assistant is bridged by tooling, not research. webSLM makes that bridge reproducible: fine-tune on a free Colab GPU, compile and quantize in CPU-only CI with a pinned from-source MLC-LLM toolchain, and serve the result as static files that any WebGPU browser runs locally. The worked example, `WebSLM-Medical-0.5B`, demonstrates the full path end to end — and a controlled in-browser A/B confirms, honestly, both what a small fine-tune buys (concise, stable, on-style, in-distribution behavior) and what it does not (new knowledge or factual reliability). For focused, behavior-defined applications that must run without a server, that trade is often exactly the right one.
