# Build a domain SLM in Google Colab

Interactive alternative to the GitHub Actions build — handy for fast iteration and for
trying different base models / quantizations before you commit to a CI run.

**Open it directly in Colab:**

https://colab.research.google.com/github/yourname/webSLM/blob/main/colab/build_webslm_colab.ipynb

(or in Colab: File → Open notebook → GitHub → `yourname/webSLM`)

## Steps
1. Runtime → Change runtime type → **T4 GPU** (free). The GPU isn't required for the WebGPU
   compile (that's emcc codegen), but it speeds up the optional LoRA-merge step.
2. Run the cells top to bottom. The TVM build (~25–35 min) runs once and stays built for the
   session, so you can re-run the **compile** cell freely while iterating across models.
3. Set the model variables near the bottom (`MODEL`, `ARCH`, `CONV`, `NAME`, `QUANT`).
4. Paste a Hugging Face **Write** token at the login cell to upload the weights + wasm.

## Why this builds from source
The current MLC nightly wheels are ABI-broken (mid `apache-tvm-ffi` migration). This notebook
builds TVM + mlc-llm **from source** pinned to **`v0.19.0`** (the last pre-migration release),
with the LLVM/Polly and cargo-sparse-index fixes baked in — the same recipe as
[`.github/workflows/build-slm.yml`](../.github/workflows/build-slm.yml), just interactive.

## Fine-tuned base?
If you trained a LoRA adapter on domain data, merge it first with
[`merge_lora.py`](../merge_lora.py) (a cell for this is included), then point the compile cell
at the merged checkpoint dir.

## Output → browser
- Weights → Hugging Face (`yourname/WebSLM-<Domain>-MLC`)
- `.wasm` → download from `dist-model/libs/` and attach to a GitHub Release (or push to HF)

Then load both URLs in [`demo/index.html`](../demo/index.html) or your own WebLLM app.
You can't test WebGPU in Colab — testing happens in the browser.
