# webSLM — fine-tune your own domain base model

Train a small base model (Qwen by default) on **your** medical / legal / insurance / any-domain
data, then hand the result to the webSLM build pipeline, which quantizes it into a WebGPU model
that runs in the browser.

```
your domain data (JSONL chat)
      │  train_lora.py   (LoRA fine-tune → merge → push)
      ▼
a merged model on Hugging Face            e.g. yourname/WebSLM-Medical-0.5B
      │  webSLM → Actions → "Build domain SLM" → Domain preset = Custom
      ▼
WebGPU .wasm + quantized weights → runs in demo/index.html, no server
```

**Training and quantization are separate steps, on purpose.** This folder trains the model
(needs a GPU — use the Colab below). The repo's GitHub Action quantizes it (CPU only). You don't
need MLC/emscripten to fine-tune.

## Fastest path — the clone-and-run Colab

Open **[`finetune_webslm_colab.ipynb`](finetune_webslm_colab.ipynb)** in Colab
(Runtime → T4 GPU). It clones this repo, installs deps, logs you into HF, trains on the domain
you pick, and **pushes a finished model to your HF account**. The last cell tells you exactly
what to enter in the webSLM Action.

Direct link (after you push the repo):
`https://colab.research.google.com/github/vishalmysore/webSLM/blob/main/finetune/finetune_webslm_colab.ipynb`

## Or run the script directly (any CUDA GPU)

```bash
pip install -r finetune/requirements.txt
python finetune/train_lora.py \
    --base Qwen/Qwen2.5-0.5B-Instruct \
    --data finetune/data/medical.jsonl \
    --push-merged yourname/WebSLM-Medical-0.5B \
    --epochs 3
# larger base on a 16 GB GPU? add  --bits 4   (QLoRA; uncomment bitsandbytes in requirements.txt)
```

Then: webSLM → **Actions** → **Build domain SLM** → Domain preset = **Custom**,
`custom_model_hf = yourname/WebSLM-Medical-0.5B`, `custom_arch = qwen2`, `custom_conv = qwen2`.

## The data

Sample starter sets live in [`data/`](data):

| File | Domain | Examples |
|---|---|---|
| `data/medical.jsonl` | general health info | ~12 |
| `data/legal.jsonl` | general legal concepts | ~12 |
| `data/insurance.jsonl` | insurance concepts | ~12 |

**These are tiny, illustrative samples — they prove the pipeline works, they do NOT specialize a
model.** For real domain quality, replace them with hundreds-to-thousands of your own examples.

**Format** — one JSON object per line, chat/conversational:

```json
{"messages":[
  {"role":"system","content":"You are a careful medical information assistant..."},
  {"role":"user","content":"What are common signs of dehydration?"},
  {"role":"assistant","content":"Common signs include increased thirst, dry mouth, ..."}
]}
```

Tips for building your dataset:
- One task per line; keep the `system` message consistent so the model learns a stable persona.
- The `assistant` turns are what the model imitates — make them the quality you want out.
- Train on `Qwen/Qwen2.5-0.5B-Instruct` (or another Instruct base); `arch=qwen2`, `conv=qwen2`
  flow straight through to the build step.

## Already have a model?

- **Full fine-tune** (no LoRA): you already have a standard HF checkpoint — skip this folder and
  point the build Action's Custom path straight at it.
- **A LoRA adapter** trained elsewhere: merge it with the repo-root
  [`merge_lora.py`](../merge_lora.py), then build the merged checkpoint.

## ⚠️ Responsible use

Medical, legal, and insurance outputs can cause real harm if wrong. These domains are provided as
*examples of the pipeline*. Keep a qualified human in the loop, include disclaimers (the sample
data does), and validate any model before relying on it.
