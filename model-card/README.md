---
license: apache-2.0
language:
- en
base_model:
- Qwen/Qwen2.5-Coder-1.5B-Instruct
pipeline_tag: text-generation
library_name: mlc-llm
tags:
- webllm
- webgpu
- mlc-llm
- browser
- on-device
- small-language-model
- qwen2
---

# WebSLM-Code-1.5B-MLC

> Template model card — replace the name, base model, domain, and URLs with yours.

A **WebGPU / WebLLM** build of a domain-specific Small Language Model, compiled from
source with [MLC-LLM](https://github.com/mlc-ai/mlc-llm) (quantization `q4f16_1`). It runs
**entirely in the browser** — no server, no API key, fully on-device and private.

> Build pipeline & sources: **https://github.com/yourname/webSLM**

## What this is

An ordinary chat SLM, made fast/private/offline in the browser. This particular build is a
**code** assistant ([Qwen2.5-Coder-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct));
the same pipeline produces medical, legal, math, or any-domain SLMs from a domain base model
or your own fine-tune. There is **no latent loop / no exposed hidden states** — it's a plain
`input_ids → logits` chat model.

## Files

| File | What |
|------|------|
| `params_shard_*.bin`, `ndarray-cache.json` | `q4f16_1` quantized weights |
| `mlc-chat-config.json` | MLC chat config (template, tokenizer metadata) |
| `tokenizer.json`, `tokenizer_config.json`, ... | tokenizer |
| `libs/WebSLM-Code-1.5B-q4f16_1-webgpu.wasm` | the WebGPU model library |

## Usage (WebLLM, in the browser)

```js
import * as webllm from "@mlc-ai/web-llm";  // pin to 0.2.79 (matches mlc-llm v0.19.0)

const appConfig = {
  model_list: [{
    model:     "https://huggingface.co/yourname/WebSLM-Code-1.5B-MLC",
    model_id:  "webslm-code-1.5b",
    model_lib: "https://huggingface.co/yourname/WebSLM-Code-1.5B-MLC/resolve/main/libs/WebSLM-Code-1.5B-q4f16_1-webgpu.wasm",
  }],
};
const engine = await webllm.CreateMLCEngine("webslm-code-1.5b", { appConfig });
const r = await engine.chat.completions.create({
  messages: [{ role: "user", content: "Write a binary search in Rust." }],
});
console.log(r.choices[0].message.content);
```

## How it was built

`mlc_llm convert_weight` → `gen_config` → `compile --device webgpu`, built **from source**
against `mlc-llm` **v0.19.0** (the last release before the `apache-tvm-ffi` migration), TVM
compiled with LLVM. The full reproducible pipeline (CI workflow + Colab notebook + LoRA-merge
helper for fine-tunes) is in the [webSLM](https://github.com/yourname/webSLM) repo.

## ⚠️ Version compatibility

The `.wasm` model library is tied to the runtime version. Load it with a **compatible
`@mlc-ai/web-llm`** build — **0.2.79** matches mlc-llm v0.19.0. A mismatch raises a
“model lib version” error; pin both, or recompile the `.wasm` against your runtime's version.

## License

Inherits the base model's license (Apache-2.0 for the Qwen2.5 example above — check yours).
Research/educational artifact.
