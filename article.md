# webSLM: Building Domain-Specific Browser AI with SLMs, Quantized LLMs, WebLLM, and MLC-LLM

webSLM is a pipeline/toolkit — not a pre-trained model — for creating and deploying small language models (SLMs) that run entirely in the browser using WebGPU via WebLLM and MLC-LLM.

It makes it easy to take a compatible base model from Hugging Face, optionally fine-tune it for a domain, compile it into browser assets, and serve it as an on-device chatbot. The result is a private, offline-capable model that loads by URL with no server or API keys required.

## What webSLM does

### Core functionality

- Starts with a small base model such as Qwen2.5-0.5B / 1.5B, Phi-3.5-mini, or another MLC-LLM-compatible model.
- Optionally fine-tunes it on domain-specific data using the provided LoRA script and Colab notebook.
- Compiles and quantizes it with MLC-LLM for WebGPU.
- Produces browser-runnable artifacts: quantized weight shards and a `.wasm` model library.
- Hosts the outputs on Hugging Face, GitHub Releases, or another static host.
- Loads the model in a browser through a WebLLM config and demo page.

### Why this matters

- Domain-specific models can be far more useful than a generic chatbot for industry verticals such as insurance, legal, or medical.
- Running in the browser preserves privacy, removes backend costs, and enables offline usage.
- Small models are easier to build, tune, and deploy for WebGPU environments.

## What is an SLM?

SLM stands for Small Language Model.

Small language models are designed with far fewer parameters than traditional large LLMs. In this context, webSLM targets models in the 0.5B–2B range, and sometimes up to around 3–4B.

### Key characteristics of SLMs

- Natively small architecture — built or distilled to be compact.
- Efficient on low-resource hardware such as browsers, laptops, and edge devices.
- Often the best trade-off when privacy, latency, and cost are more important than extreme scale.
- Easier to fine-tune for specialized domains.

### Typical SLM examples

- Qwen2.5-0.5B / 1.5B
- Phi-3.5-mini
- Gemma and Mistral small variants

## What is a Quantized LLM?

Quantization is a compression technique applied to neural network weights. It reduces numerical precision from high-precision formats such as FP32 or FP16/BF16 into lower-precision formats like INT8, INT4, or even lower-bit representations.

### What quantization does

- Shrinks model size substantially.
- Reduces VRAM and memory demands.
- Improves inference speed on supported hardware.
- Makes larger models more practical to run in constrained environments.

### Common quantization formats

- INT8
- INT4
- Lower-bit schemes such as 3-bit or 2-bit
- Advanced methods: GPTQ, AWQ, GGUF

### Trade-offs

- Mild quantization often preserves most of the model's capability.
- Aggressive quantization can introduce quality degradation, shorter answers, and more repetition.
- The impact depends on the model, the quantization method, and the task.

## SLM vs Quantized LLM

| Aspect | SLM | Quantized LLM |
|---|---|---|
| Definition | Model designed to be small from the start | Large model compressed via reduced numeric precision |
| Origin | Trained/distilled as a small model | Post-training quantization of a larger model |
| Typical size | 0.5B–7B | 13B–405B+ before quantization |
| Efficiency | Efficient by design | Efficient only after quantization |
| Best use cases | Browser, mobile, privacy-first, offline | Local GPU inference with limited VRAM |
| Quality trade-off | Optimized for small size | Depends on bit precision and quant method |
| Can it be quantized? | Yes — often further quantized for browser use | Yes — but it starts large |

### What this means

- SLM is about model scale and architecture.
- Quantization is about numeric precision and compression.
- An SLM can be quantized further, but a quantized LLM usually begins as a much larger model.
- For browser-based private AI, SLMs are usually the stronger choice.

## What is WebLLM?

WebLLM is an open-source browser runtime by MLC-AI for running LLMs and SLMs with WebGPU.

### What WebLLM provides

- Client-side inference inside the browser.
- No backend server, no API keys, and no external inference endpoint.
- A JavaScript/TypeScript API that resembles OpenAI-style `chat.completions` usage.
- Support for streaming, JSON mode, and browser-friendly workflows.

### Best fit

WebLLM is ideal for small and quantized models that can run comfortably in a browser environment.

## What is MLC-LLM?

MLC-LLM is the compilation backend that prepares models for WebLLM and other device targets.

### What MLC-LLM does

- Converts model checkpoints into quantized weight shards.
- Generates a browser-compatible `.wasm` model library.
- Supports compilation for WebGPU, iOS, Android, and native backends.

### Why MLC-LLM matters for webSLM

MLC-LLM is the tool that turns a domain-specific SLM into a real browser model. Without it, the model cannot be executed efficiently in WebLLM.

## How webSLM works

### Workflow summary

1. Choose a compatible small base model.
2. Optionally fine-tune it with domain data using LoRA.
3. Compile and quantize the model with MLC-LLM.
4. Host the `.wasm` and quantized weights.
5. Load the model in a browser via WebLLM.

### Supported build paths

- GitHub Actions: automated pipeline for build and deployment.
- Colab: cloud-friendly fine-tuning and compilation notebooks.
- Local machine: run the pipeline from source with `build.sh`.

### What the repo includes

- `finetune/` — LoRA fine-tuning workflow, starter datasets, and training scripts.
- `colab/` — interactive notebooks for training and building.
- `demo/index.html` — ready-to-use browser demo.
- `build.sh` — local build script.
- `merge_lora.py` — merges LoRA adapters before compilation.
- `normalize_config.py` — fixes Hugging Face config issues for MLC-LLM v0.19.0.
- `model-card/` — template and publishing guidance for Hugging Face.
- `.github/workflows/build-slm.yml` — CI pipeline for building custom models.

## Domain-specific fine-tuning

The repository includes starter domain samples in `finetune/data/` for medical, legal, and insurance.

### Important note on training data

- The included samples are illustrative, not sufficient for production.
- Each starter dataset contains only a small number of examples.
- For real domain quality, replace them with hundreds or thousands of high-quality examples.
- Consistent prompts and assistant behavior are critical for reliable output.

### Why fine-tuning helps

- Small models adapt efficiently to domain-specific language and style.
- LoRA makes fine-tuning affordable and fast.
- The resulting model can answer specialized questions more accurately than a generic SLM.

## Practical considerations

### Strengths of webSLM

- Clean separation between training and compilation.
- Built for browser deployment, not just server inference.
- Automates complex MLC-LLM build steps.
- Provides a strong developer experience with notebooks, scripts, and demos.

### Limitations

- Browser models should remain small; 7B+ is generally too large.
- The repo pins MLC-LLM v0.19.0 for stability.
- Data quality matters more than quantity for sensitive domains.
- Some models require specific quant formats such as `q4f32_1` to avoid NaNs.

## Conclusion

webSLM is a practical, model-agnostic toolkit for creating domain-specific browser AI with WebLLM.

It bridges:

- small SLM architectures,
- quantized browser-ready assets,
- WebLLM runtime execution,
- and automated build workflows via GitHub Actions and Colab.

This makes it possible to build custom browser agents that are private, offline-capable, and tailored to real-world domains.
