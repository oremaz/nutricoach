# FoodLMM setup and packed checkpoints

See the [benchmark overview](README.md) for the evaluation protocol and
[method notes](METHODS.md#foodlmm-foodlmm-foodlmm_cached-foodlmm_4bit) for the inference variants.

FoodLMM runs in an isolated Python 3.10 environment because its Transformers
version conflicts with the main app. Its worker adds the environment’s executable
directory to PATH so PyTorch can find Ninja for native extensions. From the repository root:

```bash
uv venv --python 3.10 nut_agent/nutricoach/food_vision/data/foodlmm_env
uv pip install --python nut_agent/nutricoach/food_vision/data/foodlmm_env/bin/python \
    -r requirements_foodlmm.in
```

On CUDA, install matching CUDA Torch/Torchvision wheels in that environment.
The app finds it automatically; `FOODLMM_PYTHON` selects another interpreter
and `FOODLMM_DEVICE` selects `auto`, `cuda`, `mps` or `cpu`. Auto prefers CUDA,
then MPS, then CPU. Git and Hugging Face access are needed for initial assets.

### FoodLMM 4-bit and Apple Silicon

The app loads an existing packed export from `data/foodlmm_4bit`, or from
`FOODLMM_QUANTIZED_CHECKPOINT`. Without one, it loads and quantizes the original
weights, requiring more loading memory, then automatically saves a packed export
to `data/foodlmm_4bit` for future starts. The terminal reports loading, export,
and model readiness. Both export files must be present to reuse the checkpoint.
FoodLMM prediction uses `torch.no_grad()` to preserve the tensor version counters
required by Quanto. On MPS, the runner bypasses Quanto 0.2.7's native unpacker
using tensor operations with the same packed value ordering. The native unpacker
recompiles Metal shaders and synchronizes the GPU on every call. A small
quantized MPS linear layer produced matching outputs with the replacement;
full-model MPS latency remains unvalidated. Generation stops after all five
whole-dish nutrition tokens appear, or at EOS/the 256-token limit. The full-prefix
nutrition-head pass is retained; causal states preceding the required tokens do
not depend on discarded trailing text. Generation and nutrition-head timings
are reported separately in the terminal. Existing benchmark reports predate this
early stopping change and have not been rerun. Prepare a packed export on a machine
with sufficient memory and copy it to the target machine:

```bash
PYTHONPATH=. nut_agent/nutricoach/food_vision/data/foodlmm_env/bin/python \
    nut_agent/nutricoach/food_vision/evaluate_paper_system.py \
    --method foodlmm --device cuda --quantization 4bit --prepare-only \
    --export-quantized nut_agent/nutricoach/food_vision/data/foodlmm_4bit \
    --output-dir /path/to/foodlmm-prepare
PYTHONPATH=. nut_agent/nutricoach/food_vision/data/foodlmm_env/bin/python \
    nut_agent/nutricoach/food_vision/evaluate_paper_system.py \
    --method foodlmm --device auto --quantization 4bit \
    --quantized-checkpoint nut_agent/nutricoach/food_vision/data/foodlmm_4bit \
    --output-dir /path/to/foodlmm-run
```

The benchmark runner also accepts `--quantization none` for the 16-bit control.
The app prepares its CLIP assets automatically; the benchmark needs them cached
first with `prepare_benchmark_assets.py --papers`. On macOS, Quanto's MPS kernels
require Xcode command line tools. CUDA memory/latency results do not establish
Apple Silicon performance; real FoodLMM MPS inference remains unvalidated here.
