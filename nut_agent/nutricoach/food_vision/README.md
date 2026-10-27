# Food Vision

Estimate whole-dish calories, mass, fat, carbohydrate and protein from one RGB
photo. The benchmark uses a frozen Nutrition5k test subset. NutriCoach reuses a
subset of the methods for editable meal estimates and explicit journal logging.
Benchmark accuracy and app usability are separate questions.

## Start here

| Goal | Read or run |
|---|---|
| Inspect measured performance | [500-dish results and preliminary guided comparison](nutrition5k_500_qwen_results.md) |
| Understand each model and its training | [Method notes](METHODS.md) |
| Follow the shared output contract | [base.py](base.py), then [supervised_analyzers.py](supervised_analyzers.py) |
| Review the evaluation protocol | [evaluate_nutrition5k.py](evaluate_nutrition5k.py), then the interpretation section below |
| Run FoodLMM | [Isolated environment and packed checkpoints](FOODLMM.md) |
| Trace edits into the journal | [review.py](review.py), then [photo_ui.py](../photo_ui.py) |

## Methods at a glance

| Method ID | Prediction path | Entry point |
|---|---|---|
| `vlm_single` | One VLM request, food-level estimates summed to dish totals | [vlm_analyzer.py](vlm_analyzer.py) |
| `vlm_single_guided` | Same interface, with portion-first prompt and higher reasoning | [vlm_analyzer.py](vlm_analyzer.py) |
| `rgb_regression` | Frozen visual features and a locally fitted ridge head | [train_rgb_regression.py](train_rgb_regression.py) |
| `food_r1` | Released specialized VLM, parsed into the five benchmark targets | [supervised_analyzers.py](supervised_analyzers.py) |
| `foodlmm`, `foodlmm_cached`, `foodlmm_4bit` | Released FoodLMM checkpoint, with distinct execution/precision variants | [evaluate_paper_system.py](evaluate_paper_system.py) |
| `dpf_reproduction` | Released third-party RGB and predicted-depth regressor | [dpf_analyzer.py](dpf_analyzer.py) |

The benchmark's five targets are kcal and grams. The app's `FoodAnalysisResult`
stores nutrient totals and food-item quantities; benchmark adapters map these
to the five-target representation. The three FoodLMM rows use the same learned
checkpoint. Guided VLM changes both the prompt and reasoning effort, so its
comparison is not a prompt-only ablation. See the method notes for details.

## Results

[The 500-dish Qwen comparison](nutrition5k_500_qwen_results.md) includes the local
Qwen single-shot baseline and guided variant, RGB regression, Food-R1, FoodLMM
precision variants and DPF. The report generator writes
`nutrition5k_500_qwen_results.json` with predictions, failures and configurations.

## Evaluation and interpretation

The frozen subset contains **500 dishes**, selected with seed 42 from the
intersection of Nutrition5k's official RGB and depth test IDs. Each method
receives one overhead RGB image; measured depth, true mass and ingredient
annotations are withheld. Source and image hashes identify the shared inputs.

| Report field | Meaning |
|---|---|
| MAE | Mean absolute error; kcal for energy, grams for mass and macros |
| Normalized MAE | MAE divided by the mean reference value for that target |
| RMSE / R² | Additional measures of error magnitude and explained variance |
| Bootstrap interval | Sampling variability across dishes, not uncertainty for an individual photo |

Errors use successful predictions only, so rows with different success counts
can describe different dish subsets. Paired comparisons use common successes.
The optional oracle-mass diagnostic rescales nutrients using true mass **during
scoring only**; it is not deployable RGB-only performance.

The locally fitted RGB head has recorded training/validation IDs. External model
training is described by its publishers: we cannot independently verify the
complete training membership of the released weights or API models. In
particular, the DPF publisher reports correcting an earlier split leakage issue,
but its checkpoint does not contain the fitted dish IDs needed to audit that fix.
Training on Nutrition5k and evaluating its held-out test split is legitimate;
using the same dataset name alone does not imply leakage.

A single photo leaves scale, hidden ingredients and added oil ambiguous.
Results on controlled cafeteria images should not be treated as evidence of
reliable portions on web photos. These adapters and inputs also differ from
some paper settings; published paper scores are not substituted for our runs.

## Run the benchmark

From the repository root, use the main `uv` environment for RGB, VLM, Food-R1
and DPF. FoodLMM requires its separate environment below. Models, releases and
datasets need internet access on first download; `HF_HOME` selects the model cache.

```bash
PYTHONPATH=. uv run python nut_agent/nutricoach/food_vision/evaluate_nutrition5k.py --prepare-only
PYTHONPATH=. uv run python nut_agent/nutricoach/food_vision/train_rgb_regression.py --device cuda
PYTHONPATH=. uv run python nut_agent/nutricoach/food_vision/evaluate_nutrition5k.py \
    --methods rgb_regression food_r1 --device cuda --output-dir /path/to/local-run
PYTHONPATH=. uv run python nut_agent/nutricoach/food_vision/evaluate_nutrition5k.py \
    --methods vlm_single --provider local --model Qwen/Qwen3.8-27B \
    --reasoning-effort none --workers 1 --device cuda --output-dir /path/to/qwen-run
PYTHONPATH=. uv run python nut_agent/nutricoach/food_vision/evaluate_nutrition5k.py \
    --methods vlm_single_guided --provider local --model Qwen/Qwen3.8-27B \
    --reasoning-effort none --workers 1 --device cuda --output-dir /path/to/qwen-guided-run
PYTHONPATH=. uv run python nut_agent/nutricoach/food_vision/evaluate_paper_system.py \
    --method dpf_reproduction --device auto --output-dir /path/to/dpf-run
PYTHONPATH=. uv run python nut_agent/nutricoach/food_vision/report_nutrition5k.py \
    --model Qwen/Qwen3.8-27B \
    --run-dir /path/to/qwen-run /path/to/local-run /path/to/dpf-run /path/to/foodlmm-run
```

For hosted VLM inference, choose `--provider openrouter` or `groq`, an
image-capable `--model`, and configure the corresponding API key. Local Qwen
BF16 needs roughly 58 GB for weights alone; `--local-quantization 4bit` uses
bitsandbytes on CUDA. Food-R1's checkpoint is roughly 17.5 GB before runtime
memory. These memory requirements are separate from FoodLMM's quantization.

Runs save predictions incrementally and resume completed dishes; `--retry-errors`
retries failures. Use a new output directory when code or configuration changes.
For offline GPU jobs, prepare assets with [prepare_benchmark_assets.py](prepare_benchmark_assets.py)
and submit [nutrition5k_jean_zay.sbatch](jobs/nutrition5k_jean_zay.sbatch) using
an account and GPU allocation appropriate to your site.
The guided prompt's current partial comparison has 47 paired predictions; its
calories, mass and fat MAEs are recorded in
[the Qwen results report](nutrition5k_500_qwen_results.md). The remaining
benchmark run was canceled, so these values are preliminary.

## Food vision methods in NutriCoach

NutriCoach's Food Analysis tab offers a one-shot VLM, FoodLMM 4-bit and DPF reproduction.
The VLM uses the provider and model selected in the sidebar, always with high reasoning;
its guided portion-first prompt is on by default and can be disabled. The results table
shows each detected food in its first column. FoodLMM and DPF run locally, independently
of the chat provider. DPF uses the existing RGB and predicted-depth
adapter in the main environment, downloads its assets on first use, and reuses
its loaded model. Switching methods clears the previous result.
The adapters return whole-dish totals; users can review,
correct and explicitly save them to the nutrition journal.

## FoodLMM in NutriCoach

FoodLMM needs a separate Python 3.10 environment because its Transformers version
conflicts with the main application. Follow [FOODLMM.md](FOODLMM.md) for the exact
installation commands, interpreter overrides, packed 4-bit export and Apple
Silicon limitations. These commands and artifact paths are preserved.

## Focused checks

From the NutriCoach repository root:

```bash
PYTHONPATH=. uv run python -m pytest nut_agent/tests/test_nutrition5k_eval.py \
  nut_agent/tests/test_supervised_food_vision.py nut_agent/tests/test_photo_review.py -q
```

These use fixtures and mocks. They do not reproduce real-model benchmark scores.
The [validation record](../../../docs/cleanup_validation.md) records the checks
and environments used during the cleanup.
