# NutriCoach & Recipe Lab

A Streamlit application exploring how LLM agents, local classifiers, and
vision models can support nutrition tracking and recipe analysis. NutriCoach
keeps a user profile and journal; its Recipe Lab tab analyzes recipes within the
same user session.
NutriCoach and its Recipe Lab tab are experimental; their nutrition outputs are estimates.

## NutriCoach

After registering and entering a profile, users can chat about their nutrition
goals, ask for meal plans, and record meals and water intake. The app calculates
calorie and macro targets from profile inputs, tracks intake against those
targets, and displays logged nutrition and weight trends. Daily tracking allows
meals and weight measurements to be recorded on separate dates.

The assistant uses an OpenRouter LLM in a LangGraph tool-calling loop. Tools
handle calculations, food lookups, profile updates, logging, and saved meal plans.
Structured profiles and journals are stored as per-user JSON files, while
LangGraph's SQLite checkpointer persists conversation state. Each LLM call uses
structured user context and a bounded recent conversation history.

Users can also upload a food photo to estimate meal mass, calories, and
macros using **One-shot VLM** (`vlm_single`, default), **FoodLMM (4-bit)** (`foodlmm_4bit`), or **DPF (reproduction)**
(`dpf_reproduction`), selected in the Food Analysis tab. One-shot VLM uses the sidebar provider and model, with high reasoning and the guided nutrition prompt enabled by default. FoodLMM and DPF run locally without an
API key and estimate whole-dish mass and nutrients. The app adapter returns
whole-dish totals. The first analysis downloads the pinned release
and model weights, then saves a packed 4-bit checkpoint for future starts;
subsequent photos reuse the loaded model. DPF uses the existing RGB and predicted-depth
research adapter in the main environment and caches its loaded model. Switching
methods clears the previous photo result and reruns the page so the photo UI
stays inside its Food Analysis tab. Automatic device
selection uses CUDA, then Apple Silicon MPS, then CPU. FoodLMM uses a persistent
worker in its isolated environment; see [FoodLMM setup](nut_agent/nutricoach/food_vision/README.md#foodlmm-in-nutricoach).
Users can edit the dish name and quantities, add or remove ingredients, and review
recalculated totals before explicitly saving the meal. New or renamed ingredients need a
database match or manually entered per-100g values.

The [Food Vision README](nut_agent/nutricoach/food_vision/README.md) describes each benchmark
method, its training data, the evaluation protocol and the result reports.

## Recipe Lab

Enter a recipe description to analyze it, or compare two recipes side by side.
Recipe Lab is available only as a tab inside NutriCoach.
It does not maintain a nutrition journal or conversation history.

Classification runs locally using frozen
`LiquidAI/LFM2.5-Embedding-350M` text embeddings and LightGBM classifiers:

- Difficulty: `Easy` or `More effort`, with `A challenge` merged into the latter.
- Meal type: breakfast or lunch/dinner.
- Total preparation and cooking time: `<15`, `15-30`, `30-60`, or `>60 min`.

On first analysis, the app loads the embedding model, embeds
`nut_agent/recipes_df.csv`, and trains the classifiers. This can take several
minutes and requires the dataset and an initial model download. Embeddings and
classifiers are cached under `nut_agent/secrets/recipe_lab_cache/`, keyed by
dataset content and the pinned model revision. Compatible existing BGE or Jina
checkpoints can serve as a fallback if the LFM2.5 path is unavailable.

With an OpenRouter key, the app also extracts recipe structure, estimates
per-serving calories and seven nutrients in zero shot, and explains the results.
The LLM may infer ingredients, quantities, or steps omitted from the description,
so these assumptions affect both classification and nutrition estimates.
Without a key, local classification remains available, with basic text parsing.

Recipe benchmark code lives in
[`nut_agent/bench_recipe_methods.py`](nut_agent/bench_recipe_methods.py).
Historical BGE results do not validate the current LFM2.5 classifiers, and recipe
classification scores do not measure nutrition estimation accuracy.

## Running the app

Install the standalone application's dependencies with `uv venv` followed by `uv pip install -r requirements.txt`. The repository uses its own `uv` environment and root dependency files
(`requirements.txt`, or `requirements-mac.txt` on macOS). On macOS, LightGBM
also requires the OpenMP runtime: `brew install libomp`.

Run from the repository root:

```bash
# NutriCoach, including the Recipe Lab tab
PYTHONPATH=. uv run streamlit run nut_agent/nutricoach/app.py
```

The sidebar's **Model provider** selects **OpenRouter**, **Groq**, or
**Local (Qwen)** for NutriCoach chat. OpenRouter is the default, using
`openrouter/free`.
Set `OPENROUTER_API_KEY` or `GROQ_API_KEY`, or enter the selected provider's key
in the sidebar. When `OPENROUTER_API_KEY` is set, NutriCoach and its Recipe Lab tab
use it directly and show a disabled `***` field instead of asking
for the key. Keys are separate. Local inference needs no API key and runs
`Qwen/Qwen3.8-27B` through Transformers in the Streamlit process.
The chat model ID remains editable. `OPENROUTER_MODEL_ID`,
`GROQ_MODEL_ID`, and `LOCAL_MODEL_ID` override provider defaults.
Provider switches reset the chat model ID and reasoning settings to compatible
defaults and rebuild the chat agent.
Reasoning defaults to `high` for hosted providers and `none` locally;
the menu exposes each backend's supported levels. Local generation is greedy.
Responses have an 8,192-token budget. Hosted chat goes to the selected
provider. Photo analysis follows the selected method: one-shot VLM uses the sidebar provider and model, while FoodLMM and DPF run locally.
Local **BF16** needs roughly 58 GB just for weights plus working memory;
**4-bit (CUDA)** needs `bitsandbytes` from the root GPU dependencies and reduces
GPU memory use. BF16 is the default and the precision used for the H100 benchmark;
4-bit is an optional memory-saving setting whose accuracy needs separate evaluation.
Loading can also require substantial host RAM; GPU allocation alone does not
establish compatibility with a machine having 32 GB of total memory.
`LOCAL_LLM_DEVICE` can select an explicit local device.
See [Food Vision setup and benchmark commands](nut_agent/nutricoach/food_vision/README.md).
Recipe Lab retains its existing OpenRouter configuration.
The environment variable must be available in the terminal or service that
launches Streamlit. When it is absent, the entered sidebar key is used.
Local user data and Recipe Lab caches live under `nut_agent/secrets/`, which is
excluded from version control. Model release code is cached in Food Vision's
gitignored `data/paper_releases/`; its weights use the Hugging Face cache.

## Code and tests

`nutricoach/` contains the agent, tools, UI, and food vision methods;
`recipe_lab/` contains the recipe UI and prediction pipeline. `shared/` holds
configuration, authentication, Pydantic schemas, and structured memory.
Recipe classifiers and supervised food-vision models reuse the installed `ml_pipeline` package from [ML Playground](https://github.com/oremaz/ml-playground), pinned to a specific Git commit in the dependency specifications.

Run the tests that do not require live APIs or GPU integration:

```bash
PYTHONPATH=. uv run python -m pytest nut_agent/tests/ -q \
    --ignore=nut_agent/tests/test_gpu_pipeline.py \
    --ignore=nut_agent/tests/test_openrouter_live.py \
    --ignore=nut_agent/tests/test_local_qwen_live.py
```

The separate GPU and live OpenRouter test modules exercise model inference and
API integration when their required resources are available.

## Recipe experiments and artifact paths

Recipe datasets and domain-specific training scripts belong to this repository. Run from its root:

```bash
PYTHONPATH=. uv run python nut_agent/bench_recipe_methods.py
PYTHONPATH=. uv run python nut_agent/train_recipe_models.py
```

The default recipe dataset is `nut_agent/recipes_df.csv`; set `NUTRICOACH_RECIPE_DATA` to use another training CSV. The training scripts resolve the test CSVs beside that file. Recipe checkpoints and metadata default to `results/` at the repository root; set `NUTRICOACH_MODELS_DIR` to override them.

Food-vision datasets, pinned model releases, and packed checkpoints retain their paths under `nut_agent/nutricoach/food_vision/data/` and remain outside Git. Recreate the isolated FoodLMM environment using the Food Vision README, or set `FOODLMM_PYTHON` to an existing compatible interpreter. Virtual environments, caches, secrets, user journals, and logs are not copied into public repositories.

The one-shot VLM photo analyzer keeps its guided nutrition prompt enabled by default. Existing benchmark reports describe their original runs; moving files does not revalidate model accuracy.

When running CPU checks on a shared VM, set `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=2` to keep test workers within the allocated resources.

## Split validation (2026-10-14)

The CPU and mock-based suite passed: **204 passed, 4 skipped**. The GPU recipe, live OpenRouter, and live local-Qwen test modules were excluded. The application used the installed `ml_pipeline` wheel in a separate validation interpreter, with its declared NumPy, SciPy, scikit-learn, pandas, LightGBM, Streamlit, LangGraph, LangChain, and OpenAI versions installed.

Three failures were reproduced in the original checkout before updating the copied tests: photo-method expectations now include the existing one-shot VLM, Recipe Lab uses the shared reasoning setting, and the DPF checkpoint test mocks model creation without a download. These test changes preserve application behavior. `langgraph-prebuilt` is pinned to the original compatible version to prevent an incompatible newer dependency from being selected.

This migration does not validate live providers, GPU training, or full real-model food-photo inference. The copied FoodLMM checkpoint and Nutrition5k datasets remain local and ignored by Git.

The full NutriCoach Streamlit entry point passed a startup check without logging in or loading large models. Default recipe dataset and checkpoint paths were also checked.

The shared framework also built and installed successfully directly from the pinned public Git commit. The applications do not require a sibling `ml-playground` checkout.
