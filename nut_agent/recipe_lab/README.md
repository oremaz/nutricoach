# Recipe Lab

Recipe Lab is a tab inside NutriCoach. It predicts recipe difficulty, meal type
and preparation-time class locally, and optionally uses an OpenRouter model for
recipe extraction, per-serving nutrition estimates and explanation. It has no
separate journal or chat history.

## Follow one recipe

1. [app.py](app.py) accepts a recipe description or two recipes to compare.
2. [predictor.py](predictor.py) extracts/normalizes recipe text and combines the
   local predictions with optional LLM outputs.
3. [local_models.py](local_models.py) loads frozen text embeddings and cached
   LightGBM classifiers, fitting the heads on the recipe CSV when needed.
4. The UI presents the predicted classes and estimates. These are not logged
   automatically as nutrition intake.

The default encoder is `LiquidAI/LFM2.5-Embedding-350M`, pinned by commit in
`local_models.py`. Difficulty merges `A challenge` into `More effort`; meal type
merges lunch and dinner. Time classes use boundaries 15, 30 and 60 minutes.
Training CSV columns are `recipe_text`, `difficult`, `subcategory` and `times`.
Embeddings and classifier caches are keyed by dataset content, model identity
and cache version, under `nut_agent/secrets/recipe_lab_cache/`.

The encoder stays frozen; only supervised classifier heads are fitted. This
first-use fitting supports the application, rather than constituting a held-out
accuracy evaluation. Compatible older BGE/Jina artifact loading remains in the
predictor. Historical results from another encoder do not validate the current one.

## Run and inspect

From the NutriCoach repository root, after installing its dependencies:

```bash
PYTHONPATH=. uv run streamlit run nut_agent/nutricoach/app.py
PYTHONPATH=. uv run python -m pytest nut_agent/tests/test_predictor.py \
  nut_agent/tests/test_reasoning_settings.py -q
```

The first analysis needs the training CSV and an initial encoder download.
Without an API key, local classification and basic parsing remain available.
With OpenRouter, inferred ingredients/quantities affect both subsequent
classification and nutrition estimates. Per-serving calorie/nutrient estimates
are separate from classification accuracy.

The existing experiment entry points remain
[bench_recipe_methods.py](../bench_recipe_methods.py) and
[train_recipe_models.py](../train_recipe_models.py). Dataset and artifact path
overrides are documented in the [project README](../../README.md#recipe-experiments-and-artifact-paths).
See the [validation record](../../docs/cleanup_validation.md) for executed checks.
