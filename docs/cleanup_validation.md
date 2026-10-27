# Cleanup validation (2026-10-27)

The Streamlit entry point delegates photo analysis/review, registration and
journal tracking to separate UI modules. Existing app entry points, widget keys,
provider settings and data formats are preserved. Training scripts and job scripts
are unchanged.

## Checks in this checkout

- CPU/mock suite: **204 passed, 4 skipped**.
- The three GPU/live-provider modules listed below were excluded.
- The full NutriCoach landing page passed a Streamlit startup check.
- Photo review checks cover edits, explicit saving, reruns and method changes;
  provider checks cover graph rebuilding without changing the conversation thread.

Run from the NutriCoach root. This VM has no project-local environment; the shared
VM environment lacks an installed `ml_pipeline`, so validation included the sibling
source checkout explicitly, without installing or upgrading packages:

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=2
PYTHONPATH=.:../ml-playground UV_PROJECT_ENVIRONMENT=../../.venv \
  uv run --no-sync python -m pytest nut_agent/tests -q -p no:cacheprovider \
  --ignore=nut_agent/tests/test_gpu_pipeline.py \
  --ignore=nut_agent/tests/test_openrouter_live.py \
  --ignore=nut_agent/tests/test_local_qwen_live.py
```

Live APIs, GPU training and full FoodLMM/DPF inference were not rerun. Existing
Food Vision benchmarks retain their original source and configuration receipts;
UI startup and mock tests do not revalidate their accuracy.

## Component-guide follow-up

The Food Vision overview now links to separate method and FoodLMM setup guides;
Recipe Lab has its own data, task and inference map. Existing commands and result
values were retained. The same CPU/mock command above was repeated against the
updated sibling `ml_pipeline` source: **204 passed, 4 skipped**, with the same
three GPU/live-provider module exclusions. This does not remeasure Food Vision
or Recipe Lab model accuracy.

The request-audit follow-up added an offline photo-review example to the project
README. The exact snippet executed successfully: increasing the synthetic rice
quantity from 100 g to 150 g changed 130 kcal to 195 kcal without a model call
or journal write. This was a documentation-only follow-up.
