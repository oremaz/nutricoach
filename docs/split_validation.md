## Split validation (2026-10-14)

The CPU and mock-based suite passed: **204 passed, 4 skipped**. The GPU recipe, live OpenRouter, and live local-Qwen test modules were excluded. The application used the installed `ml_pipeline` wheel in a separate validation interpreter, with its declared NumPy, SciPy, scikit-learn, pandas, LightGBM, Streamlit, LangGraph, LangChain, and OpenAI versions installed.

Three failures were reproduced in the original checkout before updating the copied tests: photo-method expectations now include the existing one-shot VLM, Recipe Lab uses the shared reasoning setting, and the DPF checkpoint test mocks model creation without a download. These test changes preserve application behavior. `langgraph-prebuilt` is pinned to the original compatible version to prevent an incompatible newer dependency from being selected.

This migration does not validate live providers, GPU training, or full real-model food-photo inference. The copied FoodLMM checkpoint and Nutrition5k datasets remain local and ignored by Git.

The full NutriCoach Streamlit entry point passed a startup check without logging in or loading large models. Default recipe dataset and checkpoint paths were also checked.

The shared framework also built and installed successfully directly from the pinned public Git commit. The applications do not require a sibling `ml-playground` checkout.
