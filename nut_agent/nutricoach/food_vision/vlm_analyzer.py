"""
Single-shot vision-language baseline via OpenRouter, Groq or local Qwen.

One image prompt produces foods, estimated grams, and calorie/macro values.
No local nutrition database is queried. Hosted inference requires the selected
provider's API key; local inference uses cached Transformers weights.
"""

import json
import logging
import os
import time
from typing import Optional

from shared.config import OPENROUTER_REASONING_EFFORT, OPENROUTER_MAX_OUTPUT_TOKENS
from shared.llm_provider import create_client, default_model, request_settings, LOCAL_MODEL_REVISION

from .base import (
    FoodAnalyzer,
    FoodAnalysisResult,
    FoodItem,
    encode_image_to_base64,
    get_image_media_type,
)

logger = logging.getLogger(__name__)


class VLMAnalyzerSingleShot(FoodAnalyzer):
    """Identify foods and estimate portions and nutrients in one image API call."""

    method_name = "vlm_single"

    SINGLE_PROMPT = """You are an expert nutritionist. Analyze this meal photo and provide a complete nutritional breakdown.

For each food item visible:
1. Identify the food (be specific about preparation method)
2. Estimate the portion size in grams
3. Calculate calories, protein (g), carbs (g), and fat (g)

Return ONLY a JSON object:
{
  "items": [
    {
      "name": "food name",
      "quantity_grams": 180,
      "portion_description": "1 medium serving",
      "confidence": 0.8,
      "calories": 250,
      "protein_g": 30.0,
      "carbs_g": 5.0,
      "fat_g": 12.0
    }
  ]
}

Be thorough — include sauces, condiments, garnishes, drinks. Use standard nutrition values."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        provider: str = "openrouter",
        device: Optional[str] = None,
        quantization: str = "none",
        revision: str = LOCAL_MODEL_REVISION,
        max_output_tokens: int = OPENROUTER_MAX_OUTPUT_TOKENS,
    ):
        self.provider = provider
        self.api_key = api_key or os.environ.get(f"{provider.upper()}_API_KEY", "")
        self.model = model or default_model(provider)
        self.reasoning_effort = reasoning_effort if reasoning_effort is not None else (
            "none" if provider == "local" else OPENROUTER_REASONING_EFFORT)
        self.device, self.quantization, self.revision = device, quantization, revision
        self.max_output_tokens = max_output_tokens
        self._client = None

    def _get_client(self):
        if self._client is None:
            self._client = create_client(self.provider, self.api_key, model=self.model,
                                         device=self.device, quantization=self.quantization, revision=self.revision)
        return self._client

    def _request_options(self):
        token_field = "max_completion_tokens" if self.provider == "groq" else "max_tokens"
        return {token_field: self.max_output_tokens, "temperature": 0.1,
                **request_settings(self.provider, self.reasoning_effort)}

    def _messages(self, image_path):
        b64 = encode_image_to_base64(image_path)
        messages = ([{"role": "system", "content": self.SYSTEM_PROMPT}]
                    if getattr(self, "SYSTEM_PROMPT", None) else [])
        messages.append({"role": "user", "content": [
            {"type": "text", "text": getattr(self, "USER_PROMPT", self.SINGLE_PROMPT)},
            {"type": "image_url", "image_url": {"url": f"data:{get_image_media_type(image_path)};base64,{b64}"}},
        ]})
        return messages

    def _parse_response(self, response):
        result = FoodAnalysisResult(method=self.method_name)
        raw = (response.choices[0].message.content or "").strip()
        result.raw_response = raw
        parsed = self._parse_json(raw, dict)
        if parsed and isinstance(parsed.get("items"), list) and parsed["items"]:
            if not all(isinstance(item, dict) for item in parsed["items"]):
                result.error = "Food items must be JSON objects"
            else:
                result.food_items = [FoodItem(
                    name=item.get("name", "unknown"), quantity_grams=item.get("quantity_grams", 100),
                    confidence=item.get("confidence", 0.5), calories=item.get("calories", 0),
                    protein_g=item.get("protein_g", 0), carbs_g=item.get("carbs_g", 0),
                    fat_g=item.get("fat_g", 0), portion_description=item.get("portion_description", ""))
                    for item in parsed["items"]]
                result.compute_totals()
        else:
            result.error = "Could not parse response"
        return result

    def analyze(self, image_path: str) -> FoodAnalysisResult:
        start = time.time()
        result = FoodAnalysisResult(method=self.method_name)

        try:
            if self.provider != "local" and not self.api_key:
                raise ValueError(f"{self.provider.upper()}_API_KEY not set.")

            client = self._get_client()
            response = client.chat.completions.create(
                model=self.model,
                messages=self._messages(image_path),
                **self._request_options(),
            )
            result = self._parse_response(response)

        except Exception as e:
            result.error = str(e)

        result.elapsed_seconds = time.time() - start
        return result

    def _parse_json(self, text: str, expected_type: type):
        """Extract and parse JSON from LLM response, handling markdown fences."""
        # Strip markdown code fences
        cleaned = text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            # Remove first and last fence lines
            lines = [l for l in lines if not l.strip().startswith("```")]
            cleaned = "\n".join(lines)

        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, expected_type):
                return parsed
        except json.JSONDecodeError:
            pass

        # Try to find JSON array/object in the text
        for start_char, end_char in [("[", "]"), ("{", "}")]:
            start_idx = cleaned.find(start_char)
            end_idx = cleaned.rfind(end_char)
            if start_idx != -1 and end_idx > start_idx:
                try:
                    parsed = json.loads(cleaned[start_idx:end_idx + 1])
                    if isinstance(parsed, expected_type):
                        return parsed
                except json.JSONDecodeError:
                    continue

        logger.warning("Could not parse JSON from VLM response: %.100s...", text)
        return None


class VLMAnalyzerSingleShotGuided(VLMAnalyzerSingleShot):
    """Single-image VLM with a portion-first nutrition estimation procedure."""

    method_name = "vlm_single_guided"
    USER_PROMPT = "Analyze the attached meal photo and return its nutritional breakdown as JSON."
    SYSTEM_PROMPT = """Estimate the nutrition in the image for one person's meal. Follow these steps internally:
1. Treat the whole visible serving as one person's meal.
2. Identify each distinct food and visible component, including sauces, toppings, drinks, and cooking fats when apparent.
3. Estimate the served mass in grams for each component, using a plausible single-person portion. Do not assume hidden ingredients that are not visually supported.
4. Estimate calories, protein, carbohydrates, and fat per 100 g for each identified food using typical values for its preparation. Multiply each per-100-g estimate by its estimated mass divided by 100 to obtain that item's totals.
5. Sum item masses and nutrients for the complete meal. Keep estimates internally consistent and avoid false precision.

Use the following text examples as calibration for portion size and typical nutrient values. They are illustrative, not a lookup table:

Example 1: A person's meal of cooked white rice, about 180 g. Cooked rice is about 130 kcal, 2.7 g protein, 28.2 g carbohydrates, and 0.3 g fat per 100 g. Estimate about 234 kcal, 4.9 g protein, 50.8 g carbohydrates, and 0.5 g fat.
Example 2: A person's meal of grilled chicken breast, about 150 g. Cooked chicken breast is about 165 kcal, 31 g protein, 0 g carbohydrates, and 3.6 g fat per 100 g. Estimate about 248 kcal, 46.5 g protein, 0 g carbohydrates, and 5.4 g fat.
Example 3: A person's meal of steamed broccoli, about 120 g. Cooked broccoli is about 35 kcal, 2.4 g protein, 7.2 g carbohydrates, and 0.4 g fat per 100 g. Estimate about 42 kcal, 2.9 g protein, 8.6 g carbohydrates, and 0.5 g fat.
Example 4: A person's meal of plain mashed potatoes, about 200 g. Mashed potatoes prepared with milk and a little butter are about 110 kcal, 2 g protein, 17 g carbohydrates, and 4 g fat per 100 g. Estimate about 220 kcal, 4 g protein, 34 g carbohydrates, and 8 g fat.
Example 5: A person's meal of mixed green salad with tomato and a light vinaigrette, about 180 g total: 130 g vegetables at about 25 kcal, 1.5 g protein, 5 g carbohydrates, and 0.2 g fat per 100 g, plus 50 g vinaigrette at about 300 kcal, 0.5 g protein, 4 g carbohydrates, and 32 g fat per 100 g. Estimate about 183 kcal, 2.2 g protein, 8.5 g carbohydrates, and 16.3 g fat total.

Apply the same process to the photographed meal. Return ONLY a JSON object with an `items` list. For each visible component include `name`, `quantity_grams`, `portion_description`, `confidence`, `calories`, `protein_g`, `carbs_g`, and `fat_g`. Nutrient fields are totals for that component, not per-100-g values. Do not include the examples or your intermediate calculations in the output. Include plausible zero values where appropriate; do not omit uncertain visible components."""

    def __init__(self, *args, provider="openrouter", **kwargs):
        kwargs["reasoning_effort"] = "xhigh" if provider == "local" else "high"
        super().__init__(*args, provider=provider, **kwargs)


VLMAnalyzer = VLMAnalyzerSingleShot
