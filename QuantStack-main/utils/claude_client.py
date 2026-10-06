import os
from typing import Optional


class ClaudeClient:
    """Thin wrapper around Anthropic's Claude API for market-analysis summaries."""

    DEFAULT_MODEL = "claude-sonnet-4-20250514"
    DEFAULT_SYSTEM_PROMPT = (
        "You are a disciplined market-research assistant. "
        "Provide concise, evidence-based explanations and avoid making investment guarantees."
    )

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.model = model or os.getenv("ANTHROPIC_MODEL", self.DEFAULT_MODEL)
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def _build_client(self):
        if not self.api_key:
            raise RuntimeError(
                "Anthropic API key is not configured. Set the ANTHROPIC_API_KEY environment variable."
            )

        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise RuntimeError(
                "The 'anthropic' package is required. Install it with: pip install anthropic"
            ) from exc

        return Anthropic(api_key=self.api_key)

    def generate(self, prompt: str, *, system_prompt: Optional[str] = None, max_tokens: int = 900) -> str:
        if not prompt or not prompt.strip():
            return ""

        client = self._build_client()
        system = system_prompt or self.system_prompt

        response = client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=0.2,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )

        if response and hasattr(response, "content"):
            content = response.content
            if isinstance(content, list):
                text_parts = []
                for item in content:
                    if getattr(item, "type", None) == "text":
                        text_parts.append(getattr(item, "text", ""))
                    elif isinstance(item, dict) and item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                if text_parts:
                    return "\n".join(text_parts).strip()

            if isinstance(content, str):
                return content.strip()

        return str(response).strip()

    def summarize_market_analysis(
        self,
        ticker: str,
        model_results: dict,
        *,
        prediction_horizon: int = 1,
        training_period: str = "",
    ) -> str:
        if not model_results:
            return "Claude is ready, but there are no trained models to summarize yet."

        best_model = max(model_results.keys(), key=lambda name: model_results[name]["metrics"]["test_r2"])
        best_metrics = model_results[best_model]["metrics"]
        performance_line = (
            f"Best performing model: {best_model} with test R² {best_metrics['test_r2']:.4f} "
            f"and test MSE {best_metrics['test_mse']:.6f}."
        )

        model_lines = "\n".join(
            f"- {name}: test R-squared {result['metrics']['test_r2']:.4f}, "
            f"test MSE {result['metrics']['test_mse']:.6f}"
            for name, result in model_results.items()
        )

        prompt = (
            f"Analyze this market snapshot for {ticker} using the results below. "
            f"Focus on what the numbers imply, what to watch, and the confidence level. "
            f"Keep the answer concise but practical.\n\n"
            f"Prediction horizon: {prediction_horizon} day(s)\n"
            f"Training period: {training_period or 'selected historical window'}\n"
            f"Model results:\n"
            f"{model_lines}\n\n"
            f"Summary context: {performance_line}"
        )

        return self.generate(prompt)
