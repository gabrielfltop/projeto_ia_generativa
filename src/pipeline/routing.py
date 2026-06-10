"""Model routing cheap-first com fallback."""

from __future__ import annotations

import os
from dataclasses import dataclass

from openai import OpenAI


@dataclass(frozen=True)
class RouteDecision:
    model: str
    complexity: str  # "simple" | "complex"
    reason: str


def classify_complexity(query: str) -> RouteDecision:
    """Classifica a complexidade da query para escolher o modelo adequado.

    Queries analíticas (explique, compare, analise, projete) vão para o modelo
    premium. Perguntas curtas e diretas usam o modelo barato.
    """
    cheap_model = os.environ.get("CHEAP_MODEL", "gemini-2.5-flash-lite")
    premium_model = os.environ.get("PREMIUM_MODEL", "gemini-2.5-pro")

    keywords = ["explique", "compare", "analise", "projete"]

    if any(k in query.lower() for k in keywords):
        return RouteDecision(model=premium_model, complexity="complex", reason="query analítica")
    elif len(query) < 60 and query.strip().endswith("?"):
        return RouteDecision(model=cheap_model, complexity="simple", reason="pergunta curta")
    else:
        return RouteDecision(model=cheap_model, complexity="simple", reason="default")


def make_client() -> OpenAI:
    """Cliente OpenAI-compatible para o provider configurado."""
    if "GEMINI_API_KEY" in os.environ:
        return OpenAI(
            api_key=os.environ["GEMINI_API_KEY"],
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
    return OpenAI()