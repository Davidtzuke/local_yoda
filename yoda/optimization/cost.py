"""Per-model pricing, usage reports, and budget alerts."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model pricing (USD per 1M tokens)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelPricing:
    """Pricing for a specific model (per 1M tokens)."""

    model: str
    input_price: float  # USD per 1M input tokens
    output_price: float  # USD per 1M output tokens
    cached_input_price: float = 0.0  # discount for cached/prompt-cached

    def cost_for(self, input_tokens: int, output_tokens: int, cached_tokens: int = 0) -> float:
        """Calculate cost for a specific usage."""
        input_cost = (input_tokens - cached_tokens) * self.input_price / 1_000_000
        cached_cost = cached_tokens * self.cached_input_price / 1_000_000
        output_cost = output_tokens * self.output_price / 1_000_000
        return input_cost + cached_cost + output_cost


# Well-known pricing (as of early 2026)
MODEL_PRICING: dict[str, ModelPricing] = {
    # Anthropic
    "claude-sonnet-4-20250514": ModelPricing("claude-sonnet-4-20250514", 3.0, 15.0, 1.5),
    "claude-3-5-sonnet-20241022": ModelPricing("claude-3-5-sonnet-20241022", 3.0, 15.0, 1.5),
    "claude-3-haiku-20240307": ModelPricing("claude-3-haiku-20240307", 0.25, 1.25, 0.125),
    "claude-3-opus-20240229": ModelPricing("claude-3-opus-20240229", 15.0, 75.0, 7.5),
    # OpenAI
    "gpt-4": ModelPricing("gpt-4", 30.0, 60.0),
    "gpt-4-turbo": ModelPricing("gpt-4-turbo", 10.0, 30.0),
    "gpt-4o": ModelPricing("gpt-4o", 2.5, 10.0, 1.25),
    "gpt-4o-mini": ModelPricing("gpt-4o-mini", 0.15, 0.60, 0.075),
    "gpt-3.5-turbo": ModelPricing("gpt-3.5-turbo", 0.5, 1.5),
    # Local (free)
    "ollama": ModelPricing("ollama", 0.0, 0.0),
    "local": ModelPricing("local", 0.0, 0.0),
}


def get_pricing(model: str) -> ModelPricing:
    """Get pricing for a model, with fuzzy matching."""
    if model in MODEL_PRICING:
        return MODEL_PRICING[model]
    for key, pricing in MODEL_PRICING.items():
        if model.startswith(key) or key.startswith(model):
            return pricing
    # Default: assume moderate pricing
    return ModelPricing(model, 3.0, 15.0)


# ---------------------------------------------------------------------------
# Usage records
# ---------------------------------------------------------------------------

@dataclass
class UsageRecord:
    """Single API call usage record."""

    timestamp: float = field(default_factory=time.time)
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    cost: float = 0.0
    latency_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BudgetAlert:
    """Budget alert configuration."""

    name: str
    threshold_usd: float
    period: str = "daily"  # "daily", "weekly", "monthly", "total"
    callback: Callable[[str, float, float], None] | None = None  # (name, current, threshold)
    triggered: bool = False


# ---------------------------------------------------------------------------
# Cost tracker
# ---------------------------------------------------------------------------

class CostTracker:
    """Track API costs with per-model pricing, usage reports, and budget alerts.

    Features:
    - Per-request cost calculation
    - Cumulative tracking with persistence
    - Time-windowed reports (daily, weekly, monthly)
    - Budget alerts with callbacks
    - Export to JSON
    """

    def __init__(
        self,
        persist_path: str | Path = "~/.yoda/cost_log.json",
        model: str = "default",
    ) -> None:
        self.persist_path = Path(persist_path).expanduser()
        self.model = model
        self._records: list[UsageRecord] = []
        self._alerts: list[BudgetAlert] = []
        self._session_start = time.time()

    # -- Lifecycle ---------------------------------------------------------

    def initialize(self) -> None:
        """Load persisted records."""
        if self.persist_path.exists():
            try:
                data = json.loads(self.persist_path.read_text())
                self._records = [
                    UsageRecord(**r) for r in data.get("records", [])
                ]
                logger.info("Loaded %d cost records", len(self._records))
            except Exception:
                logger.warning("Failed to load cost log, starting fresh")
                self._records = []

    def save(self) -> None:
        """Persist records to disk."""
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "records": [
                {
                    "timestamp": r.timestamp,
                    "model": r.model,
                    "input_tokens": r.input_tokens,
                    "output_tokens": r.output_tokens,
                    "cached_tokens": r.cached_tokens,
                    "cost": r.cost,
                    "latency_ms": r.latency_ms,
                    "metadata": r.metadata,
                }
                for r in self._records
            ]
        }
        self.persist_path.write_text(json.dumps(data, indent=2))

    # -- Recording ---------------------------------------------------------

    def record(
        self,
        input_tokens: int,
        output_tokens: int,
        model: str | None = None,
        cached_tokens: int = 0,
        latency_ms: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> UsageRecord:
        """Record a single API call and calculate cost."""
        m = model or self.model
        pricing = get_pricing(m)
        cost = pricing.cost_for(input_tokens, output_tokens, cached_tokens)

        record = UsageRecord(
            model=m,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
            cost=cost,
            latency_ms=latency_ms,
            metadata=metadata or {},
        )
        self._records.append(record)
        self._check_alerts()

        # Auto-save every 10 records
        if len(self._records) % 10 == 0:
            self.save()

        return record

    # -- Alerts ------------------------------------------------------------

    def add_alert(
        self,
        name: str,
        threshold_usd: float,
        period: str = "daily",
        callback: Callable[[str, float, float], None] | None = None,
    ) -> BudgetAlert:
        """Add a budget alert."""
        alert = BudgetAlert(
            name=name,
            threshold_usd=threshold_usd,
            period=period,
            callback=callback,
        )
        self._alerts.append(alert)
        return alert

    def _check_alerts(self) -> None:
        """Check all alerts against current spending."""
        for alert in self._alerts:
            if alert.triggered:
                continue
            current = self._cost_for_period(alert.period)
            if current >= alert.threshold_usd:
                alert.triggered = True
                msg = (
                    f"Budget alert '{alert.name}': ${current:.4f} "
                    f"exceeds ${alert.threshold_usd:.4f} ({alert.period})"
                )
                logger.warning(msg)
                if alert.callback:
                    try:
                        alert.callback(alert.name, current, alert.threshold_usd)
                    except Exception:
                        logger.exception("Alert callback failed")

    def _cost_for_period(self, period: str) -> float:
        """Calculate total cost for a time period."""
        now = time.time()
        if period == "total":
            cutoff = 0.0
        elif period == "daily":
            cutoff = now - 86400
        elif period == "weekly":
            cutoff = now - 604800
        elif period == "monthly":
            cutoff = now - 2592000
        elif period == "session":
            cutoff = self._session_start
        else:
            cutoff = 0.0

        return sum(r.cost for r in self._records if r.timestamp >= cutoff)

    # -- Reports -----------------------------------------------------------

    def session_report(self) -> dict[str, Any]:
        """Report for current session."""
        session_records = [
            r for r in self._records if r.timestamp >= self._session_start
        ]
        return self._build_report(session_records, "session")

    def daily_report(self) -> dict[str, Any]:
        """Report for last 24 hours."""
        cutoff = time.time() - 86400
        records = [r for r in self._records if r.timestamp >= cutoff]
        return self._build_report(records, "daily")

    def total_report(self) -> dict[str, Any]:
        """Report for all time."""
        return self._build_report(self._records, "total")

    def model_breakdown(self) -> dict[str, dict[str, Any]]:
        """Cost breakdown by model."""
        by_model: dict[str, list[UsageRecord]] = {}
        for r in self._records:
            by_model.setdefault(r.model, []).append(r)

        result = {}
        for model, records in by_model.items():
            result[model] = {
                "requests": len(records),
                "input_tokens": sum(r.input_tokens for r in records),
                "output_tokens": sum(r.output_tokens for r in records),
                "total_cost": f"${sum(r.cost for r in records):.6f}",
                "avg_cost": f"${sum(r.cost for r in records) / len(records):.6f}",
            }
        return result

    def _build_report(
        self, records: list[UsageRecord], period: str
    ) -> dict[str, Any]:
        if not records:
            return {
                "period": period,
                "requests": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_cost": "$0.000000",
                "avg_cost_per_request": "$0.000000",
                "avg_latency_ms": 0.0,
            }

        total_cost = sum(r.cost for r in records)
        total_latency = sum(r.latency_ms for r in records)
        return {
            "period": period,
            "requests": len(records),
            "input_tokens": sum(r.input_tokens for r in records),
            "output_tokens": sum(r.output_tokens for r in records),
            "total_cost": f"${total_cost:.6f}",
            "avg_cost_per_request": f"${total_cost / len(records):.6f}",
            "avg_latency_ms": round(total_latency / len(records), 1),
        }

    # -- Convenience -------------------------------------------------------

    @property
    def total_cost(self) -> float:
        return sum(r.cost for r in self._records)

    @property
    def session_cost(self) -> float:
        return self._cost_for_period("session")

    @property
    def request_count(self) -> int:
        return len(self._records)
