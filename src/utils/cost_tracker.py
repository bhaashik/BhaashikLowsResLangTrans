"""Cost tracking for API usage."""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict


@dataclass
class APICall:
    """Represents a single API call with cost information."""
    timestamp: str
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    input_cost: float
    output_cost: float
    total_cost: float
    currency: str
    language_pair: str
    sample_count: int


class CostTracker:
    """Tracks and manages API usage costs."""

    def __init__(self, log_file: str, currency: str = "INR", enabled: bool = True):
        """
        Initialize cost tracker.

        Args:
            log_file: Path to JSON log file for storing cost data
            currency: Currency for cost tracking (INR, USD)
            enabled: Whether cost tracking is enabled
        """
        self.log_file = Path(log_file)
        self.currency = currency
        self.enabled = enabled

        # Create log directory
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        # Load existing data
        self.calls = []
        self._load_data()

    def _load_data(self):
        """Load existing cost data from file."""
        if self.log_file.exists():
            try:
                with open(self.log_file, "r") as f:
                    data = json.load(f)
                    self.calls = [APICall(**call) for call in data.get("calls", [])]
            except (json.JSONDecodeError, TypeError):
                self.calls = []

    def _save_data(self):
        """Save cost data to file."""
        if not self.enabled:
            return

        data = {
            "currency": self.currency,
            "total_calls": len(self.calls),
            "total_cost": sum(call.total_cost for call in self.calls),
            "last_updated": datetime.now().isoformat(),
            "calls": [asdict(call) for call in self.calls]
        }

        with open(self.log_file, "w") as f:
            json.dump(data, f, indent=2)

    def log_api_call(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        input_cost_per_1m: float,
        output_cost_per_1m: float,
        language_pair: str = "",
        sample_count: int = 1
    ) -> float:
        """
        Log an API call and calculate cost.

        Args:
            provider: API provider (e.g., 'anthropic', 'openai')
            model: Model name
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            input_cost_per_1m: Cost per 1M input tokens
            output_cost_per_1m: Cost per 1M output tokens
            language_pair: Language pair (e.g., 'en-bho')
            sample_count: Number of samples translated

        Returns:
            Total cost for this call
        """
        if not self.enabled:
            return 0.0

        # Calculate costs
        input_cost = (input_tokens / 1_000_000) * input_cost_per_1m
        output_cost = (output_tokens / 1_000_000) * output_cost_per_1m
        total_cost = input_cost + output_cost

        # Create call record
        call = APICall(
            timestamp=datetime.now().isoformat(),
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=total_cost,
            currency=self.currency,
            language_pair=language_pair,
            sample_count=sample_count
        )

        self.calls.append(call)
        self._save_data()

        return total_cost

    def get_total_cost(self, provider: Optional[str] = None, model: Optional[str] = None) -> float:
        """
        Get total cost, optionally filtered by provider and/or model.

        Args:
            provider: Filter by provider
            model: Filter by model

        Returns:
            Total cost
        """
        filtered_calls = self.calls

        if provider:
            filtered_calls = [c for c in filtered_calls if c.provider == provider]

        if model:
            filtered_calls = [c for c in filtered_calls if c.model == model]

        return sum(call.total_cost for call in filtered_calls)

    def get_cost_by_language(self) -> Dict[str, float]:
        """
        Get cost breakdown by language pair.

        Returns:
            Dictionary mapping language pairs to total cost
        """
        costs = defaultdict(float)
        for call in self.calls:
            if call.language_pair:
                costs[call.language_pair] += call.total_cost
        return dict(costs)

    def get_cost_by_provider(self) -> Dict[str, float]:
        """
        Get cost breakdown by provider.

        Returns:
            Dictionary mapping providers to total cost
        """
        costs = defaultdict(float)
        for call in self.calls:
            costs[call.provider] += call.total_cost
        return dict(costs)

    def get_statistics(self) -> Dict:
        """
        Get comprehensive cost statistics.

        Returns:
            Dictionary with various statistics
        """
        if not self.calls:
            return {
                "total_calls": 0,
                "total_cost": 0,
                "currency": self.currency
            }

        return {
            "total_calls": len(self.calls),
            "total_cost": sum(call.total_cost for call in self.calls),
            "total_input_tokens": sum(call.input_tokens for call in self.calls),
            "total_output_tokens": sum(call.output_tokens for call in self.calls),
            "total_samples": sum(call.sample_count for call in self.calls),
            "cost_by_provider": self.get_cost_by_provider(),
            "cost_by_language": self.get_cost_by_language(),
            "currency": self.currency,
            "first_call": self.calls[0].timestamp if self.calls else None,
            "last_call": self.calls[-1].timestamp if self.calls else None
        }

    def print_summary(self):
        """Print a summary of costs to console."""
        stats = self.get_statistics()

        print("\n" + "=" * 70)
        print("COST TRACKING SUMMARY")
        print("=" * 70)
        print(f"Total API Calls: {stats['total_calls']:,}")
        print(f"Total Samples: {stats['total_samples']:,}")
        print(f"Total Cost: {stats['currency']} {stats['total_cost']:,.2f}")
        print(f"\nTotal Input Tokens: {stats['total_input_tokens']:,}")
        print(f"Total Output Tokens: {stats['total_output_tokens']:,}")

        print("\nCost by Provider:")
        for provider, cost in stats['cost_by_provider'].items():
            print(f"  {provider}: {stats['currency']} {cost:,.2f}")

        if stats['cost_by_language']:
            print("\nCost by Language Pair:")
            for lang_pair, cost in sorted(stats['cost_by_language'].items(), key=lambda x: x[1], reverse=True):
                print(f"  {lang_pair}: {stats['currency']} {cost:,.2f}")

        print("=" * 70 + "\n")

    def check_budget(self, daily_budget: Optional[float] = None, total_budget: Optional[float] = None) -> Dict:
        """
        Check if costs are within budget limits.

        Args:
            daily_budget: Daily budget limit
            total_budget: Total budget limit

        Returns:
            Dictionary with budget status
        """
        today = datetime.now().date().isoformat()
        today_calls = [c for c in self.calls if c.timestamp.startswith(today)]
        today_cost = sum(call.total_cost for call in today_calls)
        total_cost = self.get_total_cost()

        status = {
            "today_cost": today_cost,
            "total_cost": total_cost,
            "currency": self.currency,
            "within_daily_budget": True,
            "within_total_budget": True,
            "warnings": []
        }

        if daily_budget and today_cost > daily_budget:
            status["within_daily_budget"] = False
            status["warnings"].append(f"Daily budget exceeded: {today_cost:.2f} > {daily_budget:.2f}")

        if total_budget and total_cost > total_budget:
            status["within_total_budget"] = False
            status["warnings"].append(f"Total budget exceeded: {total_cost:.2f} > {total_budget:.2f}")

        return status
