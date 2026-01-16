"""
Hybrid translation pipeline with quality-based routing.

Implements a multi-tier translation strategy:
1. Fine-tuned model (primary, fast, zero-cost)
2. API-based refinement (for low-confidence translations)
3. Ensemble methods (for critical translations)

The pipeline automatically routes translations based on quality estimates,
achieving optimal quality/cost tradeoffs according to configured thresholds.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union
from enum import Enum
import logging
from pathlib import Path

from src.training.pipeline.quality_estimator import QualityEstimator, QualityEstimate

logger = logging.getLogger(__name__)


class RouteDecision(str, Enum):
    """Routing decision for a translation."""
    FINETUNED = "finetuned"
    API = "api"
    ENSEMBLE = "ensemble"


@dataclass
class HybridConfig:
    """Configuration for hybrid translation pipeline.

    Attributes:
        finetuned_percentage: Target % of translations using fine-tuned model
        api_percentage: Target % using API refinement
        ensemble_percentage: Target % using ensemble methods
        quality_threshold: Quality score threshold for API fallback
        ensemble_threshold: Threshold for ensemble (lower = more critical)
        use_comet_qe: Whether to use COMET-QE for quality estimation
        force_percentages: Enforce exact percentages (vs. quality-based)
        api_provider: API provider for fallback ('openai', 'anthropic', etc.)
        api_model: API model name
        ensemble_method: Ensemble method ('comet', 'mbr', 'voting')
        track_costs: Whether to track and report costs
        custom_params: Additional custom parameters
    """
    # Target distribution
    finetuned_percentage: float = 90.0
    api_percentage: float = 9.0
    ensemble_percentage: float = 1.0

    # Quality thresholds
    quality_threshold: float = 0.7  # Below this → API
    ensemble_threshold: float = 0.5  # Below this → ensemble

    # Quality estimation
    use_comet_qe: bool = False

    # Routing strategy
    force_percentages: bool = False  # If True, use percentages regardless of quality

    # API configuration
    api_provider: str = "openai"
    api_model: str = "gpt-4o-mini"

    # Ensemble configuration
    ensemble_method: str = "comet"  # comet, mbr, voting

    # Monitoring
    track_costs: bool = True

    # Extensibility
    custom_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate configuration."""
        total = self.finetuned_percentage + self.api_percentage + self.ensemble_percentage
        if abs(total - 100.0) > 0.01:
            raise ValueError(
                f"Percentages must sum to 100%, got {total}%"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "finetuned_percentage": self.finetuned_percentage,
            "api_percentage": self.api_percentage,
            "ensemble_percentage": self.ensemble_percentage,
            "quality_threshold": self.quality_threshold,
            "ensemble_threshold": self.ensemble_threshold,
            "use_comet_qe": self.use_comet_qe,
            "force_percentages": self.force_percentages,
            "api_provider": self.api_provider,
            "api_model": self.api_model,
            "ensemble_method": self.ensemble_method,
            "track_costs": self.track_costs,
            **self.custom_params,
        }


@dataclass
class HybridResult:
    """Result from hybrid translation.

    Attributes:
        translations: Final translations
        routes: Routing decision for each translation
        quality_estimates: Quality estimates for fine-tuned outputs
        costs: Cost breakdown if tracking enabled
        metadata: Additional metadata (timing, model info, etc.)
    """
    translations: List[str]
    routes: List[RouteDecision]
    quality_estimates: List[QualityEstimate]
    costs: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_route_statistics(self) -> Dict[str, Any]:
        """Get statistics about routing decisions."""
        total = len(self.routes)
        if total == 0:
            return {}

        route_counts = {}
        for route in RouteDecision:
            count = sum(1 for r in self.routes if r == route)
            route_counts[route.value] = {
                "count": count,
                "percentage": 100.0 * count / total
            }

        return route_counts


class HybridPipeline:
    """
    Hybrid translation pipeline with quality-based routing.

    Combines fine-tuned models, API translation, and ensemble methods
    to achieve optimal quality/cost tradeoffs.

    Example:
        # Create pipeline
        pipeline = HybridPipeline(
            finetuned_model=finetuned_provider,
            api_model=openai_provider,
            config=HybridConfig(
                finetuned_percentage=90.0,
                api_percentage=10.0
            )
        )

        # Translate
        result = pipeline.translate(
            texts=["नमस्ते", "आप कैसे हैं?"],
            source_lang="hi",
            target_lang="bho"
        )

        # Check routing
        print(result.get_route_statistics())
    """

    def __init__(
        self,
        finetuned_model: Any,
        api_model: Optional[Any] = None,
        ensemble_models: Optional[List[Any]] = None,
        config: Optional[HybridConfig] = None,
    ):
        """
        Initialize hybrid pipeline.

        Args:
            finetuned_model: Fine-tuned model (AbstractMTModel or provider)
            api_model: API model for fallback (BaseTranslator)
            ensemble_models: List of models for ensemble (optional)
            config: Hybrid configuration
        """
        self.finetuned_model = finetuned_model
        self.api_model = api_model
        self.ensemble_models = ensemble_models or []
        self.config = config or HybridConfig()

        # Initialize quality estimator
        self.quality_estimator = QualityEstimator(
            use_comet_qe=self.config.use_comet_qe,
            threshold=self.config.quality_threshold,
        )

        # Initialize ensemble if configured
        self.ensemble = None
        if self.ensemble_models and self.config.ensemble_percentage > 0:
            self._init_ensemble()

        # Statistics
        self.stats = {
            "total_translations": 0,
            "finetuned_count": 0,
            "api_count": 0,
            "ensemble_count": 0,
            "total_cost": 0.0,
        }

        logger.info(
            f"Initialized HybridPipeline: "
            f"finetuned={self.config.finetuned_percentage}%, "
            f"api={self.config.api_percentage}%, "
            f"ensemble={self.config.ensemble_percentage}%"
        )

    def _init_ensemble(self):
        """Initialize ensemble method."""
        from src.training.ensemble import COMETEnsemble, MBREnsemble, VotingEnsemble
        from src.training.ensemble.base import EnsembleConfig

        ensemble_config = EnsembleConfig(method=self.config.ensemble_method)

        if self.config.ensemble_method == "comet":
            self.ensemble = COMETEnsemble(self.ensemble_models, ensemble_config)
        elif self.config.ensemble_method == "mbr":
            self.ensemble = MBREnsemble(self.ensemble_models, ensemble_config)
        elif self.config.ensemble_method == "voting":
            self.ensemble = VotingEnsemble(self.ensemble_models, ensemble_config)
        else:
            raise ValueError(f"Unknown ensemble method: {self.config.ensemble_method}")

    def translate(
        self,
        texts: List[str],
        source_lang: str,
        target_lang: str,
        **kwargs
    ) -> HybridResult:
        """
        Translate texts using hybrid pipeline.

        Args:
            texts: Source texts
            source_lang: Source language code
            target_lang: Target language code
            **kwargs: Additional arguments

        Returns:
            HybridResult with translations and metadata
        """
        logger.info(
            f"Hybrid translation: {len(texts)} texts, "
            f"{source_lang} → {target_lang}"
        )

        # Step 1: Translate all with fine-tuned model
        finetuned_translations = self._translate_finetuned(
            texts, source_lang, target_lang
        )

        # Step 2: Estimate quality
        quality_estimates = self.quality_estimator.estimate_batch(
            texts, finetuned_translations
        )

        # Step 3: Decide routing
        routes = self._decide_routes(quality_estimates, len(texts))

        # Step 4: Apply fallbacks
        final_translations = []
        total_cost = 0.0

        for idx, (text, finetuned, route) in enumerate(
            zip(texts, finetuned_translations, routes)
        ):
            if route == RouteDecision.FINETUNED:
                final_translations.append(finetuned)
                self.stats["finetuned_count"] += 1

            elif route == RouteDecision.API:
                # Use API for refinement
                api_translation, cost = self._translate_api(
                    text, source_lang, target_lang
                )
                final_translations.append(api_translation)
                total_cost += cost
                self.stats["api_count"] += 1

            elif route == RouteDecision.ENSEMBLE:
                # Use ensemble
                ensemble_translation = self._translate_ensemble(
                    text, source_lang, target_lang
                )
                final_translations.append(ensemble_translation)
                self.stats["ensemble_count"] += 1

        self.stats["total_translations"] += len(texts)
        self.stats["total_cost"] += total_cost

        # Create result
        result = HybridResult(
            translations=final_translations,
            routes=routes,
            quality_estimates=quality_estimates,
            costs={
                "total": total_cost,
                "per_translation": total_cost / len(texts) if texts else 0,
                "currency": "USD",
            },
            metadata={
                "source_lang": source_lang,
                "target_lang": target_lang,
                "config": self.config.to_dict(),
                "num_texts": len(texts),
            }
        )

        logger.info(
            f"Hybrid translation complete: "
            f"{result.get_route_statistics()}, "
            f"cost=${total_cost:.4f}"
        )

        return result

    def _translate_finetuned(
        self,
        texts: List[str],
        source_lang: str,
        target_lang: str
    ) -> List[str]:
        """Translate with fine-tuned model."""
        logger.debug(f"Translating {len(texts)} texts with fine-tuned model")

        # Check if it's a BaseTranslator (provider) or AbstractMTModel
        if hasattr(self.finetuned_model, 'translate_sync'):
            # It's a provider
            from universal_translate.core import TranslationRequest
            request = TranslationRequest(
                units=texts,
                source_lang=source_lang,
                target_lang=target_lang
            )
            response = self.finetuned_model.translate_sync(request)
            return [r.translation for r in response.results]
        else:
            # It's an AbstractMTModel
            return self.finetuned_model.translate(
                texts, source_lang=source_lang, target_lang=target_lang
            )

    def _translate_api(
        self,
        text: str,
        source_lang: str,
        target_lang: str
    ) -> tuple:
        """Translate with API model.

        Returns:
            (translation, cost)
        """
        if not self.api_model:
            logger.warning("No API model configured, using fine-tuned")
            return text, 0.0

        logger.debug(f"Using API fallback for text: {text[:50]}...")

        try:
            from universal_translate.core import TranslationRequest
            request = TranslationRequest(
                units=[text],
                source_lang=source_lang,
                target_lang=target_lang
            )
            response = self.api_model.translate_sync(request)
            translation = response.results[0].translation
            cost = response.cost_info.total_cost if response.cost_info else 0.0
            return translation, cost
        except Exception as e:
            logger.error(f"API translation failed: {e}, using fine-tuned")
            return text, 0.0

    def _translate_ensemble(
        self,
        text: str,
        source_lang: str,
        target_lang: str
    ) -> str:
        """Translate with ensemble."""
        if not self.ensemble:
            logger.warning("No ensemble configured, using fine-tuned")
            return text

        logger.debug(f"Using ensemble for text: {text[:50]}...")

        try:
            result = self.ensemble.translate(
                [text], source_lang, target_lang
            )
            return result.translations[0]
        except Exception as e:
            logger.error(f"Ensemble translation failed: {e}, using fine-tuned")
            return text

    def _decide_routes(
        self,
        quality_estimates: List[QualityEstimate],
        num_texts: int
    ) -> List[RouteDecision]:
        """Decide routing for each translation."""
        if self.config.force_percentages:
            # Force exact percentages (deterministic)
            return self._decide_routes_percentage(num_texts)
        else:
            # Quality-based routing
            return self._decide_routes_quality(quality_estimates)

    def _decide_routes_quality(
        self,
        quality_estimates: List[QualityEstimate]
    ) -> List[RouteDecision]:
        """Decide routes based on quality scores."""
        routes = []

        for estimate in quality_estimates:
            if estimate.score >= self.config.quality_threshold:
                # High quality → use fine-tuned
                routes.append(RouteDecision.FINETUNED)
            elif estimate.score >= self.config.ensemble_threshold:
                # Medium quality → API refinement
                routes.append(RouteDecision.API)
            else:
                # Low quality → ensemble if available, else API
                if self.ensemble and self.config.ensemble_percentage > 0:
                    routes.append(RouteDecision.ENSEMBLE)
                else:
                    routes.append(RouteDecision.API)

        return routes

    def _decide_routes_percentage(self, num_texts: int) -> List[RouteDecision]:
        """Decide routes to match target percentages exactly."""
        num_finetuned = int(num_texts * self.config.finetuned_percentage / 100)
        num_api = int(num_texts * self.config.api_percentage / 100)
        num_ensemble = num_texts - num_finetuned - num_api

        routes = (
            [RouteDecision.FINETUNED] * num_finetuned +
            [RouteDecision.API] * num_api +
            [RouteDecision.ENSEMBLE] * num_ensemble
        )

        return routes

    def get_statistics(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        total = self.stats["total_translations"]
        if total == 0:
            return self.stats

        return {
            **self.stats,
            "finetuned_percentage": 100.0 * self.stats["finetuned_count"] / total,
            "api_percentage": 100.0 * self.stats["api_count"] / total,
            "ensemble_percentage": 100.0 * self.stats["ensemble_count"] / total,
            "average_cost_per_translation": self.stats["total_cost"] / total,
        }

    def reset_statistics(self):
        """Reset pipeline statistics."""
        self.stats = {
            "total_translations": 0,
            "finetuned_count": 0,
            "api_count": 0,
            "ensemble_count": 0,
            "total_cost": 0.0,
        }
