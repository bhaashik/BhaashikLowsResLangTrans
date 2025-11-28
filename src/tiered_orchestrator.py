"""
Tiered translation orchestrator implementing the 70/20/10 strategy.

This module orchestrates translations using a cost-optimized tiered approach:
- Tier 1 (70%): Free IndicTrans2 Hindi pivot
- Tier 2 (20%): Claude Haiku 3.5 enhancement
- Tier 3 (10%): Claude Haiku 4.5 premium quality
"""

import random
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import json
from datetime import datetime

from .translators import IndicTrans2Translator, HindiPivotTranslator
from .api_clients import ClaudeTranslator
from .quality import QualityMetrics
from .utils.config import Config
from .utils.logger import get_logger
from .utils.cost_tracker import CostTracker


logger = get_logger(__name__)


class TieredOrchestrator:
    """
    Orchestrates translation using a tiered approach for cost optimization.

    The orchestrator automatically routes texts to different translation systems
    based on the configured tier percentages and can optionally use quality
    metrics to decide which texts get premium treatment.
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        cost_tracker: Optional[CostTracker] = None,
        tier_percentages: Optional[Dict[str, float]] = None
    ):
        """
        Initialize tiered orchestrator.

        Args:
            config: Configuration object
            cost_tracker: Cost tracker instance
            tier_percentages: Dictionary with 'free', 'enhancement', 'premium' percentages.
                            If None, uses config defaults (70, 20, 10).
        """
        self.config = config or Config()
        self.cost_tracker = cost_tracker or CostTracker(
            log_file=self.config.cost_log_file,
            currency=self.config.get('cost_tracking.currency', 'INR'),
            enabled=self.config.enable_cost_tracking
        )

        # Get tier percentages
        if tier_percentages is None:
            strategy = self.config.get('strategy.unsupported_strategy', {})
            tiers = strategy.get('tiers', [])
            tier_percentages = {
                'free': tiers[0]['percentage'] if len(tiers) > 0 else 70,
                'enhancement': tiers[1]['percentage'] if len(tiers) > 1 else 20,
                'premium': tiers[2]['percentage'] if len(tiers) > 2 else 10
            }

        self.tier_percentages = tier_percentages

        # Validate percentages
        total = sum(tier_percentages.values())
        if abs(total - 100) > 0.1:
            raise ValueError(f"Tier percentages must sum to 100, got {total}")

        logger.info("="*70)
        logger.info("TIERED ORCHESTRATOR INITIALIZED")
        logger.info("="*70)
        logger.info(f"Tier 1 (Free Hindi Pivot): {tier_percentages['free']}%")
        logger.info(f"Tier 2 (Claude Haiku 3.5): {tier_percentages['enhancement']}%")
        logger.info(f"Tier 3 (Claude Haiku 4.5): {tier_percentages['premium']}%")
        logger.info("="*70)

        # Initialize translators (lazy loading)
        self._hindi_pivot = None
        self._claude_haiku_3_5 = None
        self._claude_haiku_4_5 = None
        self._quality_metrics = None

    @property
    def hindi_pivot(self) -> HindiPivotTranslator:
        """Lazy load Hindi pivot translator."""
        if self._hindi_pivot is None:
            logger.info("Loading Hindi pivot translator...")
            self._hindi_pivot = HindiPivotTranslator(config=self.config)
        return self._hindi_pivot

    @property
    def claude_haiku_3_5(self) -> ClaudeTranslator:
        """Lazy load Claude Haiku 3.5 translator."""
        if self._claude_haiku_3_5 is None:
            logger.info("Loading Claude Haiku 3.5 translator...")
            self._claude_haiku_3_5 = ClaudeTranslator(
                model='haiku_3_5',
                config=self.config,
                cost_tracker=self.cost_tracker
            )
        return self._claude_haiku_3_5

    @property
    def claude_haiku_4_5(self) -> ClaudeTranslator:
        """Lazy load Claude Haiku 4.5 translator."""
        if self._claude_haiku_4_5 is None:
            logger.info("Loading Claude Haiku 4.5 translator...")
            self._claude_haiku_4_5 = ClaudeTranslator(
                model='haiku_4_5',
                config=self.config,
                cost_tracker=self.cost_tracker
            )
        return self._claude_haiku_4_5

    @property
    def quality_metrics(self) -> QualityMetrics:
        """Lazy load quality metrics."""
        if self._quality_metrics is None:
            self._quality_metrics = QualityMetrics()
        return self._quality_metrics

    def _assign_tiers(
        self,
        num_samples: int,
        seed: Optional[int] = None
    ) -> List[str]:
        """
        Assign tier labels to samples based on configured percentages.

        Args:
            num_samples: Total number of samples
            seed: Random seed for reproducibility

        Returns:
            List of tier assignments ('free', 'enhancement', 'premium')
        """
        if seed is not None:
            random.seed(seed)

        # Calculate counts
        free_count = int(num_samples * self.tier_percentages['free'] / 100)
        enhancement_count = int(num_samples * self.tier_percentages['enhancement'] / 100)
        premium_count = num_samples - free_count - enhancement_count

        # Create tier list
        tiers = (
            ['free'] * free_count +
            ['enhancement'] * enhancement_count +
            ['premium'] * premium_count
        )

        # Shuffle for random distribution
        random.shuffle(tiers)

        return tiers

    def translate(
        self,
        texts: List[str],
        src_lang: str,
        tgt_lang: str,
        batch_size: int = 32,
        show_progress: bool = True,
        seed: Optional[int] = None,
        save_results: bool = True,
        output_dir: Optional[str] = None
    ) -> Dict:
        """
        Translate texts using tiered approach.

        Args:
            texts: List of texts to translate
            src_lang: Source language code
            tgt_lang: Target language code
            batch_size: Batch size for processing
            show_progress: Whether to show progress
            seed: Random seed for tier assignment
            save_results: Whether to save results to file
            output_dir: Output directory for results

        Returns:
            Dictionary with translations and metadata
        """
        logger.info("="*70)
        logger.info("TIERED TRANSLATION")
        logger.info("="*70)
        logger.info(f"Source language: {src_lang}")
        logger.info(f"Target language: {tgt_lang}")
        logger.info(f"Total samples: {len(texts):,}")
        logger.info("="*70)

        # Assign tiers
        tier_assignments = self._assign_tiers(len(texts), seed=seed)

        # Count assignments
        tier_counts = {
            'free': tier_assignments.count('free'),
            'enhancement': tier_assignments.count('enhancement'),
            'premium': tier_assignments.count('premium')
        }

        logger.info(f"\nTier Assignment:")
        logger.info(f"  Free (Hindi Pivot): {tier_counts['free']:,} ({tier_counts['free']/len(texts)*100:.1f}%)")
        logger.info(f"  Enhancement (Haiku 3.5): {tier_counts['enhancement']:,} ({tier_counts['enhancement']/len(texts)*100:.1f}%)")
        logger.info(f"  Premium (Haiku 4.5): {tier_counts['premium']:,} ({tier_counts['premium']/len(texts)*100:.1f}%)")

        # Initialize result structures
        translations = [''] * len(texts)
        tier_used = [''] * len(texts)

        # Process each tier
        logger.info("\n" + "="*70)
        logger.info("TIER 1: FREE HINDI PIVOT")
        logger.info("="*70)

        free_indices = [i for i, t in enumerate(tier_assignments) if t == 'free']
        if free_indices:
            free_texts = [texts[i] for i in free_indices]
            free_translations = self.hindi_pivot.translate(
                free_texts,
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                batch_size=batch_size,
                show_progress=show_progress
            )

            for idx, trans in zip(free_indices, free_translations):
                translations[idx] = trans
                tier_used[idx] = 'free'

        # Tier 2: Enhancement
        logger.info("\n" + "="*70)
        logger.info("TIER 2: CLAUDE HAIKU 3.5 ENHANCEMENT")
        logger.info("="*70)

        enhancement_indices = [i for i, t in enumerate(tier_assignments) if t == 'enhancement']
        if enhancement_indices:
            enhancement_texts = [texts[i] for i in enhancement_indices]
            enhancement_translations = self.claude_haiku_3_5.translate(
                enhancement_texts,
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                show_progress=show_progress
            )

            for idx, trans in zip(enhancement_indices, enhancement_translations):
                translations[idx] = trans
                tier_used[idx] = 'enhancement'

        # Tier 3: Premium
        logger.info("\n" + "="*70)
        logger.info("TIER 3: CLAUDE HAIKU 4.5 PREMIUM")
        logger.info("="*70)

        premium_indices = [i for i, t in enumerate(tier_assignments) if t == 'premium']
        if premium_indices:
            premium_texts = [texts[i] for i in premium_indices]
            premium_translations = self.claude_haiku_4_5.translate(
                premium_texts,
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                show_progress=show_progress
            )

            for idx, trans in zip(premium_indices, premium_translations):
                translations[idx] = trans
                tier_used[idx] = 'premium'

        # Compile results
        results = {
            'source_language': src_lang,
            'target_language': tgt_lang,
            'total_samples': len(texts),
            'tier_counts': tier_counts,
            'tier_percentages': self.tier_percentages,
            'sources': texts,
            'translations': translations,
            'tier_assignments': tier_used,
            'cost_summary': self.cost_tracker.get_statistics(),
            'timestamp': datetime.now().isoformat()
        }

        # Save results
        if save_results:
            if output_dir is None:
                output_dir = self.config.base_dir / 'output'
            else:
                output_dir = Path(output_dir)

            output_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"translation_{src_lang}_{tgt_lang}_{timestamp}.json"
            output_path = output_dir / filename

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

            logger.info(f"\n✓ Results saved to: {output_path}")

        # Print summary
        logger.info("\n" + "="*70)
        logger.info("TRANSLATION COMPLETE")
        logger.info("="*70)
        logger.info(f"Total samples translated: {len(translations):,}")
        logger.info(f"\nCost Summary:")
        self.cost_tracker.print_summary()

        return results

    def translate_dataset(
        self,
        dataset,
        src_lang: str,
        tgt_lang: str,
        src_field: str = 'src',
        tgt_field: str = 'tgt',
        max_samples: Optional[int] = None,
        **kwargs
    ) -> Dict:
        """
        Translate a HuggingFace dataset.

        Args:
            dataset: HuggingFace Dataset object
            src_lang: Source language code
            tgt_lang: Target language code
            src_field: Field name for source text
            tgt_field: Field name for target text (for reference)
            max_samples: Maximum number of samples to translate
            **kwargs: Additional arguments passed to translate()

        Returns:
            Dictionary with translations and metadata
        """
        # Extract texts
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))

        texts = [item[src_field] for item in dataset]

        logger.info(f"Translating {len(texts):,} samples from dataset")

        # Translate
        results = self.translate(texts, src_lang, tgt_lang, **kwargs)

        # Add references if available
        if tgt_field in dataset.column_names:
            results['references'] = [item[tgt_field] for item in dataset]

            # Calculate quality metrics
            logger.info("\nCalculating quality metrics...")
            metrics = self.quality_metrics.calculate_all_metrics(
                results['translations'],
                results['references']
            )
            results['quality_metrics'] = metrics

            self.quality_metrics.print_metrics(metrics, "Translation Quality")

        return results

    def get_cost_estimate(
        self,
        num_samples: int,
        avg_tokens_per_sample: int = 100
    ) -> Dict:
        """
        Estimate total cost for translating a number of samples.

        Args:
            num_samples: Number of samples to translate
            avg_tokens_per_sample: Average tokens per sample

        Returns:
            Dictionary with cost breakdown by tier
        """
        tier_assignments = self._assign_tiers(num_samples, seed=42)
        tier_counts = {
            'free': tier_assignments.count('free'),
            'enhancement': tier_assignments.count('enhancement'),
            'premium': tier_assignments.count('premium')
        }

        # Free tier has no cost
        free_cost = 0

        # Enhancement tier cost
        enhancement_estimate = self.claude_haiku_3_5.get_cost_estimate(
            tier_counts['enhancement'],
            avg_tokens_per_sample
        ) if tier_counts['enhancement'] > 0 else {'total_cost': 0}

        # Premium tier cost
        premium_estimate = self.claude_haiku_4_5.get_cost_estimate(
            tier_counts['premium'],
            avg_tokens_per_sample
        ) if tier_counts['premium'] > 0 else {'total_cost': 0}

        total_cost = (
            free_cost +
            enhancement_estimate['total_cost'] +
            premium_estimate['total_cost']
        )

        return {
            'total_samples': num_samples,
            'tier_counts': tier_counts,
            'tier_percentages': self.tier_percentages,
            'free_cost': free_cost,
            'enhancement_cost': enhancement_estimate['total_cost'],
            'premium_cost': premium_estimate['total_cost'],
            'total_cost': total_cost,
            'currency': self.cost_tracker.currency,
            'cost_per_sample': total_cost / num_samples if num_samples > 0 else 0
        }

    def print_cost_estimate(self, num_samples: int, avg_tokens_per_sample: int = 100):
        """
        Print a formatted cost estimate.

        Args:
            num_samples: Number of samples to translate
            avg_tokens_per_sample: Average tokens per sample
        """
        estimate = self.get_cost_estimate(num_samples, avg_tokens_per_sample)

        print("\n" + "="*70)
        print("COST ESTIMATE")
        print("="*70)
        print(f"Total Samples: {estimate['total_samples']:,}")
        print(f"\nTier Distribution:")
        print(f"  Free (Hindi Pivot): {estimate['tier_counts']['free']:,} samples")
        print(f"  Enhancement (Haiku 3.5): {estimate['tier_counts']['enhancement']:,} samples")
        print(f"  Premium (Haiku 4.5): {estimate['tier_counts']['premium']:,} samples")
        print(f"\nCost Breakdown:")
        print(f"  Free Tier: {estimate['currency']} {estimate['free_cost']:.2f}")
        print(f"  Enhancement Tier: {estimate['currency']} {estimate['enhancement_cost']:.2f}")
        print(f"  Premium Tier: {estimate['currency']} {estimate['premium_cost']:.2f}")
        print(f"\nTotal Cost: {estimate['currency']} {estimate['total_cost']:.2f}")
        print(f"Cost per Sample: {estimate['currency']} {estimate['cost_per_sample']:.4f}")
        print("="*70 + "\n")
