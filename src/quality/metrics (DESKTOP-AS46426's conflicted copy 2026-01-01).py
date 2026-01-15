"""Quality metrics for translation evaluation."""

from typing import List, Dict, Optional, Tuple
import numpy as np
try:
    from sacrebleu import corpus_bleu, sentence_bleu
    from sacrebleu.metrics import CHRF, TER
    SACREBLEU_AVAILABLE = True
except ImportError:
    SACREBLEU_AVAILABLE = False

from ..utils.logger import get_logger


logger = get_logger(__name__)


class QualityMetrics:
    """
    Quality assessment metrics for translations.

    Supports:
    - BLEU score
    - chrF score
    - TER (Translation Error Rate)
    - Length ratio
    - Basic statistics
    """

    def __init__(self):
        """Initialize quality metrics calculator."""
        if not SACREBLEU_AVAILABLE:
            logger.warning(
                "sacrebleu not available. Install with: pip install sacrebleu\n"
                "Some metrics will not be available."
            )

        self.chrf = CHRF() if SACREBLEU_AVAILABLE else None
        self.ter = TER() if SACREBLEU_AVAILABLE else None

    def calculate_bleu(
        self,
        hypotheses: List[str],
        references: List[str],
        tokenize: str = 'intl'
    ) -> Dict[str, float]:
        """
        Calculate BLEU score.

        Args:
            hypotheses: List of translated texts
            references: List of reference translations
            tokenize: Tokenization method ('intl', '13a', 'zh', 'ja-mecab')

        Returns:
            Dictionary with BLEU scores
        """
        if not SACREBLEU_AVAILABLE:
            logger.error("sacrebleu not available")
            return {}

        # Corpus-level BLEU
        refs = [[ref] for ref in references]
        bleu = corpus_bleu(hypotheses, refs, tokenize=tokenize)

        # Sentence-level BLEU
        sentence_bleus = []
        for hyp, ref in zip(hypotheses, references):
            try:
                sb = sentence_bleu(hyp, [ref], tokenize=tokenize)
                sentence_bleus.append(sb.score)
            except Exception:
                sentence_bleus.append(0.0)

        return {
            'corpus_bleu': bleu.score,
            'sentence_bleu_mean': np.mean(sentence_bleus),
            'sentence_bleu_std': np.std(sentence_bleus),
            'sentence_bleu_min': np.min(sentence_bleus),
            'sentence_bleu_max': np.max(sentence_bleus)
        }

    def calculate_chrf(
        self,
        hypotheses: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """
        Calculate chrF score.

        Args:
            hypotheses: List of translated texts
            references: List of reference translations

        Returns:
            Dictionary with chrF scores
        """
        if not self.chrf:
            logger.error("chrF metric not available")
            return {}

        # Corpus-level chrF
        refs = [[ref] for ref in references]
        chrf_score = self.chrf.corpus_score(hypotheses, refs)

        return {
            'chrf': chrf_score.score
        }

    def calculate_ter(
        self,
        hypotheses: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """
        Calculate TER (Translation Error Rate).

        Args:
            hypotheses: List of translated texts
            references: List of reference translations

        Returns:
            Dictionary with TER scores
        """
        if not self.ter:
            logger.error("TER metric not available")
            return {}

        # Corpus-level TER
        refs = [[ref] for ref in references]
        ter_score = self.ter.corpus_score(hypotheses, refs)

        return {
            'ter': ter_score.score
        }

    def calculate_length_ratio(
        self,
        hypotheses: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """
        Calculate length ratio statistics.

        Args:
            hypotheses: List of translated texts
            references: List of reference translations

        Returns:
            Dictionary with length ratio statistics
        """
        ratios = []
        for hyp, ref in zip(hypotheses, references):
            hyp_len = len(hyp.split())
            ref_len = len(ref.split())
            if ref_len > 0:
                ratios.append(hyp_len / ref_len)
            else:
                ratios.append(0.0)

        return {
            'length_ratio_mean': np.mean(ratios),
            'length_ratio_std': np.std(ratios),
            'length_ratio_min': np.min(ratios),
            'length_ratio_max': np.max(ratios)
        }

    def calculate_all_metrics(
        self,
        hypotheses: List[str],
        references: List[str],
        include_bleu: bool = True,
        include_chrf: bool = True,
        include_ter: bool = False,
        include_length: bool = True
    ) -> Dict[str, float]:
        """
        Calculate all available metrics.

        Args:
            hypotheses: List of translated texts
            references: List of reference translations
            include_bleu: Whether to calculate BLEU
            include_chrf: Whether to calculate chrF
            include_ter: Whether to calculate TER
            include_length: Whether to calculate length ratios

        Returns:
            Dictionary with all metrics
        """
        metrics = {}

        if include_bleu:
            metrics.update(self.calculate_bleu(hypotheses, references))

        if include_chrf:
            metrics.update(self.calculate_chrf(hypotheses, references))

        if include_ter:
            metrics.update(self.calculate_ter(hypotheses, references))

        if include_length:
            metrics.update(self.calculate_length_ratio(hypotheses, references))

        return metrics

    def compare_translations(
        self,
        translations: Dict[str, List[str]],
        references: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare multiple translation systems.

        Args:
            translations: Dictionary mapping system names to translations
            references: Reference translations

        Returns:
            Dictionary mapping system names to their metrics
        """
        results = {}

        for system_name, hyps in translations.items():
            logger.info(f"Evaluating {system_name}...")
            metrics = self.calculate_all_metrics(hyps, references)
            results[system_name] = metrics

        return results

    def print_metrics(self, metrics: Dict[str, float], title: str = "Quality Metrics"):
        """
        Print metrics in a formatted way.

        Args:
            metrics: Dictionary of metrics
            title: Title for the metrics display
        """
        print("\n" + "="*70)
        print(title)
        print("="*70)

        # Group metrics
        bleu_metrics = {k: v for k, v in metrics.items() if 'bleu' in k.lower()}
        chrf_metrics = {k: v for k, v in metrics.items() if 'chrf' in k.lower()}
        ter_metrics = {k: v for k, v in metrics.items() if 'ter' in k.lower()}
        length_metrics = {k: v for k, v in metrics.items() if 'length' in k.lower()}

        if bleu_metrics:
            print("\nBLEU Scores:")
            for k, v in bleu_metrics.items():
                print(f"  {k}: {v:.2f}")

        if chrf_metrics:
            print("\nchrF Scores:")
            for k, v in chrf_metrics.items():
                print(f"  {k}: {v:.2f}")

        if ter_metrics:
            print("\nTER Scores:")
            for k, v in ter_metrics.items():
                print(f"  {k}: {v:.2f}")

        if length_metrics:
            print("\nLength Ratios:")
            for k, v in length_metrics.items():
                print(f"  {k}: {v:.4f}")

        print("="*70 + "\n")

    def sample_for_review(
        self,
        sources: List[str],
        translations: List[str],
        references: Optional[List[str]] = None,
        sample_rate: float = 0.01,
        min_samples: int = 10,
        max_samples: int = 100,
        seed: int = 42
    ) -> List[Tuple[int, str, str, Optional[str]]]:
        """
        Sample translations for manual review.

        Args:
            sources: Source texts
            translations: Translated texts
            references: Optional reference translations
            sample_rate: Fraction of data to sample
            min_samples: Minimum number of samples
            max_samples: Maximum number of samples
            seed: Random seed for reproducibility

        Returns:
            List of tuples: (index, source, translation, reference)
        """
        np.random.seed(seed)

        num_samples = max(
            min_samples,
            min(max_samples, int(len(sources) * sample_rate))
        )

        indices = np.random.choice(len(sources), size=num_samples, replace=False)
        indices = sorted(indices)

        samples = []
        for idx in indices:
            ref = references[idx] if references else None
            samples.append((
                int(idx),
                sources[idx],
                translations[idx],
                ref
            ))

        return samples

    def assess_quality_tier(
        self,
        translations: List[str],
        references: List[str],
        threshold_high: float = 40.0,
        threshold_medium: float = 25.0
    ) -> Dict:
        """
        Assess translation quality and categorize into tiers.

        Args:
            translations: Translated texts
            references: Reference translations
            threshold_high: BLEU threshold for high quality
            threshold_medium: BLEU threshold for medium quality

        Returns:
            Dictionary with quality assessment
        """
        metrics = self.calculate_all_metrics(translations, references)
        bleu_score = metrics.get('corpus_bleu', 0)

        if bleu_score >= threshold_high:
            quality_tier = 'high'
            description = 'High quality - suitable for direct use'
        elif bleu_score >= threshold_medium:
            quality_tier = 'medium'
            description = 'Medium quality - may need post-editing'
        else:
            quality_tier = 'low'
            description = 'Low quality - requires significant post-editing'

        return {
            'quality_tier': quality_tier,
            'description': description,
            'bleu_score': bleu_score,
            'all_metrics': metrics,
            'num_samples': len(translations)
        }
