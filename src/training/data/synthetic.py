"""Synthetic data creation from API translations."""

from typing import List, Tuple, Optional, Dict, Any
from tqdm import tqdm
import time


class SyntheticDataCreator:
    """
    Create synthetic parallel corpus from API translations.

    Used for bootstrapping training data when no seed corpus exists.
    """

    def __init__(
        self,
        api_translator,
        source_lang: str,
        target_lang: str,
        rate_limit_delay: float = 0.0
    ):
        """
        Initialize synthetic data creator.

        Args:
            api_translator: API translator instance (from universal_translate)
            source_lang: Source language code
            target_lang: Target language code
            rate_limit_delay: Delay between API calls (seconds)
        """
        self.api_translator = api_translator
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.rate_limit_delay = rate_limit_delay

    def create_parallel_corpus(
        self,
        source_texts: List[str],
        show_progress: bool = True,
        quality_filter: bool = False,
        min_quality_score: float = 0.7
    ) -> List[Tuple[str, str]]:
        """
        Create parallel corpus by translating source texts.

        Args:
            source_texts: List of source texts
            show_progress: Show progress bar
            quality_filter: Filter low-quality translations
            min_quality_score: Minimum quality score (if filtering)

        Returns:
            List of (source, target) pairs
        """
        print(f"\nCreating synthetic corpus: {self.source_lang} → {self.target_lang}")
        print(f"Source texts: {len(source_texts):,}")

        pairs = []
        failed = 0

        iterator = tqdm(source_texts, desc="Translating") if show_progress else source_texts

        for source_text in iterator:
            try:
                # Translate
                translation = self.api_translator.translate(
                    source_text,
                    source_lang=self.source_lang,
                    target_lang=self.target_lang
                )

                # Quality filtering (simple length-based heuristic)
                if quality_filter:
                    quality_score = self._estimate_quality(source_text, translation)
                    if quality_score < min_quality_score:
                        continue

                pairs.append((source_text, translation))

                # Rate limiting
                if self.rate_limit_delay > 0:
                    time.sleep(self.rate_limit_delay)

            except Exception as e:
                failed += 1
                if show_progress:
                    iterator.set_postfix({"failed": failed})
                continue

        print(f"\n✓ Created {len(pairs):,} pairs")
        if failed > 0:
            print(f"  Failed: {failed:,} translations")

        return pairs

    def _estimate_quality(self, source: str, target: str) -> float:
        """
        Simple quality estimation based on length ratio.

        Args:
            source: Source text
            target: Target text

        Returns:
            Quality score (0-1)
        """
        if not source or not target:
            return 0.0

        source_len = len(source.split())
        target_len = len(target.split())

        # Length ratio should be between 0.5 and 2.0 for good translations
        ratio = target_len / max(source_len, 1)

        if ratio < 0.5 or ratio > 2.0:
            return 0.0
        elif 0.8 <= ratio <= 1.2:
            return 1.0
        else:
            # Penalize based on distance from ideal ratio
            distance = min(abs(ratio - 0.8), abs(ratio - 1.2))
            return max(0.5, 1.0 - distance)

    @staticmethod
    def save_to_tsv(
        pairs: List[Tuple[str, str]],
        output_path: str
    ):
        """
        Save parallel corpus to TSV file.

        Args:
            pairs: List of (source, target) pairs
            output_path: Output file path
        """
        from pathlib import Path
        import csv

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(["source", "target"])

            for source, target in pairs:
                writer.writerow([source, target])

        print(f"✓ Saved {len(pairs):,} pairs to {output_path}")

    @staticmethod
    def save_to_jsonl(
        pairs: List[Tuple[str, str]],
        output_path: str
    ):
        """
        Save parallel corpus to JSONL file.

        Args:
            pairs: List of (source, target) pairs
            output_path: Output file path
        """
        from pathlib import Path
        import json

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            for source, target in pairs:
                obj = {"source": source, "target": target}
                f.write(json.dumps(obj, ensure_ascii=False) + '\n')

        print(f"✓ Saved {len(pairs):,} pairs to {output_path}")
