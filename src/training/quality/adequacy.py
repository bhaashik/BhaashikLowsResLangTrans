"""
Adequacy estimation component.

Measures how well the translation preserves the meaning of the source.
Inspired by IBM model adequacy in SMT, but using modern semantic similarity.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Tuple
import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)


class AbstractAdequacyEstimator(ABC):
    """
    Abstract base class for adequacy estimators.

    Adequacy measures how well the translation preserves the meaning
    of the source sentence.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        batch_size: int = 32,
        cache: bool = True,
    ):
        """
        Initialize adequacy estimator.

        Args:
            model_name: Name of model to use
            device: Device to run on
            batch_size: Batch size for processing
            cache: Whether to cache scores
        """
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.cache = cache
        self._score_cache: Dict[Tuple[str, str], float] = {}
        self._initialized = False

    @abstractmethod
    def initialize(self):
        """Initialize model and resources."""
        pass

    @abstractmethod
    def estimate_adequacy(
        self,
        source_texts: List[str],
        translation_texts: List[str],
    ) -> np.ndarray:
        """
        Estimate adequacy scores for translations.

        Args:
            source_texts: Source sentences
            translation_texts: Translation sentences

        Returns:
            Adequacy scores [batch_size]
        """
        pass

    def estimate_single(
        self,
        source_text: str,
        translation_text: str,
    ) -> float:
        """
        Estimate adequacy for single pair.

        Args:
            source_text: Source sentence
            translation_text: Translation sentence

        Returns:
            Adequacy score
        """
        # Check cache
        cache_key = (source_text, translation_text)
        if self.cache and cache_key in self._score_cache:
            return self._score_cache[cache_key]

        # Compute score
        scores = self.estimate_adequacy([source_text], [translation_text])
        score = float(scores[0])

        # Cache
        if self.cache:
            self._score_cache[cache_key] = score

        return score

    def cleanup(self):
        """Cleanup resources."""
        self._score_cache.clear()


class SentenceEmbeddingAdequacy(AbstractAdequacyEstimator):
    """
    Adequacy estimation using sentence embeddings.

    Uses cosine similarity between source and translation embeddings
    as adequacy score.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/LaBSE",
        device: str = "cuda",
        batch_size: int = 32,
        cache: bool = True,
    ):
        """
        Initialize sentence embedding adequacy estimator.

        Args:
            model_name: Sentence transformer model name
            device: Device to run on
            batch_size: Batch size for encoding
            cache: Whether to cache scores
        """
        super().__init__(model_name, device, batch_size, cache)
        self.model = None
        self.tokenizer = None

    def initialize(self):
        """Initialize sentence transformer model."""
        if self._initialized:
            return

        logger.info(f"Initializing SentenceEmbeddingAdequacy with {self.model_name}...")

        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name, device=self.device)
            logger.info("✓ Sentence transformer loaded")
        except ImportError:
            raise ImportError(
                "sentence-transformers required. Install with: "
                "pip install sentence-transformers"
            )

        self._initialized = True

    def estimate_adequacy(
        self,
        source_texts: List[str],
        translation_texts: List[str],
    ) -> np.ndarray:
        """
        Estimate adequacy using sentence embeddings.

        Args:
            source_texts: Source sentences
            translation_texts: Translation sentences

        Returns:
            Adequacy scores (cosine similarities) [batch_size]
        """
        if not self._initialized:
            self.initialize()

        # Encode sentences
        source_embeddings = self.model.encode(
            source_texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        translation_embeddings = self.model.encode(
            translation_texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        # Compute cosine similarity (embeddings are already normalized)
        similarities = np.sum(source_embeddings * translation_embeddings, axis=1)

        # Clip to [0, 1] range
        similarities = np.clip(similarities, 0.0, 1.0)

        return similarities


class EntailmentAdequacy(AbstractAdequacyEstimator):
    """
    Adequacy estimation using entailment model.

    Uses entailment probability as adequacy score.
    Good translation should be entailed by source (in cross-lingual setting).
    """

    def __init__(
        self,
        model_name: str = "microsoft/mdeberta-v3-base-xnli",
        device: str = "cuda",
        batch_size: int = 32,
        cache: bool = True,
    ):
        """
        Initialize entailment adequacy estimator.

        Args:
            model_name: Entailment model name (XNLI)
            device: Device to run on
            batch_size: Batch size for inference
            cache: Whether to cache scores
        """
        super().__init__(model_name, device, batch_size, cache)
        self.model = None
        self.tokenizer = None

    def initialize(self):
        """Initialize entailment model."""
        if self._initialized:
            return

        logger.info(f"Initializing EntailmentAdequacy with {self.model_name}...")

        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name
            ).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model.eval()
            logger.info("✓ Entailment model loaded")
        except Exception as e:
            raise RuntimeError(f"Failed to load entailment model: {e}")

        self._initialized = True

    def estimate_adequacy(
        self,
        source_texts: List[str],
        translation_texts: List[str],
    ) -> np.ndarray:
        """
        Estimate adequacy using entailment.

        Args:
            source_texts: Source sentences (premises)
            translation_texts: Translation sentences (hypotheses)

        Returns:
            Adequacy scores (entailment probabilities) [batch_size]
        """
        if not self._initialized:
            self.initialize()

        scores = []

        # Process in batches
        for i in range(0, len(source_texts), self.batch_size):
            batch_sources = source_texts[i:i + self.batch_size]
            batch_translations = translation_texts[i:i + self.batch_size]

            # Tokenize
            inputs = self.tokenizer(
                batch_sources,
                batch_translations,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)

            # Get entailment probability
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)

                # Get entailment probability (usually index 2 for XNLI models)
                # Label mapping: 0=contradiction, 1=neutral, 2=entailment
                entailment_probs = probs[:, 2].cpu().numpy()

            scores.extend(entailment_probs)

        return np.array(scores)


class WordAlignmentAdequacy(AbstractAdequacyEstimator):
    """
    Adequacy estimation using word alignment.

    Uses word alignment coverage as adequacy score.
    Inspired by IBM models in SMT.
    """

    def __init__(
        self,
        model_name: str = "bert-base-multilingual-cased",
        device: str = "cuda",
        batch_size: int = 32,
        cache: bool = True,
    ):
        """
        Initialize word alignment adequacy estimator.

        Args:
            model_name: Model for word embeddings
            device: Device to run on
            batch_size: Batch size for processing
            cache: Whether to cache scores
        """
        super().__init__(model_name, device, batch_size, cache)
        self.model = None
        self.tokenizer = None

    def initialize(self):
        """Initialize model for word alignment."""
        if self._initialized:
            return

        logger.info(f"Initializing WordAlignmentAdequacy with {self.model_name}...")

        try:
            from transformers import AutoModel, AutoTokenizer

            self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model.eval()
            logger.info("✓ Word alignment model loaded")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

        self._initialized = True

    def estimate_adequacy(
        self,
        source_texts: List[str],
        translation_texts: List[str],
    ) -> np.ndarray:
        """
        Estimate adequacy using word alignment coverage.

        Args:
            source_texts: Source sentences
            translation_texts: Translation sentences

        Returns:
            Adequacy scores (alignment coverage) [batch_size]
        """
        if not self._initialized:
            self.initialize()

        scores = []

        # Process in batches
        for i in range(0, len(source_texts), self.batch_size):
            batch_sources = source_texts[i:i + self.batch_size]
            batch_translations = translation_texts[i:i + self.batch_size]

            batch_scores = self._compute_alignment_scores(
                batch_sources,
                batch_translations
            )
            scores.extend(batch_scores)

        return np.array(scores)

    def _compute_alignment_scores(
        self,
        sources: List[str],
        translations: List[str],
    ) -> List[float]:
        """Compute alignment-based adequacy scores."""
        scores = []

        for src, tgt in zip(sources, translations):
            # Tokenize
            src_inputs = self.tokenizer(
                src, return_tensors="pt", padding=True
            ).to(self.device)
            tgt_inputs = self.tokenizer(
                tgt, return_tensors="pt", padding=True
            ).to(self.device)

            # Get embeddings
            with torch.no_grad():
                src_outputs = self.model(**src_inputs)
                tgt_outputs = self.model(**tgt_inputs)

                src_embeddings = src_outputs.last_hidden_state[0]  # [src_len, dim]
                tgt_embeddings = tgt_outputs.last_hidden_state[0]  # [tgt_len, dim]

            # Compute alignment matrix (cosine similarity)
            src_norm = torch.nn.functional.normalize(src_embeddings, p=2, dim=1)
            tgt_norm = torch.nn.functional.normalize(tgt_embeddings, p=2, dim=1)
            alignment_matrix = torch.matmul(src_norm, tgt_norm.t())  # [src_len, tgt_len]

            # Compute coverage (max alignment for each source word)
            coverage = torch.max(alignment_matrix, dim=1)[0].mean().item()

            scores.append(coverage)

        return scores


def create_adequacy_estimator(
    method: str,
    model_name: Optional[str] = None,
    device: str = "cuda",
    batch_size: int = 32,
    cache: bool = True,
) -> AbstractAdequacyEstimator:
    """
    Create adequacy estimator by method name.

    Args:
        method: Estimation method ("sentence_embedding", "entailment", "word_alignment")
        model_name: Model name (uses default if None)
        device: Device to run on
        batch_size: Batch size
        cache: Whether to cache scores

    Returns:
        Adequacy estimator instance
    """
    if method == "sentence_embedding":
        model_name = model_name or "sentence-transformers/LaBSE"
        estimator = SentenceEmbeddingAdequacy(
            model_name=model_name,
            device=device,
            batch_size=batch_size,
            cache=cache,
        )
    elif method == "entailment":
        model_name = model_name or "microsoft/mdeberta-v3-base-xnli"
        estimator = EntailmentAdequacy(
            model_name=model_name,
            device=device,
            batch_size=batch_size,
            cache=cache,
        )
    elif method == "word_alignment":
        model_name = model_name or "bert-base-multilingual-cased"
        estimator = WordAlignmentAdequacy(
            model_name=model_name,
            device=device,
            batch_size=batch_size,
            cache=cache,
        )
    else:
        raise ValueError(f"Unknown adequacy method: {method}")

    estimator.initialize()
    return estimator
