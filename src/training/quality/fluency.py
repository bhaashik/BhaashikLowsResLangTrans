"""
Fluency estimation component.

Measures how fluent/natural the translation is in the target language.
Inspired by N-gram language models in SMT, but using modern approaches.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict
import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)


class AbstractFluencyEstimator(ABC):
    """
    Abstract base class for fluency estimators.

    Fluency measures how natural/fluent the translation is,
    independent of the source.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        batch_size: int = 32,
        cache: bool = True,
    ):
        """
        Initialize fluency estimator.

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
        self._score_cache: Dict[str, float] = {}
        self._initialized = False

    @abstractmethod
    def initialize(self):
        """Initialize model and resources."""
        pass

    @abstractmethod
    def estimate_fluency(
        self,
        texts: List[str],
    ) -> np.ndarray:
        """
        Estimate fluency scores for texts.

        Args:
            texts: Sentences to score

        Returns:
            Fluency scores [batch_size]
        """
        pass

    def estimate_single(self, text: str) -> float:
        """
        Estimate fluency for single sentence.

        Args:
            text: Sentence to score

        Returns:
            Fluency score
        """
        # Check cache
        if self.cache and text in self._score_cache:
            return self._score_cache[text]

        # Compute score
        scores = self.estimate_fluency([text])
        score = float(scores[0])

        # Cache
        if self.cache:
            self._score_cache[text] = score

        return score

    def cleanup(self):
        """Cleanup resources."""
        self._score_cache.clear()


class PerplexityFluency(AbstractFluencyEstimator):
    """
    Fluency estimation using language model perplexity.

    Lower perplexity = higher fluency.
    Inspired by N-gram LM in SMT.
    """

    def __init__(
        self,
        model_name: str = "gpt2",
        device: str = "cuda",
        batch_size: int = 32,
        cache: bool = True,
        max_length: int = 512,
    ):
        """
        Initialize perplexity fluency estimator.

        Args:
            model_name: Language model name
            device: Device to run on
            batch_size: Batch size for inference
            cache: Whether to cache scores
            max_length: Maximum sequence length
        """
        super().__init__(model_name, device, batch_size, cache)
        self.max_length = max_length
        self.model = None
        self.tokenizer = None

    def initialize(self):
        """Initialize language model."""
        if self._initialized:
            return

        logger.info(f"Initializing PerplexityFluency with {self.model_name}...")

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name
            ).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            # Set padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model.eval()
            logger.info("✓ Language model loaded")
        except Exception as e:
            raise RuntimeError(f"Failed to load language model: {e}")

        self._initialized = True

    def estimate_fluency(
        self,
        texts: List[str],
    ) -> np.ndarray:
        """
        Estimate fluency using perplexity.

        Args:
            texts: Sentences to score

        Returns:
            Fluency scores (normalized, higher=better) [batch_size]
        """
        if not self._initialized:
            self.initialize()

        perplexities = []

        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]

            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            ).to(self.device)

            # Compute perplexity
            with torch.no_grad():
                outputs = self.model(**inputs, labels=inputs.input_ids)
                loss = outputs.loss

                # Perplexity = exp(loss)
                # For batch, compute per-example perplexity
                batch_perplexities = self._compute_batch_perplexity(
                    inputs.input_ids,
                    inputs.attention_mask
                )

            perplexities.extend(batch_perplexities)

        perplexities = np.array(perplexities)

        # Convert to fluency score (lower perplexity = higher fluency)
        # Use negative log perplexity and normalize to [0, 1]
        fluency_scores = self._perplexity_to_fluency(perplexities)

        return fluency_scores

    def _compute_batch_perplexity(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> List[float]:
        """Compute per-example perplexity."""
        perplexities = []

        for i in range(input_ids.size(0)):
            ids = input_ids[i:i+1]
            mask = attention_mask[i:i+1]

            # Get logits
            with torch.no_grad():
                outputs = self.model(ids, attention_mask=mask)
                logits = outputs.logits

            # Shift for next token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = ids[..., 1:].contiguous()
            shift_mask = mask[..., 1:].contiguous()

            # Compute loss
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            losses = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )

            # Average over valid tokens
            losses = losses.view(shift_labels.size())
            masked_losses = losses * shift_mask
            avg_loss = masked_losses.sum() / shift_mask.sum()

            # Perplexity
            perplexity = torch.exp(avg_loss).item()
            perplexities.append(perplexity)

        return perplexities

    def _perplexity_to_fluency(self, perplexities: np.ndarray) -> np.ndarray:
        """Convert perplexity to fluency score in [0, 1]."""
        # Use negative log perplexity
        log_perplexities = np.log(perplexities + 1e-10)

        # Normalize to [0, 1] using sigmoid-like function
        # Lower perplexity -> higher fluency
        fluency = 1.0 / (1.0 + np.exp(log_perplexities / 10.0))

        return fluency


class ParseBasedFluency(AbstractFluencyEstimator):
    """
    Fluency estimation using parse tree metrics.

    Uses syntactic well-formedness as fluency indicator.
    Inspired by syntactic language models.
    """

    def __init__(
        self,
        model_name: str = "stanza",
        language: str = "en",
        device: str = "cuda",
        batch_size: int = 32,
        cache: bool = True,
    ):
        """
        Initialize parse-based fluency estimator.

        Args:
            model_name: Parser name
            language: Target language code
            device: Device to run on
            batch_size: Batch size for parsing
            cache: Whether to cache scores
        """
        super().__init__(model_name, device, batch_size, cache)
        self.language = language
        self.parser = None

    def initialize(self):
        """Initialize dependency parser."""
        if self._initialized:
            return

        logger.info(f"Initializing ParseBasedFluency with {self.model_name}...")

        from src.training.linguistic.parsers.base import create_parser

        self.parser = create_parser(
            self.model_name,
            self.language,
            use_gpu=(self.device == "cuda"),
            batch_size=self.batch_size
        )

        logger.info("✓ Parser loaded for fluency estimation")
        self._initialized = True

    def estimate_fluency(
        self,
        texts: List[str],
    ) -> np.ndarray:
        """
        Estimate fluency using parse tree metrics.

        Args:
            texts: Sentences to score

        Returns:
            Fluency scores [batch_size]
        """
        if not self._initialized:
            self.initialize()

        # Parse texts
        parses = self.parser.parse(texts)

        # Compute fluency from parse features
        fluency_scores = []
        for parse in parses:
            score = self._compute_parse_fluency(parse)
            fluency_scores.append(score)

        return np.array(fluency_scores)

    def _compute_parse_fluency(self, parse) -> float:
        """
        Compute fluency from parse tree.

        Metrics:
        - Tree well-formedness (all words have heads)
        - Tree depth (not too deep, not too shallow)
        - Dependency label distribution
        - Root identification
        """
        if len(parse.words) == 0:
            return 0.0

        score = 0.0
        n_metrics = 0

        # 1. Tree completeness (all words except root should have non-zero head)
        non_root_words = [h for h in parse.heads if h != 0]
        if len(non_root_words) > 0:
            completeness = len(non_root_words) / len(parse.words)
            score += completeness
            n_metrics += 1

        # 2. Single root (exactly one word with head=0)
        n_roots = parse.heads.count(0)
        if n_roots == 1:
            score += 1.0
        else:
            score += 1.0 / (1.0 + abs(n_roots - 1))
        n_metrics += 1

        # 3. Tree depth (reasonable depth)
        depth = parse.get_tree_depth()
        ideal_depth = np.log(len(parse.words)) + 1  # Log-scale with sentence length
        depth_score = 1.0 / (1.0 + abs(depth - ideal_depth))
        score += depth_score
        n_metrics += 1

        # 4. Dependency label diversity (not all same label)
        if parse.deprels:
            unique_labels = len(set(parse.deprels))
            diversity = min(unique_labels / len(parse.deprels), 1.0)
            score += diversity
            n_metrics += 1

        # Average score
        fluency = score / n_metrics if n_metrics > 0 else 0.0

        return fluency


def create_fluency_estimator(
    method: str,
    model_name: Optional[str] = None,
    language: Optional[str] = None,
    device: str = "cuda",
    batch_size: int = 32,
    cache: bool = True,
) -> AbstractFluencyEstimator:
    """
    Create fluency estimator by method name.

    Args:
        method: Estimation method ("perplexity", "parse_based")
        model_name: Model name (uses default if None)
        language: Target language (for parse_based)
        device: Device to run on
        batch_size: Batch size
        cache: Whether to cache scores

    Returns:
        Fluency estimator instance
    """
    if method == "perplexity":
        model_name = model_name or "gpt2"
        estimator = PerplexityFluency(
            model_name=model_name,
            device=device,
            batch_size=batch_size,
            cache=cache,
        )
    elif method == "parse_based":
        model_name = model_name or "stanza"
        if language is None:
            raise ValueError("language required for parse_based fluency")
        estimator = ParseBasedFluency(
            model_name=model_name,
            language=language,
            device=device,
            batch_size=batch_size,
            cache=cache,
        )
    else:
        raise ValueError(f"Unknown fluency method: {method}")

    estimator.initialize()
    return estimator
