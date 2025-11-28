"""Data models for translation requests and responses."""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum


class TranslationStatus(Enum):
    """Status of a translation request."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class TranslationUnit:
    """A single unit of text to be translated."""
    text: str
    index: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate translation unit."""
        if not isinstance(self.text, str):
            raise ValueError(f"Text must be string, got {type(self.text)}")
        if not isinstance(self.index, int) or self.index < 0:
            raise ValueError(f"Index must be non-negative integer, got {self.index}")


@dataclass
class TranslationRequest:
    """Request for translation."""
    units: List[TranslationUnit]
    src_lang: str
    tgt_lang: str
    prompt_template: Optional[str] = None
    examples: Optional[List[Dict[str, str]]] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate request."""
        if not self.units:
            raise ValueError("Translation request must have at least one unit")
        if not self.src_lang or not self.tgt_lang:
            raise ValueError("Source and target languages are required")

    @property
    def num_units(self) -> int:
        """Number of translation units."""
        return len(self.units)

    @property
    def total_chars(self) -> int:
        """Total character count."""
        return sum(len(unit.text) for unit in self.units)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "units": [{"text": u.text, "index": u.index, "metadata": u.metadata}
                     for u in self.units],
            "src_lang": self.src_lang,
            "tgt_lang": self.tgt_lang,
            "prompt_template": self.prompt_template,
            "examples": self.examples,
            "parameters": self.parameters,
            "metadata": self.metadata
        }


@dataclass
class TranslationResult:
    """Result of translating a single unit."""
    source: str
    translation: str
    index: int
    cost: float = 0.0
    quality_score: Optional[float] = None
    status: TranslationStatus = TranslationStatus.COMPLETED
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "source": self.source,
            "translation": self.translation,
            "index": self.index,
            "cost": self.cost,
            "quality_score": self.quality_score,
            "status": self.status.value,
            "error": self.error,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class TranslationResponse:
    """Response from translation provider."""
    results: List[TranslationResult]
    total_cost: float = 0.0
    provider: str = ""
    model: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def num_results(self) -> int:
        """Number of results."""
        return len(self.results)

    @property
    def successful_count(self) -> int:
        """Number of successful translations."""
        return sum(1 for r in self.results
                  if r.status == TranslationStatus.COMPLETED)

    @property
    def failed_count(self) -> int:
        """Number of failed translations."""
        return sum(1 for r in self.results
                  if r.status == TranslationStatus.FAILED)

    @property
    def avg_quality(self) -> Optional[float]:
        """Average quality score."""
        scores = [r.quality_score for r in self.results
                 if r.quality_score is not None]
        return sum(scores) / len(scores) if scores else None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "results": [r.to_dict() for r in self.results],
            "total_cost": self.total_cost,
            "provider": self.provider,
            "model": self.model,
            "successful_count": self.successful_count,
            "failed_count": self.failed_count,
            "avg_quality": self.avg_quality,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class CostEstimate:
    """Cost estimate for translation."""
    input_cost: float
    output_cost: float
    total_cost: float
    currency: str = "INR"
    num_units: int = 0
    estimated_tokens: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "input_cost": self.input_cost,
            "output_cost": self.output_cost,
            "total_cost": self.total_cost,
            "currency": self.currency,
            "num_units": self.num_units,
            "estimated_tokens": self.estimated_tokens,
            "cost_per_unit": self.total_cost / self.num_units if self.num_units > 0 else 0,
            "metadata": self.metadata
        }
