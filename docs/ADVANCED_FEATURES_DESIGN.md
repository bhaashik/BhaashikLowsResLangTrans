# Advanced Features Design: Linguistic Integration & RL Enhancement

## Overview

This document outlines the design for Phase 4 integration: advanced linguistic features, language embeddings, SMT-inspired quality components, automatic post-editing, and reinforcement learning for low-resource scenarios.

**Goals**:
1. Integrate linguistic features (dependency parses) into translation
2. Leverage language embeddings for typological awareness
3. Incorporate SMT-inspired adequacy and fluency modeling
4. Implement automatic post-editing (APE) based on quality estimation
5. Add reinforcement learning for low-resource optimization with overfitting prevention

**Key Principles**:
- **Configurability**: All features optional and configurable
- **Modularity**: Each component works independently and can be combined
- **Extensibility**: Easy to add new parsers, embeddings, metrics
- **Research-friendly**: Support for experimental features and ablation studies

---

## 1. Linguistic Features Module

### 1.1 Motivation

Dependency parse information provides:
- **Syntactic structure**: Word-level relationships and grammatical functions
- **Reordering guidance**: Structural differences between source/target languages
- **Context enrichment**: Head-dependent relationships for ambiguity resolution
- **Quality estimation**: Syntactic well-formedness of translations

### 1.2 Architecture

```
src/training/linguistic/
├── __init__.py
├── parsers/
│   ├── base.py              # AbstractParser interface
│   ├── stanza_parser.py     # Stanza (UD-based)
│   ├── spacy_parser.py      # spaCy parser
│   ├── trankit_parser.py    # Trankit (multilingual)
│   └── udpipe_parser.py     # UDPipe
├── features/
│   ├── dependency_encoder.py    # Encode parse trees as features
│   ├── structural_features.py   # Extract structural statistics
│   └── alignment_features.py    # Source-target alignment based on parses
├── integration/
│   ├── attention_injection.py   # Inject parse into attention
│   ├── encoder_augmentation.py  # Augment encoder with parse features
│   └── decoder_constraint.py    # Constrain decoder with target parse
└── config.py                # Linguistic features configuration
```

### 1.3 Configuration

```python
@dataclass
class LinguisticFeaturesConfig:
    """Configuration for linguistic features."""

    # Enable/disable features
    use_source_parse: bool = False
    use_target_parse: bool = False
    use_both: bool = False

    # Parser selection
    parser: str = "stanza"  # stanza, spacy, trankit, udpipe
    parser_batch_size: int = 32

    # Feature types
    features: List[str] = field(default_factory=lambda: [
        "dependency_labels",  # DEPREL tags
        "pos_tags",           # UPOS tags
        "tree_depth",         # Depth in parse tree
        "head_distance",      # Distance to head
    ])

    # Integration method
    integration_method: str = "encoder_augmentation"
    # Options: encoder_augmentation, attention_injection, decoder_constraint

    # Feature encoding
    encoding_dim: int = 128
    use_graph_encoder: bool = True  # Use GNN for parse encoding

    # Custom parameters
    custom_params: Dict[str, Any] = field(default_factory=dict)
```

### 1.4 Parser Interface

```python
class AbstractParser(ABC):
    """Abstract interface for dependency parsers."""

    @abstractmethod
    def parse(self, texts: List[str], language: str) -> List[ParseTree]:
        """Parse texts and return dependency trees."""
        pass

    @abstractmethod
    def supports_language(self, language: str) -> bool:
        """Check if parser supports language."""
        pass

@dataclass
class ParseTree:
    """Dependency parse tree."""
    words: List[str]
    heads: List[int]        # Head indices (0 = root)
    deprels: List[str]      # Dependency relations
    pos_tags: List[str]     # POS tags
    lemmas: List[str]       # Lemmas
    features: Dict[str, Any]  # Additional features
```

### 1.5 Feature Extraction

```python
class DependencyEncoder:
    """Encode dependency parse as features for MT model."""

    def encode_parse_tree(
        self,
        parse: ParseTree,
        encoding_dim: int = 128
    ) -> torch.Tensor:
        """
        Encode parse tree as feature vectors.

        Returns:
            Tensor of shape [seq_len, encoding_dim]
        """
        # Extract features: deprel, pos, tree depth, head distance
        features = []
        for i in range(len(parse.words)):
            feature_vec = self._extract_word_features(parse, i)
            features.append(feature_vec)

        # Optionally use GNN to encode tree structure
        if self.use_graph_encoder:
            features = self._apply_graph_encoder(features, parse)

        return torch.stack(features)
```

### 1.6 Integration Strategies

**Strategy 1: Encoder Augmentation**
- Concatenate parse features with word embeddings
- `[word_emb; parse_features] → Encoder`

**Strategy 2: Attention Injection**
- Use parse structure to bias attention weights
- Heads attend more to their dependents

**Strategy 3: Decoder Constraint**
- Use target-side parse to constrain beam search
- Prefer syntactically well-formed outputs

### 1.7 Implementation Plan

1. Create base parser interface and Stanza implementation
2. Implement dependency encoder with GNN option
3. Add encoder augmentation integration (simplest)
4. Add attention injection (more sophisticated)
5. Add decoder constraints (most complex)
6. Create configuration and CLI support

---

## 2. Language Embeddings Module

### 2.1 Motivation

Language embeddings capture typological features:
- **Linguistic typology**: Word order, morphology, syntax
- **Genetic relationships**: Language families and similarity
- **Geographic factors**: Regional linguistic phenomena
- **Resource availability**: Training data quantity/quality

**Use cases**:
- Language-aware model adaptation
- Transfer learning guidance
- Quality estimation improvement
- Multilingual model conditioning

### 2.2 Data Sources

1. **URIEL Database** (CMU): 103 typological features, 4,005 languages
   - Syntax features (word order, case marking)
   - Phonology features (tone, stress)
   - Inventory features (vowels, consonants)

2. **lang2vec** (ACL 2019):
   - Aggregates URIEL + Ethnologue + Wikipedia
   - Pre-computed embeddings for 7,000+ languages

3. **WALS** (World Atlas of Language Structures):
   - 192 typological features
   - Expert-curated data

4. **Custom embeddings**:
   - Learned from multilingual data
   - Task-specific optimization

### 2.3 Architecture

```
src/training/embeddings/
├── __init__.py
├── sources/
│   ├── uriel.py             # URIEL database loader
│   ├── lang2vec_loader.py   # lang2vec embeddings
│   ├── wals.py              # WALS features
│   └── custom.py            # Custom learned embeddings
├── encoder.py               # Language embedding encoder
├── integration.py           # Integration with MT models
├── similarity.py            # Language similarity metrics
└── config.py                # Language embedding configuration
```

### 2.4 Configuration

```python
@dataclass
class LanguageEmbeddingConfig:
    """Configuration for language embeddings."""

    # Enable embeddings
    use_source_embedding: bool = True
    use_target_embedding: bool = True

    # Embedding source
    source: str = "lang2vec"  # uriel, lang2vec, wals, custom, combined

    # Embedding dimensions
    embedding_dim: int = 256

    # Integration method
    integration_method: str = "adapter_conditioning"
    # Options: adapter_conditioning, attention_bias, encoder_init, decoder_init

    # Features to use (for URIEL/WALS)
    features: List[str] = field(default_factory=lambda: [
        "syntax",
        "phonology",
        "inventory",
        "family",
    ])

    # Similarity-based transfer
    use_transfer_learning: bool = True
    similarity_threshold: float = 0.7

    # Custom parameters
    custom_params: Dict[str, Any] = field(default_factory=dict)
```

### 2.5 Language Embedding Encoder

```python
class LanguageEmbeddingEncoder:
    """Encode language typology as embeddings."""

    def __init__(self, config: LanguageEmbeddingConfig):
        self.config = config
        self.source = self._load_source()

    def get_embedding(self, language: str) -> torch.Tensor:
        """
        Get language embedding vector.

        Args:
            language: ISO 639-3 code (e.g., 'hin', 'bho')

        Returns:
            Embedding tensor of shape [embedding_dim]
        """
        if self.config.source == "lang2vec":
            return self._get_lang2vec(language)
        elif self.config.source == "uriel":
            return self._get_uriel(language)
        elif self.config.source == "combined":
            return self._get_combined(language)

    def get_language_similarity(
        self,
        lang1: str,
        lang2: str
    ) -> float:
        """Compute similarity between two languages."""
        emb1 = self.get_embedding(lang1)
        emb2 = self.get_embedding(lang2)
        return F.cosine_similarity(emb1, emb2, dim=0).item()
```

### 2.6 Integration with MT Models

**Method 1: Adapter Conditioning**
```python
# Add language-specific adapter layers
adapter = LanguageAdapter(language_embedding)
hidden = adapter(encoder_output)
```

**Method 2: Attention Bias**
```python
# Bias attention based on language similarity
attention_weights = attention_weights + language_bias
```

**Method 3: Encoder/Decoder Initialization**
```python
# Initialize model state with language embedding
encoder_state = encoder_state + language_embedding
```

### 2.7 Transfer Learning Support

```python
def select_transfer_languages(
    target_lang: str,
    available_languages: List[str],
    min_similarity: float = 0.7
) -> List[str]:
    """
    Select languages for transfer learning based on similarity.

    Use case: Train on high-resource similar languages, transfer to target.
    """
    similarities = [
        (lang, self.get_language_similarity(target_lang, lang))
        for lang in available_languages
    ]

    transfer_langs = [
        lang for lang, sim in similarities
        if sim >= min_similarity
    ]

    return sorted(transfer_langs, key=lambda l: similarities[l], reverse=True)
```

---

## 3. SMT-Inspired Quality Components

### 3.1 Motivation

Statistical MT decomposed translation quality into:
1. **Adequacy**: Meaning preservation (IBM models via word alignment)
2. **Fluency**: Target language naturalness (n-gram LM)

In neural MT, we adapt these concepts:
- **Adequacy** → Semantic similarity (embeddings, entailment)
- **Fluency** → Syntactic well-formedness (parse-based LM, perplexity)

### 3.2 Architecture

```
src/training/quality/
├── __init__.py
├── adequacy/
│   ├── semantic_similarity.py   # Embedding-based similarity
│   ├── entailment.py            # NLI-based adequacy
│   ├── word_alignment.py        # Attention-based alignment
│   └── cross_lingual.py         # Cross-lingual embeddings
├── fluency/
│   ├── syntactic_lm.py          # Parse-tree LM
│   ├── perplexity.py            # Standard LM perplexity
│   ├── grammaticality.py        # Grammar checker
│   └── dependency_coherence.py  # Parse coherence score
├── combined/
│   ├── decomposed_scorer.py     # Adequacy + Fluency
│   └── learned_weights.py       # Learn component weights
└── config.py                    # Quality configuration
```

### 3.3 Adequacy Measurement

**Approach 1: Semantic Similarity**
```python
class SemanticAdequacyScorer:
    """Measure adequacy via semantic similarity."""

    def __init__(self, model: str = "labse"):
        # Use LaBSE (Language-agnostic BERT Sentence Embedding)
        # or SONAR, Laser2, etc.
        self.encoder = SentenceEncoder(model)

    def score(
        self,
        source: str,
        translation: str,
        source_lang: str,
        target_lang: str
    ) -> float:
        """
        Score adequacy as cosine similarity in shared space.

        Returns:
            Score in [0, 1], higher = more adequate
        """
        src_emb = self.encoder.encode(source, source_lang)
        tgt_emb = self.encoder.encode(translation, target_lang)

        similarity = F.cosine_similarity(src_emb, tgt_emb, dim=0)
        return similarity.item()
```

**Approach 2: Cross-Lingual Entailment**
```python
class EntailmentAdequacyScorer:
    """Measure adequacy via NLI entailment."""

    def __init__(self, model: str = "xnli"):
        self.nli_model = CrossLingualNLI(model)

    def score(
        self,
        source: str,
        translation: str,
        source_lang: str,
        target_lang: str
    ) -> float:
        """
        Score adequacy as P(entailment | source, translation).

        High entailment = good adequacy (meaning preserved).
        """
        prob_entailment = self.nli_model.predict(
            premise=source,
            hypothesis=translation,
            premise_lang=source_lang,
            hypothesis_lang=target_lang
        )
        return prob_entailment
```

**Approach 3: Word Alignment Coverage**
```python
class AlignmentAdequacyScorer:
    """Measure adequacy via word alignment coverage."""

    def score(
        self,
        source: str,
        translation: str,
        attention_weights: torch.Tensor  # From MT model
    ) -> float:
        """
        Score adequacy as alignment coverage.

        Measures: Are all source words adequately translated?
        """
        # Compute coverage of source words
        coverage = attention_weights.max(dim=1).values.mean()
        return coverage.item()
```

### 3.4 Fluency Measurement

**Approach 1: Syntactic Language Model**
```python
class SyntacticFluencyScorer:
    """Measure fluency via parse-tree LM."""

    def __init__(self, parser: AbstractParser):
        self.parser = parser
        self.tree_lm = TreeLanguageModel()  # RNNG or similar

    def score(
        self,
        text: str,
        language: str
    ) -> float:
        """
        Score fluency as parse tree probability.

        Returns:
            Log probability of syntactic structure
        """
        parse = self.parser.parse([text], language)[0]
        log_prob = self.tree_lm.score_tree(parse)
        return log_prob
```

**Approach 2: Standard LM Perplexity**
```python
class PerplexityFluencyScorer:
    """Measure fluency via LM perplexity."""

    def __init__(self, model: str = "gpt2-large"):
        self.lm = LanguageModel(model)

    def score(self, text: str, language: str) -> float:
        """
        Score fluency as negative perplexity.

        Lower perplexity = more fluent.
        """
        perplexity = self.lm.compute_perplexity(text, language)
        # Convert to score (higher = better)
        return -math.log(perplexity)
```

**Approach 3: Grammaticality Checker**
```python
class GrammaticalityScorer:
    """Measure fluency via grammatical error detection."""

    def __init__(self):
        self.grammar_checker = GrammarChecker()

    def score(self, text: str, language: str) -> float:
        """
        Score fluency as 1 - (error_rate).

        Returns:
            Score in [0, 1], 1 = no errors
        """
        errors = self.grammar_checker.check(text, language)
        error_rate = len(errors) / max(len(text.split()), 1)
        return max(0, 1 - error_rate)
```

### 3.5 Combined Quality Scorer

```python
class DecomposedQualityScorer:
    """Combine adequacy and fluency for overall quality."""

    def __init__(
        self,
        adequacy_scorer: AbstractAdequacyScorer,
        fluency_scorer: AbstractFluencyScorer,
        adequacy_weight: float = 0.5,
        fluency_weight: float = 0.5
    ):
        self.adequacy_scorer = adequacy_scorer
        self.fluency_scorer = fluency_scorer
        self.adequacy_weight = adequacy_weight
        self.fluency_weight = fluency_weight

    def score(
        self,
        source: str,
        translation: str,
        source_lang: str,
        target_lang: str
    ) -> Dict[str, float]:
        """
        Compute decomposed quality score.

        Returns:
            {
                'adequacy': float,
                'fluency': float,
                'combined': float
            }
        """
        adequacy = self.adequacy_scorer.score(
            source, translation, source_lang, target_lang
        )
        fluency = self.fluency_scorer.score(translation, target_lang)

        combined = (
            self.adequacy_weight * adequacy +
            self.fluency_weight * fluency
        )

        return {
            'adequacy': adequacy,
            'fluency': fluency,
            'combined': combined
        }
```

---

## 4. Automatic Post-Editing (APE)

### 4.1 Motivation

Automatic Post-Editing improves raw MT output through:
1. **Error correction**: Fix grammatical/semantic errors
2. **Style adaptation**: Match target domain/register
3. **Terminology enforcement**: Ensure consistent terminology
4. **Quality enhancement**: Improve fluency and adequacy

**Trigger**: Quality estimation identifies low-quality translations

### 4.2 Architecture

```
src/training/post_editing/
├── __init__.py
├── models/
│   ├── base.py                  # Abstract APE model
│   ├── seq2seq_ape.py           # Seq2seq APE model
│   ├── llm_ape.py               # LLM-based APE
│   └── rule_based_ape.py        # Rule-based corrections
├── strategies/
│   ├── selective_ape.py         # Apply APE selectively
│   ├── iterative_ape.py         # Iterative refinement
│   └── ensemble_ape.py          # Ensemble multiple APE models
├── quality_aware/
│   ├── trigger.py               # Decide when to apply APE
│   └── confidence.py            # Confidence-based selection
└── config.py                    # APE configuration
```

### 4.3 Configuration

```python
@dataclass
class APEConfig:
    """Configuration for automatic post-editing."""

    # APE model
    ape_model: str = "llm"  # seq2seq, llm, rule_based, ensemble
    model_path: Optional[str] = None

    # Triggering strategy
    trigger_on: str = "quality"  # quality, always, confidence, decomposed
    quality_threshold: float = 0.7  # Apply APE if quality < threshold

    # Decomposed triggering (use adequacy/fluency)
    adequacy_threshold: float = 0.6
    fluency_threshold: float = 0.6
    fix_adequacy: bool = True   # Use APE for adequacy issues
    fix_fluency: bool = True    # Use APE for fluency issues

    # APE strategy
    strategy: str = "selective"  # selective, iterative, ensemble
    max_iterations: int = 3  # For iterative APE

    # LLM-specific
    llm_provider: str = "anthropic"
    llm_model: str = "claude-haiku-4.5"
    llm_temperature: float = 0.3

    # Cost control
    max_cost_per_sample: float = 0.01  # Max $ to spend on APE per sample

    # Custom parameters
    custom_params: Dict[str, Any] = field(default_factory=dict)
```

### 4.4 APE Models

**Model 1: Seq2Seq APE**
```python
class Seq2SeqAPE:
    """Seq2seq post-editing model."""

    def post_edit(
        self,
        source: str,
        mt_output: str,
        quality_info: Optional[Dict] = None
    ) -> str:
        """
        Post-edit MT output.

        Input: (source, mt_output) → APE model → corrected output
        """
        # Prepare input: "src: {source} mt: {mt_output}"
        input_text = f"src: {source} mt: {mt_output}"

        # Generate post-edited output
        edited = self.model.generate(input_text)

        return edited
```

**Model 2: LLM-based APE**
```python
class LLM_APE:
    """LLM-based post-editing using API providers."""

    def post_edit(
        self,
        source: str,
        mt_output: str,
        quality_info: Optional[Dict] = None
    ) -> str:
        """
        Post-edit using LLM with quality-aware prompting.
        """
        # Construct quality-aware prompt
        if quality_info:
            if quality_info['adequacy'] < 0.6:
                instruction = "Fix meaning errors and improve adequacy."
            elif quality_info['fluency'] < 0.6:
                instruction = "Fix grammatical errors and improve fluency."
            else:
                instruction = "Improve overall translation quality."
        else:
            instruction = "Improve translation quality."

        prompt = f"""Given the source text and machine translation, {instruction}

Source: {source}
Machine Translation: {mt_output}

Post-Edited Translation:"""

        response = self.llm_client.generate(prompt)
        return response.strip()
```

**Model 3: Rule-Based APE**
```python
class RuleBasedAPE:
    """Rule-based post-editing for common errors."""

    def __init__(self):
        self.rules = self._load_rules()

    def post_edit(
        self,
        source: str,
        mt_output: str,
        quality_info: Optional[Dict] = None
    ) -> str:
        """
        Apply rule-based corrections.

        Examples:
        - Fix repeated words
        - Correct common grammar errors
        - Enforce terminology
        """
        edited = mt_output

        for rule in self.rules:
            if rule.applies(edited):
                edited = rule.apply(edited)

        return edited
```

### 4.5 Quality-Aware APE Triggering

```python
class QualityAwareAPE:
    """Apply APE selectively based on quality estimation."""

    def should_apply_ape(
        self,
        quality_scores: Dict[str, float],
        config: APEConfig
    ) -> Tuple[bool, str]:
        """
        Decide whether to apply APE and which type.

        Returns:
            (should_apply, reason)
        """
        if config.trigger_on == "always":
            return True, "always"

        if config.trigger_on == "quality":
            if quality_scores['combined'] < config.quality_threshold:
                return True, "low_quality"

        if config.trigger_on == "decomposed":
            if quality_scores['adequacy'] < config.adequacy_threshold:
                return True, "low_adequacy"
            if quality_scores['fluency'] < config.fluency_threshold:
                return True, "low_fluency"

        return False, "quality_sufficient"

    def post_edit_if_needed(
        self,
        source: str,
        mt_output: str,
        quality_scores: Dict[str, float],
        config: APEConfig
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Apply APE if quality is insufficient.

        Returns:
            (final_output, metadata)
        """
        should_apply, reason = self.should_apply_ape(quality_scores, config)

        metadata = {
            'ape_applied': should_apply,
            'ape_reason': reason,
            'original_output': mt_output,
        }

        if should_apply:
            edited_output = self.ape_model.post_edit(
                source, mt_output, quality_scores
            )
            metadata['ape_output'] = edited_output
            return edited_output, metadata
        else:
            return mt_output, metadata
```

### 4.6 Iterative APE

```python
class IterativeAPE:
    """Iteratively refine translation until quality threshold met."""

    def post_edit(
        self,
        source: str,
        mt_output: str,
        max_iterations: int = 3,
        quality_threshold: float = 0.8
    ) -> Tuple[str, List[Dict]]:
        """
        Iteratively improve translation.

        Returns:
            (final_output, iteration_history)
        """
        current = mt_output
        history = []

        for i in range(max_iterations):
            # Estimate quality
            quality = self.quality_estimator.score(source, current)

            history.append({
                'iteration': i,
                'output': current,
                'quality': quality
            })

            # Stop if quality sufficient
            if quality['combined'] >= quality_threshold:
                break

            # Apply APE
            current = self.ape_model.post_edit(source, current, quality)

        return current, history
```

---

## 5. Reinforcement Learning Module

### 5.1 Motivation

Reinforcement Learning optimizes translation quality directly:
- **Reward shaping**: Optimize for BLEU, COMET, decomposed quality
- **Low-resource scenarios**: Learn from minimal feedback
- **Exploration**: Discover better translations than supervised data
- **Overfitting prevention**: Critical for low-resource (Goodhart's Law)

**Key Challenge**: Avoid overfitting to the reward signal (Goodhart's Law: "When a measure becomes a target, it ceases to be a good measure")

### 5.2 Architecture

```
src/training/reinforcement/
├── __init__.py
├── rewards/
│   ├── base.py                  # Abstract reward function
│   ├── bleu_reward.py           # BLEU-based reward
│   ├── comet_reward.py          # COMET-based reward
│   ├── decomposed_reward.py     # Adequacy + Fluency reward
│   └── multi_objective.py       # Multi-objective optimization
├── algorithms/
│   ├── reinforce.py             # REINFORCE algorithm
│   ├── proximal_policy.py       # PPO (Proximal Policy Optimization)
│   ├── minimum_risk.py          # Minimum Risk Training (MRT)
│   └── preference_optimization.py  # DPO/RLHF-style
├── overfitting_prevention/
│   ├── kl_constraint.py         # KL divergence constraint
│   ├── reference_model.py       # Keep reference model
│   ├── early_stopping.py        # Quality-based early stopping
│   ├── reward_diversification.py # Multiple reward objectives
│   └── entropy_regularization.py # Encourage exploration
├── sampling/
│   ├── on_policy.py             # On-policy sampling
│   ├── off_policy.py            # Off-policy sampling
│   └── importance_sampling.py   # Importance-weighted sampling
└── config.py                    # RL configuration
```

### 5.3 Configuration

```python
@dataclass
class RLConfig:
    """Configuration for reinforcement learning."""

    # RL algorithm
    algorithm: str = "ppo"  # reinforce, ppo, mrt, dpo

    # Reward function
    reward_type: str = "decomposed"  # bleu, comet, decomposed, multi_objective
    reward_weights: Dict[str, float] = field(default_factory=lambda: {
        'adequacy': 0.5,
        'fluency': 0.5
    })

    # Overfitting prevention (CRITICAL)
    use_kl_constraint: bool = True
    kl_coefficient: float = 0.1  # Penalty for diverging from reference

    use_reference_model: bool = True  # Keep supervised model as reference

    use_early_stopping: bool = True
    patience: int = 5  # Stop if no improvement

    use_reward_diversification: bool = True  # Multiple reward objectives

    use_entropy_regularization: bool = True
    entropy_coefficient: float = 0.01

    # Training parameters
    num_epochs: int = 5  # Keep small for low-resource
    batch_size: int = 32
    learning_rate: float = 1e-5  # Lower than supervised

    # Sampling
    num_samples: int = 4  # Sample multiple outputs per input
    temperature: float = 1.0

    # Validation
    validation_freq: int = 100  # Validate frequently
    validation_metric: str = "comet"  # Use held-out validation

    # Safety limits
    max_kl_divergence: float = 0.5  # Stop if KL > threshold
    min_bleu_score: float = 10.0  # Stop if BLEU drops too much

    # Custom parameters
    custom_params: Dict[str, Any] = field(default_factory=dict)
```

### 5.4 Reward Functions

```python
class DecomposedReward:
    """Reward based on adequacy + fluency."""

    def __init__(
        self,
        adequacy_scorer: AbstractAdequacyScorer,
        fluency_scorer: AbstractFluencyScorer,
        weights: Dict[str, float]
    ):
        self.adequacy_scorer = adequacy_scorer
        self.fluency_scorer = fluency_scorer
        self.weights = weights

    def compute_reward(
        self,
        source: str,
        translation: str,
        reference: Optional[str] = None
    ) -> float:
        """
        Compute reward as weighted combination.

        Note: Can work without reference (using semantic similarity)
        """
        adequacy = self.adequacy_scorer.score(source, translation)
        fluency = self.fluency_scorer.score(translation)

        # Bonus if reference available
        if reference:
            bleu = compute_bleu([translation], [reference])
            adequacy = 0.7 * adequacy + 0.3 * bleu

        reward = (
            self.weights['adequacy'] * adequacy +
            self.weights['fluency'] * fluency
        )

        return reward
```

### 5.5 Overfitting Prevention: KL Constraint

```python
class KLConstrainedRL:
    """RL with KL divergence constraint to prevent overfitting."""

    def __init__(
        self,
        policy_model: Model,
        reference_model: Model,
        kl_coefficient: float = 0.1
    ):
        self.policy = policy_model
        self.reference = reference_model  # Frozen supervised model
        self.kl_coefficient = kl_coefficient

    def compute_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute RL loss with KL constraint.

        Loss = -E[reward] + β * KL(policy || reference)

        The KL term prevents policy from deviating too much from
        the supervised baseline, preventing overfitting.
        """
        # Standard policy gradient loss
        log_probs = self.policy.log_prob(states, actions)
        policy_loss = -(log_probs * rewards).mean()

        # KL divergence penalty
        with torch.no_grad():
            ref_log_probs = self.reference.log_prob(states, actions)

        kl_div = (log_probs - ref_log_probs).mean()

        # Combined loss
        total_loss = policy_loss + self.kl_coefficient * kl_div

        return total_loss, {
            'policy_loss': policy_loss.item(),
            'kl_divergence': kl_div.item(),
            'total_loss': total_loss.item()
        }
```

### 5.6 Quality-Based Early Stopping

```python
class QualityBasedEarlyStopping:
    """Stop RL training if quality degrades on validation set."""

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.001,
        validation_metric: str = "comet"
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.validation_metric = validation_metric
        self.best_score = -float('inf')
        self.counter = 0

    def should_stop(
        self,
        validation_scores: Dict[str, float],
        kl_divergence: float,
        max_kl: float = 0.5
    ) -> Tuple[bool, str]:
        """
        Decide whether to stop training.

        Stop if:
        1. Validation quality hasn't improved for `patience` steps
        2. KL divergence exceeds threshold (too far from reference)
        3. Quality drops below minimum threshold
        """
        current_score = validation_scores[self.validation_metric]

        # Check KL divergence
        if kl_divergence > max_kl:
            return True, f"KL divergence too high: {kl_divergence:.4f}"

        # Check quality improvement
        if current_score > self.best_score + self.min_delta:
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            return True, f"No improvement for {self.patience} steps"

        return False, "continue"
```

### 5.7 Minimum Risk Training (MRT)

```python
class MinimumRiskTrainer:
    """
    Minimum Risk Training: Sample-based RL approach.

    More stable than REINFORCE, good for low-resource scenarios.
    """

    def train_step(
        self,
        source_batch: List[str],
        reference_batch: List[str]
    ) -> Dict[str, float]:
        """
        One MRT training step.

        Algorithm:
        1. Sample K translations per source
        2. Compute reward for each sample
        3. Update model to minimize expected risk
        """
        all_samples = []
        all_rewards = []

        for source, reference in zip(source_batch, reference_batch):
            # Sample K translations
            samples = self.model.sample(
                source,
                num_samples=self.config.num_samples,
                temperature=self.config.temperature
            )

            # Compute rewards
            rewards = [
                self.reward_fn.compute_reward(source, sample, reference)
                for sample in samples
            ]

            all_samples.extend(samples)
            all_rewards.extend(rewards)

        # Normalize rewards (risk = negative reward)
        risks = [-r for r in all_rewards]
        risk_probs = F.softmax(torch.tensor(risks), dim=0)

        # Compute MRT loss: E_sample[risk * log P(sample)]
        log_probs = self.model.log_prob(source_batch, all_samples)
        mrt_loss = (risk_probs * log_probs).sum()

        # Add KL constraint
        if self.config.use_kl_constraint:
            kl_penalty = self._compute_kl_penalty(source_batch, all_samples)
            total_loss = mrt_loss + self.config.kl_coefficient * kl_penalty
        else:
            total_loss = mrt_loss

        # Backward and update
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return {
            'mrt_loss': mrt_loss.item(),
            'mean_reward': np.mean(all_rewards),
            'kl_penalty': kl_penalty.item() if self.config.use_kl_constraint else 0.0
        }
```

---

## 6. Integration and Workflow

### 6.1 Enhanced Training Pipeline

```python
class EnhancedTrainingPipeline:
    """Training pipeline with linguistic features, embeddings, and RL."""

    def __init__(
        self,
        base_config: TrainingConfig,
        linguistic_config: Optional[LinguisticFeaturesConfig] = None,
        embedding_config: Optional[LanguageEmbeddingConfig] = None,
        quality_config: Optional[QualityConfig] = None,
        ape_config: Optional[APEConfig] = None,
        rl_config: Optional[RLConfig] = None
    ):
        self.base_config = base_config
        self.linguistic_config = linguistic_config
        self.embedding_config = embedding_config
        self.quality_config = quality_config
        self.ape_config = ape_config
        self.rl_config = rl_config

        # Initialize components
        self._init_components()

    def train(self, dataset: DatasetDict) -> Dict[str, Any]:
        """
        Full training pipeline.

        Phase 1: Supervised training with linguistic features + embeddings
        Phase 2: RL fine-tuning with decomposed reward
        Phase 3: APE model training (optional)
        """
        # Phase 1: Supervised training
        logger.info("Phase 1: Supervised training with enhancements")
        model = self._supervised_training(dataset)

        # Phase 2: RL fine-tuning (optional)
        if self.rl_config:
            logger.info("Phase 2: RL fine-tuning")
            model = self._rl_training(model, dataset)

        # Phase 3: APE training (optional)
        if self.ape_config and self.ape_config.ape_model == "seq2seq":
            logger.info("Phase 3: APE model training")
            ape_model = self._ape_training(model, dataset)

        return {
            'model': model,
            'ape_model': ape_model if self.ape_config else None,
            'metrics': self._final_evaluation(model, dataset)
        }
```

### 6.2 Enhanced Translation Pipeline

```python
class EnhancedTranslationPipeline:
    """Translation with all enhancements."""

    def translate(
        self,
        source: str,
        source_lang: str,
        target_lang: str
    ) -> Dict[str, Any]:
        """
        Full translation pipeline with all features.

        1. Parse source (optional)
        2. Get language embeddings
        3. Translate with linguistic features
        4. Decomposed quality estimation
        5. Apply APE if needed
        6. Return enhanced output
        """
        # Step 1: Parse source
        source_parse = None
        if self.linguistic_config.use_source_parse:
            source_parse = self.parser.parse([source], source_lang)[0]

        # Step 2: Get language embeddings
        src_embedding = self.embedding_encoder.get_embedding(source_lang)
        tgt_embedding = self.embedding_encoder.get_embedding(target_lang)

        # Step 3: Translate
        translation = self.model.translate(
            source,
            source_lang,
            target_lang,
            source_parse=source_parse,
            source_embedding=src_embedding,
            target_embedding=tgt_embedding
        )

        # Step 4: Quality estimation
        quality_scores = self.quality_scorer.score(
            source, translation, source_lang, target_lang
        )

        # Step 5: Apply APE if needed
        if self.ape_pipeline:
            translation, ape_metadata = self.ape_pipeline.post_edit_if_needed(
                source, translation, quality_scores, self.ape_config
            )
        else:
            ape_metadata = {}

        return {
            'translation': translation,
            'quality_scores': quality_scores,
            'ape_metadata': ape_metadata,
            'linguistic_features': {
                'source_parse': source_parse,
                'language_similarity': self.embedding_encoder.get_language_similarity(
                    source_lang, target_lang
                )
            }
        }
```

---

## 7. Implementation Roadmap

### Phase 1: Linguistic Features (2-3 weeks)
1. ✅ Design linguistic features module
2. Implement parser interface and Stanza integration
3. Implement dependency encoder
4. Add encoder augmentation integration
5. Test with NLLB model
6. Evaluate BLEU improvement

**Expected improvement**: +0.5 to +1.5 BLEU

### Phase 2: Language Embeddings (1-2 weeks)
1. ✅ Design language embeddings module
2. Integrate lang2vec loader
3. Implement adapter conditioning
4. Add transfer learning support
5. Test with multilingual model
6. Evaluate on similar language pairs

**Expected improvement**: +0.3 to +1.0 BLEU on low-resource pairs

### Phase 3: Quality Components (2-3 weeks)
1. ✅ Design adequacy and fluency modules
2. Implement semantic similarity scorer (LaBSE)
3. Implement syntactic fluency scorer
4. Implement decomposed quality scorer
5. Integrate with quality estimator
6. Evaluate correlation with human judgments

**Expected improvement**: Better quality estimation accuracy

### Phase 4: Automatic Post-Editing (2-3 weeks)
1. ✅ Design APE module
2. Implement LLM-based APE
3. Implement quality-aware triggering
4. Implement selective APE
5. Test cost-effectiveness
6. Evaluate post-editing gains

**Expected improvement**: +0.5 to +2.0 BLEU on low-quality outputs

### Phase 5: Reinforcement Learning (3-4 weeks)
1. ✅ Design RL module with overfitting prevention
2. Implement MRT algorithm
3. Implement KL constraint
4. Implement early stopping
5. Test on low-resource language
6. Monitor for overfitting (validation curves)

**Expected improvement**: +0.5 to +1.5 BLEU, but with risk of overfitting if not careful

---

## 8. Configuration Files

### 8.1 Master Configuration

```yaml
# config/advanced_features.yaml

linguistic_features:
  use_source_parse: true
  use_target_parse: false
  parser: "stanza"
  features:
    - dependency_labels
    - pos_tags
    - tree_depth
  integration_method: "encoder_augmentation"
  encoding_dim: 128

language_embeddings:
  use_source_embedding: true
  use_target_embedding: true
  source: "lang2vec"
  embedding_dim: 256
  integration_method: "adapter_conditioning"
  use_transfer_learning: true
  similarity_threshold: 0.7

quality:
  adequacy:
    method: "semantic_similarity"
    model: "labse"
  fluency:
    method: "perplexity"
    model: "gpt2-large"
  weights:
    adequacy: 0.5
    fluency: 0.5

ape:
  ape_model: "llm"
  llm_provider: "anthropic"
  llm_model: "claude-haiku-4.5"
  trigger_on: "decomposed"
  adequacy_threshold: 0.6
  fluency_threshold: 0.6
  strategy: "selective"

reinforcement_learning:
  algorithm: "mrt"
  reward_type: "decomposed"
  use_kl_constraint: true
  kl_coefficient: 0.1
  use_early_stopping: true
  patience: 5
  num_epochs: 5
  validation_freq: 100
```

---

## 9. CLI Scripts

### 9.1 Training with Advanced Features

```bash
# Train with linguistic features
python scripts/train_model.py \
  --model nllb-600m \
  --source-lang hi \
  --target-lang bho \
  --train-data data/train.tsv \
  --use-linguistic-features \
  --linguistic-config config/linguistic.yaml \
  --output models/nllb-bho-enhanced

# Train with RL fine-tuning
python scripts/train_model.py \
  --model nllb-600m \
  --source-lang hi \
  --target-lang bho \
  --train-data data/train.tsv \
  --use-rl \
  --rl-config config/rl.yaml \
  --output models/nllb-bho-rl
```

### 9.2 Translation with APE

```bash
# Translate with APE
python scripts/hybrid_translate.py \
  --finetuned-model models/nllb-bho \
  --use-ape \
  --ape-model llm \
  --ape-provider anthropic \
  --quality-threshold 0.7 \
  --source-lang hi \
  --target-lang bho \
  --input texts.txt \
  --output translations.txt
```

---

## 10. Expected Outcomes

### Quality Improvements (Cumulative)
- **Base fine-tuned model**: +3 to +8 BLEU vs base
- **+ Linguistic features**: +0.5 to +1.5 BLEU
- **+ Language embeddings**: +0.3 to +1.0 BLEU
- **+ RL optimization**: +0.5 to +1.5 BLEU
- **+ APE (selective)**: +0.5 to +2.0 BLEU on low-quality outputs
- **Total**: +5 to +14 BLEU vs base model

### Cost Analysis
- **Training cost increase**: +20-30% (parsing, RL)
- **Inference cost**: Minimal (linguistic features cached)
- **APE cost**: Selective application keeps costs low
- **Overall**: Still 8-10x cheaper than pure API

### Research Contributions
- Systematic integration of linguistic knowledge
- Decomposed quality estimation (adequacy/fluency)
- Quality-aware APE triggering
- RL with overfitting prevention for low-resource MT

---

## 11. Risks and Mitigation

### Risk 1: Overfitting in RL (Goodhart's Law)
**Mitigation**:
- KL divergence constraint (prevent deviation from supervised baseline)
- Reference model (frozen supervised model)
- Validation-based early stopping
- Multiple reward objectives (diversification)
- Conservative hyperparameters (low LR, few epochs)

### Risk 2: Parsing Errors
**Mitigation**:
- Ensemble of parsers
- Robust feature encoding (handle malformed parses)
- Optional features (can disable if hurt performance)

### Risk 3: APE Degradation
**Mitigation**:
- Quality-aware triggering (only apply when needed)
- Confidence thresholds
- Iterative APE with stopping criteria
- Cost limits

### Risk 4: Computational Cost
**Mitigation**:
- Batch parsing
- Cache embeddings and parses
- Selective application (only when beneficial)
- Efficient models (distilled LaBSE, etc.)

---

## 12. Future Extensions

1. **Multimodal features**: Images, videos for context
2. **Interactive translation**: User feedback in the loop
3. **Domain adaptation**: Linguistic features per domain
4. **Meta-learning**: Learn to adapt quickly to new languages
5. **Neural-symbolic integration**: Combine neural + symbolic parsing

This design provides a comprehensive, modular framework for enhancing MT with linguistic knowledge, quality decomposition, and reinforcement learning while maintaining configurability and preventing overfitting in low-resource scenarios.
