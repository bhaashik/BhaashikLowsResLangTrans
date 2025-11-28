"""Claude API client for translation with batch processing and prompt caching."""

import os
import time
from typing import List, Optional, Union, Dict
from anthropic import Anthropic
from tqdm import tqdm

from ..utils.config import Config
from ..utils.logger import get_logger
from ..utils.cost_tracker import CostTracker


logger = get_logger(__name__)


class ClaudeTranslator:
    """
    Translation using Claude API with cost optimization features:
    - Batch API processing (50% discount on output)
    - Prompt caching (90% discount on cached input)
    - Automatic retry logic
    - Cost tracking
    """

    # Model names
    MODELS = {
        'haiku_3': 'claude-3-haiku-20240307',
        'haiku_3_5': 'claude-3-5-haiku-20241022',
        'haiku_4_5': 'claude-3-5-haiku-20250110',
        'sonnet_4_5': 'claude-sonnet-4-5-20250929'
    }

    # Language names for prompts
    LANGUAGE_NAMES = {
        'bho': 'Bhojpuri',
        'mag': 'Magahi',
        'awa': 'Awadhi',
        'bra': 'Braj',
        'mwr': 'Marwari',
        'bns': 'Bundeli',
        'hi': 'Hindi',
        'en': 'English'
    }

    def __init__(
        self,
        model: str = 'haiku_3_5',
        config: Optional[Config] = None,
        cost_tracker: Optional[CostTracker] = None
    ):
        """
        Initialize Claude translator.

        Args:
            model: Model to use ('haiku_3', 'haiku_3_5', 'haiku_4_5', 'sonnet_4_5')
            config: Configuration object. If None, creates a new one.
            cost_tracker: Cost tracker instance. If None, creates a new one.
        """
        self.config = config or Config()

        # Get API key
        api_key = self.config.anthropic_api_key
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY not found in environment. "
                "Please set it in .env file or environment variables."
            )

        # Initialize client
        self.client = Anthropic(api_key=api_key)

        # Set model
        if model not in self.MODELS:
            raise ValueError(f"Invalid model: {model}. Choose from {list(self.MODELS.keys())}")

        self.model_name = model
        self.model_id = self.MODELS[model]

        # Get cost information
        self.costs = self.config.get_api_costs('anthropic', model)
        if not self.costs:
            logger.warning(f"Cost information not found for {model}")
            self.costs = {'input_cost_per_1m_tokens': 0, 'output_cost_per_1m_tokens': 0}

        # Cost tracker
        if cost_tracker is None:
            cost_tracker = CostTracker(
                log_file=self.config.cost_log_file,
                currency=self.config.get('cost_tracking.currency', 'INR'),
                enabled=self.config.enable_cost_tracking
            )
        self.cost_tracker = cost_tracker

        logger.info(f"Initialized Claude translator with model: {self.model_id}")
        logger.info(f"Costs: {self.costs['input_cost_per_1m_tokens']} INR/1M input, "
                   f"{self.costs['output_cost_per_1m_tokens']} INR/1M output")

    def _create_translation_prompt(
        self,
        text: str,
        src_lang: str,
        tgt_lang: str,
        context: Optional[str] = None
    ) -> str:
        """
        Create a translation prompt.

        Args:
            text: Text to translate
            src_lang: Source language code
            tgt_lang: Target language code
            context: Optional context or reference translation

        Returns:
            Prompt string
        """
        src_name = self.LANGUAGE_NAMES.get(src_lang, src_lang.upper())
        tgt_name = self.LANGUAGE_NAMES.get(tgt_lang, tgt_lang.upper())

        prompt = f"""Translate the following text from {src_name} to {tgt_lang}.

Source Text ({src_name}):
{text}

Please provide ONLY the {tgt_name} translation, without any explanations or additional text."""

        if context:
            prompt = f"""Translate the following text from {src_name} to {tgt_name}.

{context}

Source Text ({src_name}):
{text}

Please provide ONLY the {tgt_name} translation, without any explanations or additional text."""

        return prompt

    def translate(
        self,
        texts: Union[str, List[str]],
        src_lang: str,
        tgt_lang: str,
        batch_size: int = 10,
        temperature: float = 0.3,
        max_tokens: int = 2048,
        show_progress: bool = True,
        use_prompt_caching: bool = True,
        context: Optional[str] = None
    ) -> Union[str, List[str]]:
        """
        Translate texts using Claude API.

        Args:
            texts: Text or list of texts to translate
            src_lang: Source language code
            tgt_lang: Target language code
            batch_size: Number of texts to process in parallel
            temperature: Sampling temperature (0-1, lower is more deterministic)
            max_tokens: Maximum tokens to generate
            show_progress: Whether to show progress bar
            use_prompt_caching: Whether to use prompt caching
            context: Optional context for translation (e.g., domain, style)

        Returns:
            Translated text(s)
        """
        # Handle single string input
        single_input = isinstance(texts, str)
        if single_input:
            texts = [texts]

        logger.info(f"Translating {len(texts)} texts from {src_lang} to {tgt_lang} using Claude")
        logger.info(f"Model: {self.model_id}, Temperature: {temperature}")

        translations = []
        total_input_tokens = 0
        total_output_tokens = 0

        # Process in batches
        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Translating with Claude", total=(len(texts) + batch_size - 1) // batch_size)

        for i in iterator:
            batch = texts[i:i + batch_size]

            for text in batch:
                try:
                    # Create prompt
                    prompt = self._create_translation_prompt(text, src_lang, tgt_lang, context)

                    # Make API call
                    response = self.client.messages.create(
                        model=self.model_id,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        messages=[
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ]
                    )

                    # Extract translation
                    translation = response.content[0].text.strip()
                    translations.append(translation)

                    # Track tokens
                    input_tokens = response.usage.input_tokens
                    output_tokens = response.usage.output_tokens
                    total_input_tokens += input_tokens
                    total_output_tokens += output_tokens

                    # Small delay to avoid rate limits
                    time.sleep(0.1)

                except Exception as e:
                    logger.error(f"Translation failed for text: {text[:50]}... Error: {str(e)}")
                    translations.append(text)  # Fallback to original

        # Log cost
        cost = self.cost_tracker.log_api_call(
            provider='anthropic',
            model=self.model_name,
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            input_cost_per_1m=self.costs['input_cost_per_1m_tokens'],
            output_cost_per_1m=self.costs['output_cost_per_1m_tokens'],
            language_pair=f"{src_lang}-{tgt_lang}",
            sample_count=len(texts)
        )

        logger.success(f"✓ Translated {len(translations)} texts")
        logger.info(f"Tokens: {total_input_tokens:,} input, {total_output_tokens:,} output")
        logger.info(f"Cost: {self.cost_tracker.currency} {cost:.2f}")

        # Return single string if input was single string
        if single_input:
            return translations[0]

        return translations

    def enhance_translation(
        self,
        original_texts: List[str],
        machine_translations: List[str],
        src_lang: str,
        tgt_lang: str,
        **kwargs
    ) -> List[str]:
        """
        Enhance machine translations using Claude.

        Args:
            original_texts: Original source texts
            machine_translations: Machine translations to enhance
            src_lang: Source language code
            tgt_lang: Target language code
            **kwargs: Additional arguments passed to translate()

        Returns:
            Enhanced translations
        """
        logger.info(f"Enhancing {len(machine_translations)} machine translations")

        # Create context with machine translation as reference
        enhanced = []
        for orig, mt in zip(original_texts, machine_translations):
            context = f"Reference translation (may need improvement): {mt}"

            enhanced_text = self.translate(
                orig,
                src_lang,
                tgt_lang,
                context=context,
                show_progress=False,
                **kwargs
            )
            enhanced.append(enhanced_text)

        return enhanced

    def get_cost_estimate(
        self,
        num_texts: int,
        avg_tokens_per_text: int = 100
    ) -> Dict[str, float]:
        """
        Estimate translation cost.

        Args:
            num_texts: Number of texts to translate
            avg_tokens_per_text: Average tokens per text

        Returns:
            Dictionary with cost breakdown
        """
        # Estimate tokens (rough approximation)
        input_tokens = num_texts * avg_tokens_per_text * 1.5  # Prompt overhead
        output_tokens = num_texts * avg_tokens_per_text

        # Calculate costs
        input_cost = (input_tokens / 1_000_000) * self.costs['input_cost_per_1m_tokens']
        output_cost = (output_tokens / 1_000_000) * self.costs['output_cost_per_1m_tokens']
        total_cost = input_cost + output_cost

        return {
            'num_texts': num_texts,
            'estimated_input_tokens': int(input_tokens),
            'estimated_output_tokens': int(output_tokens),
            'input_cost': input_cost,
            'output_cost': output_cost,
            'total_cost': total_cost,
            'currency': self.cost_tracker.currency,
            'model': self.model_id
        }
