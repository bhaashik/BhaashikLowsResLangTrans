"""OpenAI API provider with prompt caching support."""

import os
from typing import Optional, Dict, Any, List
from ..core import (
    BaseTranslator,
    TranslationRequest,
    TranslationResponse,
    TranslationResult,
    TranslationStatus,
    CostEstimate,
    TranslationError,
    AuthenticationError
)
from ..prompts import PromptManager


class OpenAIProvider(BaseTranslator):
    """
    Translation provider using OpenAI API.

    Supports:
    - Multiple models (GPT-4o, GPT-4o-mini, GPT-3.5-turbo)
    - Prompt caching for cost optimization (automatic with chat completions)
    """

    # Model specifications
    MODELS = {
        'gpt-5-nano': {
            'input_cost': 0.10,      # USD per 1M tokens (estimated)
            'output_cost': 0.40,
            'cached_input_cost': 0.05,  # 50% discount
            'max_tokens': 200000,
            'supports_caching': True
        },
        'gpt-4o': {
            'input_cost': 2.50,      # USD per 1M tokens
            'output_cost': 10.00,
            'cached_input_cost': 1.25,  # 50% discount
            'max_tokens': 128000,
            'supports_caching': True
        },
        'gpt-4o-mini': {
            'input_cost': 0.15,
            'output_cost': 0.60,
            'cached_input_cost': 0.075,  # 50% discount
            'max_tokens': 128000,
            'supports_caching': True
        },
        'gpt-3.5-turbo': {
            'input_cost': 0.50,
            'output_cost': 1.50,
            'cached_input_cost': 0.25,
            'max_tokens': 16000,
            'supports_caching': False
        }
    }

    def __init__(
        self,
        model: str = 'gpt-5-nano',
        api_key: Optional[str] = None,
        prompt_manager: Optional[PromptManager] = None,
        config: Optional[Dict[str, Any]] = None,
        use_caching: bool = True
    ):
        """
        Initialize OpenAI provider.

        Args:
            model: OpenAI model name (default: gpt-5-nano)
            api_key: OpenAI API key (or from OPENAI_API_KEY env var)
            prompt_manager: PromptManager instance for template rendering
            config: Additional configuration
            use_caching: Whether to use prompt caching (default: True)
        """
        super().__init__(name='openai', model=model, config=config or {})

        if model not in self.MODELS:
            raise ValueError(f"Unknown model: {model}. Choose from {list(self.MODELS.keys())}")

        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise AuthenticationError("OpenAI API key not provided")

        self.prompt_manager = prompt_manager
        self.use_caching = use_caching and self.MODELS[model]['supports_caching']
        self.client = None

    def initialize(self):
        """Initialize OpenAI client."""
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
            self._initialized = True
        except ImportError:
            raise TranslationError(
                "openai package not installed. Run: pip install openai",
                provider=self.name
            )

    async def translate(self, request: TranslationRequest) -> TranslationResponse:
        """
        Translate using OpenAI API (async).

        Args:
            request: Translation request

        Returns:
            Translation response
        """
        if not self._initialized:
            self.initialize()

        results = []
        total_cost = 0.0

        # Build system message (cached if using examples)
        system_content = self._build_system_message(request)

        # For batch translation, combine all texts
        if len(request.units) > 1:
            # Build batch user message
            batch_texts = "\n\n".join([f"{i+1}. {unit.text}" for i, unit in enumerate(request.units)])
            user_content = f"""Translate the following {request.src_lang} sentences to {request.tgt_lang}.
Provide ONLY the translations, one per line, numbered to match the input.

{batch_texts}

Provide only the {request.tgt_lang} translations without explanations."""

            # Build messages array
            messages = []
            if system_content:
                messages.append({"role": "system", "content": system_content})
            messages.append({"role": "user", "content": user_content})

            try:
                # Call API
                params = self._get_api_params(request)
                response = self.client.chat.completions.create(
                    **params,
                    messages=messages
                )

                # Extract translation
                translation_text = response.choices[0].message.content if response.choices else ""

                # Calculate cost
                cost = self._calculate_cost(response.usage)
                total_cost += cost

                # Parse batch translations (split by newlines and remove numbering)
                translation_lines = translation_text.strip().split('\n')
                translations = []
                for line in translation_lines:
                    # Remove numbering like "1. ", "2. ", etc.
                    line = line.strip()
                    if line:
                        # Try to remove number prefix
                        import re
                        cleaned = re.sub(r'^\d+\.\s*', '', line)
                        translations.append(cleaned)

                # Match translations to units
                for i, unit in enumerate(request.units):
                    if i < len(translations):
                        # Extract cached tokens info
                        cached_tokens = 0
                        if hasattr(response.usage, 'prompt_tokens_details') and response.usage.prompt_tokens_details:
                            cached_tokens = getattr(response.usage.prompt_tokens_details, 'cached_tokens', 0)

                        results.append(TranslationResult(
                            source=unit.text,
                            translation=translations[i],
                            index=unit.index,
                            cost=cost / len(request.units),  # Distribute cost evenly
                            status=TranslationStatus.COMPLETED,
                            metadata={
                                'model': self.model,
                                'prompt_tokens': response.usage.prompt_tokens,
                                'completion_tokens': response.usage.completion_tokens,
                                'total_tokens': response.usage.total_tokens,
                                'cached_tokens': cached_tokens
                            }
                        ))
                    else:
                        results.append(TranslationResult(
                            source=unit.text,
                            translation="",
                            index=unit.index,
                            status=TranslationStatus.FAILED,
                            error="Missing translation in batch response"
                        ))

            except Exception as e:
                # If batch fails, mark all as failed
                for unit in request.units:
                    results.append(TranslationResult(
                        source=unit.text,
                        translation="",
                        index=unit.index,
                        status=TranslationStatus.FAILED,
                        error=str(e)
                    ))

        else:
            # Single translation
            unit = request.units[0]
            try:
                # Build user message
                user_content = self._build_user_message(unit.text, request)

                # Build messages array
                messages = []
                if system_content:
                    messages.append({"role": "system", "content": system_content})
                messages.append({"role": "user", "content": user_content})

                # Call API
                params = self._get_api_params(request)
                response = self.client.chat.completions.create(
                    **params,
                    messages=messages
                )

                # Extract translation
                translation = response.choices[0].message.content if response.choices else ""

                # Calculate cost
                cost = self._calculate_cost(response.usage)

                # Extract cached tokens info
                cached_tokens = 0
                if hasattr(response.usage, 'prompt_tokens_details') and response.usage.prompt_tokens_details:
                    cached_tokens = getattr(response.usage.prompt_tokens_details, 'cached_tokens', 0)

                results.append(TranslationResult(
                    source=unit.text,
                    translation=translation,
                    index=unit.index,
                    cost=cost,
                    status=TranslationStatus.COMPLETED,
                    metadata={
                        'model': self.model,
                        'prompt_tokens': response.usage.prompt_tokens,
                        'completion_tokens': response.usage.completion_tokens,
                        'total_tokens': response.usage.total_tokens,
                        'cached_tokens': cached_tokens
                    }
                ))

                total_cost += cost

            except Exception as e:
                results.append(TranslationResult(
                    source=unit.text,
                    translation="",
                    index=unit.index,
                    status=TranslationStatus.FAILED,
                    error=str(e)
                ))

        return TranslationResponse(
            results=results,
            total_cost=total_cost,
            provider=self.name,
            model=self.model,
            metadata={'use_caching': self.use_caching}
        )

    def translate_sync(self, request: TranslationRequest) -> TranslationResponse:
        """Synchronous translation."""
        import asyncio
        return asyncio.run(self.translate(request))

    def _build_system_message(self, request: TranslationRequest) -> Optional[str]:
        """Build system message with examples for caching."""
        if not self.prompt_manager:
            return None

        # Get base system prompt
        system_prompt = self.prompt_manager.get_system_prompt(
            source_lang=request.src_lang,
            target_lang=request.tgt_lang
        )

        # Add examples if available (these will be cached)
        if self.prompt_manager.examples and self.use_caching:
            examples_str = self.prompt_manager.format_examples_for_prompt(
                max_examples=None  # Use all examples for maximum cache benefit
            )
            if examples_str:
                system_prompt = f"{system_prompt}\n\nExamples:\n{examples_str}"

        return system_prompt

    def _build_user_message(self, text: str, request: TranslationRequest) -> str:
        """Build user message for translation."""
        if self.prompt_manager:
            return self.prompt_manager.get_user_prompt(
                text=text,
                source_lang=request.src_lang,
                target_lang=request.tgt_lang
            )
        else:
            # Fallback simple prompt
            return f"Translate the following {request.src_lang} text to {request.tgt_lang}:\n\n{text}\n\nProvide only the translation."

    def _get_api_params(self, request: TranslationRequest) -> Dict[str, Any]:
        """Get API parameters."""
        # GPT-5-nano has specific parameter requirements
        if self.model in ['gpt-5-nano']:
            # GPT-5-nano is a reasoning model that uses tokens for internal reasoning
            # Default to 16000 to allow for both reasoning and output
            params = {
                'model': self.model,
                'max_completion_tokens': request.parameters.get('max_tokens', 16000)
                # Note: GPT-5-nano only supports temperature=1 (default), so we omit it
            }
        else:
            params = {
                'model': self.model,
                'max_tokens': request.parameters.get('max_tokens', 2048),
                'temperature': request.parameters.get('temperature', 0.3)
            }

        return params

    def _calculate_cost(self, usage) -> float:
        """Calculate cost from usage statistics."""
        model_spec = self.MODELS[self.model]

        # Get token counts
        prompt_tokens = usage.prompt_tokens
        completion_tokens = usage.completion_tokens

        # Check for cached tokens (OpenAI provides this in prompt_tokens_details)
        cached_tokens = 0
        if hasattr(usage, 'prompt_tokens_details') and usage.prompt_tokens_details:
            cached_tokens = getattr(usage.prompt_tokens_details, 'cached_tokens', 0)

        # Calculate costs
        regular_input_tokens = prompt_tokens - cached_tokens
        input_cost = (regular_input_tokens / 1_000_000) * model_spec['input_cost']

        if self.use_caching and cached_tokens > 0:
            cached_cost = (cached_tokens / 1_000_000) * model_spec.get('cached_input_cost', model_spec['input_cost'] * 0.5)
            input_cost += cached_cost

        output_cost = (completion_tokens / 1_000_000) * model_spec['output_cost']

        return input_cost + output_cost

    def get_cost_estimate(self, request: TranslationRequest) -> CostEstimate:
        """Estimate translation cost."""
        model_spec = self.MODELS[self.model]

        # Rough token estimation (4 chars per token average)
        avg_input_tokens = request.total_chars // 4

        # Add system prompt tokens if using prompt manager
        system_tokens = 0
        if self.prompt_manager:
            if self.prompt_manager.examples:
                # Estimate ~100 tokens per example
                system_tokens = len(self.prompt_manager.examples) * 100
            system_tokens += 200  # Base system prompt

        total_input_tokens = avg_input_tokens + (system_tokens * request.num_units)
        avg_output_tokens = request.total_chars // 4  # Assume similar length

        # If caching, most system tokens will be cached after first request
        if self.use_caching and request.num_units > 1:
            # First request: full cost
            # Subsequent: cached
            regular_input_cost = (avg_input_tokens * request.num_units / 1_000_000) * model_spec['input_cost']
            cached_input_cost = (system_tokens * (request.num_units - 1) / 1_000_000) * model_spec.get('cached_input_cost', model_spec['input_cost'] * 0.5)
            system_first_cost = (system_tokens / 1_000_000) * model_spec['input_cost']

            input_cost = regular_input_cost + cached_input_cost + system_first_cost
        else:
            input_cost = (total_input_tokens / 1_000_000) * model_spec['input_cost']

        output_cost = (avg_output_tokens / 1_000_000) * model_spec['output_cost']

        return CostEstimate(
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=input_cost + output_cost,
            currency="USD",
            num_units=request.num_units,
            estimated_tokens=total_input_tokens + avg_output_tokens,
            metadata={
                'model': self.model,
                'caching': self.use_caching,
                'estimated_system_tokens': system_tokens
            }
        )

    def supports_batch(self) -> bool:
        """OpenAI supports batch processing via Batch API."""
        return True

    def supports_prompt_caching(self) -> bool:
        """Check if model supports prompt caching."""
        return self.MODELS[self.model].get('supports_caching', False)

    def get_max_tokens(self) -> int:
        """Get maximum tokens for model."""
        return self.MODELS[self.model]['max_tokens']

    def get_rate_limits(self) -> Dict[str, int]:
        """Get rate limits."""
        return {
            'requests_per_minute': 500,
            'tokens_per_minute': 150000
        }
