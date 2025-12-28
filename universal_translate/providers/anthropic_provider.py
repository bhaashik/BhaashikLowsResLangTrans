"""Anthropic Claude API provider."""

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


class AnthropicProvider(BaseTranslator):
    """
    Translation provider using Anthropic Claude API.

    Supports:
    - Multiple Claude models (Haiku, Sonnet, Opus)
    - Prompt caching for cost optimization
    - Batch API for additional savings
    """

    # Model specifications (INR per 1M tokens, ~₹85 per USD)
    MODELS = {
        # Haiku models (fast, cost-effective)
        'claude-haiku-3': {
            'input_cost': 0.21,      # INR per 1M tokens
            'output_cost': 1.06,
            'max_tokens': 200000,
            'supports_caching': False
        },
        'claude-haiku-3.5': {
            'input_cost': 0.68,
            'output_cost': 3.40,
            'max_tokens': 200000,
            'supports_caching': True,
            'cache_write_cost': 1.02,
            'cache_read_cost': 0.068
        },
        'claude-haiku-4.5': {
            'input_cost': 0.85,
            'output_cost': 4.25,
            'max_tokens': 200000,
            'supports_caching': True,
            'cache_write_cost': 1.275,
            'cache_read_cost': 0.085
        },
        # Sonnet models (balanced quality and speed)
        'claude-sonnet-3.5': {
            'input_cost': 2.55,
            'output_cost': 12.75,
            'max_tokens': 200000,
            'supports_caching': True,
            'cache_write_cost': 3.825,
            'cache_read_cost': 0.255
        },
        'claude-sonnet-4': {
            'input_cost': 2.55,
            'output_cost': 12.75,
            'max_tokens': 200000,
            'supports_caching': True,
            'cache_write_cost': 3.825,
            'cache_read_cost': 0.255
        },
        'claude-sonnet-4.5': {
            'input_cost': 2.55,
            'output_cost': 12.75,
            'max_tokens': 200000,
            'supports_caching': True,
            'cache_write_cost': 3.825,
            'cache_read_cost': 0.255
        },
        # Opus models (highest quality)
        'claude-opus-3': {
            'input_cost': 12.75,     # INR per 1M tokens
            'output_cost': 63.75,
            'max_tokens': 200000,
            'supports_caching': True,
            'cache_write_cost': 19.13,
            'cache_read_cost': 1.28
        },
        'claude-opus-4': {
            'input_cost': 12.75,
            'output_cost': 63.75,
            'max_tokens': 200000,
            'supports_caching': True,
            'cache_write_cost': 19.13,
            'cache_read_cost': 1.28
        }
    }

    def __init__(
        self,
        model: str = 'claude-haiku-4.5',
        api_key: Optional[str] = None,
        prompt_manager: Optional[PromptManager] = None,
        config: Optional[Dict[str, Any]] = None,
        use_batch_api: bool = False
    ):
        """
        Initialize Anthropic provider.

        Args:
            model: Claude model name
            api_key: Anthropic API key (or from ANTHROPIC_API_KEY env var)
            prompt_manager: PromptManager instance for template rendering
            config: Additional configuration
            use_batch_api: Whether to use Batch API (50% output discount)
        """
        super().__init__(name='anthropic', model=model, config=config or {})

        if model not in self.MODELS:
            raise ValueError(f"Unknown model: {model}. Choose from {list(self.MODELS.keys())}")

        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        if not self.api_key:
            raise AuthenticationError("Anthropic API key not provided")

        self.prompt_manager = prompt_manager
        self.use_batch_api = use_batch_api
        self.client = None

    def initialize(self):
        """Initialize Anthropic client."""
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=self.api_key)
            self._initialized = True
        except ImportError:
            raise TranslationError(
                "anthropic package not installed. Run: pip install anthropic",
                provider=self.name
            )

    async def translate(self, request: TranslationRequest) -> TranslationResponse:
        """
        Translate using Claude API (async).

        Args:
            request: Translation request

        Returns:
            Translation response
        """
        if not self._initialized:
            self.initialize()

        results = []
        total_cost = 0.0

        for unit in request.units:
            try:
                # Render prompts
                system_prompt = None
                user_prompt = unit.text

                if self.prompt_manager:
                    system_prompt = self.prompt_manager.get_system_prompt(
                        source_lang=request.src_lang,
                        target_lang=request.tgt_lang
                    )
                    user_prompt = self.prompt_manager.get_user_prompt(
                        text=unit.text,
                        source_lang=request.src_lang,
                        target_lang=request.tgt_lang
                    )

                    # Add examples if available
                    if self.prompt_manager.examples:
                        examples_str = self.prompt_manager.format_examples_for_prompt()
                        user_prompt = f"Examples:\n{examples_str}\n\n{user_prompt}"

                # Build messages
                messages = [{"role": "user", "content": user_prompt}]

                # Call API
                params = self._get_api_params(request, system_prompt)
                response = self.client.messages.create(**params, messages=messages)

                # Extract translation
                translation = response.content[0].text if response.content else ""

                # Calculate cost
                cost = self._calculate_cost(response.usage)

                results.append(TranslationResult(
                    source=unit.text,
                    translation=translation,
                    index=unit.index,
                    cost=cost,
                    status=TranslationStatus.COMPLETED,
                    metadata={
                        'model': self.model,
                        'input_tokens': response.usage.input_tokens,
                        'output_tokens': response.usage.output_tokens,
                        'cache_creation_tokens': getattr(response.usage, 'cache_creation_input_tokens', 0),
                        'cache_read_tokens': getattr(response.usage, 'cache_read_input_tokens', 0)
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
            metadata={'use_batch_api': self.use_batch_api}
        )

    def translate_sync(self, request: TranslationRequest) -> TranslationResponse:
        """Synchronous translation."""
        import asyncio
        return asyncio.run(self.translate(request))

    def _get_api_params(
        self,
        request: TranslationRequest,
        system_prompt: Optional[str]
    ) -> Dict[str, Any]:
        """Get API parameters."""
        params = {
            'model': self.model,
            'max_tokens': request.parameters.get('max_tokens', 2048),
            'temperature': request.parameters.get('temperature', 0.3)
        }

        # Add system prompt with caching if supported
        if system_prompt:
            if self.supports_prompt_caching() and self.prompt_manager.supports_caching():
                params['system'] = [
                    {
                        'type': 'text',
                        'text': system_prompt,
                        'cache_control': {'type': 'ephemeral'}
                    }
                ]
            else:
                params['system'] = system_prompt

        return params

    def _calculate_cost(self, usage) -> float:
        """Calculate cost from usage statistics."""
        model_spec = self.MODELS[self.model]

        # Base costs
        input_cost = (usage.input_tokens / 1_000_000) * model_spec['input_cost']
        output_cost = (usage.output_tokens / 1_000_000) * model_spec['output_cost']

        # Caching costs
        if hasattr(usage, 'cache_creation_input_tokens') and usage.cache_creation_input_tokens:
            cache_write_cost = (usage.cache_creation_input_tokens / 1_000_000) * \
                             model_spec.get('cache_write_cost', model_spec['input_cost'])
            input_cost += cache_write_cost

        if hasattr(usage, 'cache_read_input_tokens') and usage.cache_read_input_tokens:
            cache_read_cost = (usage.cache_read_input_tokens / 1_000_000) * \
                            model_spec.get('cache_read_cost', model_spec['input_cost'] * 0.1)
            input_cost += cache_read_cost

        # Batch API discount (50% off output)
        if self.use_batch_api:
            output_cost *= 0.5

        return input_cost + output_cost

    def get_cost_estimate(self, request: TranslationRequest) -> CostEstimate:
        """Estimate translation cost."""
        model_spec = self.MODELS[self.model]

        # Rough token estimation (4 chars per token average)
        avg_input_tokens = request.total_chars // 4
        avg_output_tokens = request.total_chars // 4  # Assume similar length

        input_cost = (avg_input_tokens / 1_000_000) * model_spec['input_cost']
        output_cost = (avg_output_tokens / 1_000_000) * model_spec['output_cost']

        if self.use_batch_api:
            output_cost *= 0.5

        return CostEstimate(
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=input_cost + output_cost,
            currency="INR",
            num_units=request.num_units,
            estimated_tokens=avg_input_tokens + avg_output_tokens,
            metadata={'model': self.model, 'batch_api': self.use_batch_api}
        )

    def supports_batch(self) -> bool:
        """Anthropic supports batch processing."""
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
            'requests_per_minute': 1000,
            'tokens_per_minute': 100000
        }
