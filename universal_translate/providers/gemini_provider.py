"""Google Gemini API provider."""

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


class GeminiProvider(BaseTranslator):
    """
    Translation provider using Google Gemini API.

    Supports:
    - Multiple Gemini models (Flash, Pro)
    - Prompt caching for cost optimization
    - Context caching for faster responses
    """

    # Model specifications (INR per 1M tokens via Gemini API)
    MODELS = {
        # Gemini 2.0 models (latest)
        'gemini-2.0-flash-exp': {
            'input_cost': 0.0,       # Free tier experimental
            'output_cost': 0.0,
            'max_tokens': 1000000,
            'supports_caching': True
        },
        'gemini-2.0-flash': {
            'input_cost': 3.19,      # INR per 1M tokens
            'output_cost': 9.56,
            'cached_input_cost': 0.80,  # 75% discount
            'max_tokens': 1000000,
            'supports_caching': True
        },
        # Gemini 1.5 models
        'gemini-1.5-flash': {
            'input_cost': 3.19,      # INR per 1M tokens
            'output_cost': 9.56,
            'cached_input_cost': 0.80,  # 75% discount
            'max_tokens': 1000000,
            'supports_caching': True
        },
        'gemini-1.5-flash-8b': {
            'input_cost': 1.59,      # INR per 1M tokens (smaller, faster)
            'output_cost': 4.78,
            'cached_input_cost': 0.40,  # 75% discount
            'max_tokens': 1000000,
            'supports_caching': True
        },
        'gemini-1.5-pro': {
            'input_cost': 106.25,    # INR per 1M tokens
            'output_cost': 425.00,
            'cached_input_cost': 26.56,  # 75% discount
            'max_tokens': 2000000,
            'supports_caching': True
        },
        # Gemini 1.0 models (legacy)
        'gemini-1.0-pro': {
            'input_cost': 42.50,
            'output_cost': 127.50,
            'max_tokens': 32000,
            'supports_caching': False
        },
        'gemini-pro': {
            'input_cost': 42.50,     # Alias for gemini-1.0-pro
            'output_cost': 127.50,
            'max_tokens': 32000,
            'supports_caching': False
        }
    }

    def __init__(
        self,
        model: str = 'gemini-1.5-flash',
        api_key: Optional[str] = None,
        prompt_manager: Optional[PromptManager] = None,
        config: Optional[Dict[str, Any]] = None,
        use_caching: bool = True
    ):
        """
        Initialize Gemini provider.

        Args:
            model: Gemini model name
            api_key: Google API key (or from GOOGLE_API_KEY env var)
            prompt_manager: PromptManager instance for template rendering
            config: Additional configuration
            use_caching: Whether to use context caching (default: True)
        """
        super().__init__(name='gemini', model=model, config=config or {})

        if model not in self.MODELS:
            raise ValueError(f"Unknown model: {model}. Choose from {list(self.MODELS.keys())}")

        self.api_key = api_key or os.getenv('GOOGLE_API_KEY')
        if not self.api_key:
            raise AuthenticationError("Google API key not provided")

        self.prompt_manager = prompt_manager
        self.use_caching = use_caching and self.MODELS[model]['supports_caching']
        self.client = None
        self.genai = None

    def initialize(self):
        """Initialize Gemini client."""
        try:
            import google.generativeai as genai
            self.genai = genai
            self.genai.configure(api_key=self.api_key)
            self.client = self.genai.GenerativeModel(self.model)
            self._initialized = True
        except ImportError:
            raise TranslationError(
                "google-generativeai package not installed. Run: pip install google-generativeai",
                provider=self.name
            )

    async def translate(self, request: TranslationRequest) -> TranslationResponse:
        """
        Translate using Gemini API (async).

        Args:
            request: Translation request

        Returns:
            Translation response
        """
        if not self._initialized:
            self.initialize()

        results = []
        total_cost = 0.0

        # Build system instruction
        system_instruction = self._build_system_instruction(request)

        # For batch translation, combine all texts
        if len(request.units) > 1:
            # Build batch user message
            batch_texts = "\n\n".join([f"{i+1}. {unit.text}" for i, unit in enumerate(request.units)])
            user_content = f"""Translate the following {request.src_lang} sentences to {request.tgt_lang}.
Provide ONLY the translations, one per line, numbered to match the input.

{batch_texts}

Provide only the {request.tgt_lang} translations without explanations."""

            try:
                # Configure generation
                generation_config = self._get_generation_config(request)

                # Create model with system instruction if provided
                if system_instruction:
                    model = self.genai.GenerativeModel(
                        self.model,
                        system_instruction=system_instruction
                    )
                else:
                    model = self.client

                # Call API
                response = model.generate_content(
                    user_content,
                    generation_config=generation_config
                )

                # Extract translation
                translation_text = response.text

                # Get usage
                usage = self._get_usage(response)

                # Calculate cost
                cost = self._calculate_cost(usage)
                total_cost += cost

                # Parse batch translations
                translation_lines = translation_text.strip().split('\n')
                translations = []
                for line in translation_lines:
                    line = line.strip()
                    if line:
                        # Remove number prefix
                        import re
                        cleaned = re.sub(r'^\d+\.\s*', '', line)
                        translations.append(cleaned)

                # Match translations to units
                for i, unit in enumerate(request.units):
                    if i < len(translations):
                        results.append(TranslationResult(
                            source=unit.text,
                            translation=translations[i],
                            index=unit.index,
                            cost=cost / len(request.units),
                            status=TranslationStatus.COMPLETED,
                            metadata={
                                'model': self.model,
                                **usage
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
                # Build user prompt
                user_content = self._build_user_message(unit.text, request)

                # Configure generation
                generation_config = self._get_generation_config(request)

                # Create model with system instruction if provided
                if system_instruction:
                    model = self.genai.GenerativeModel(
                        self.model,
                        system_instruction=system_instruction
                    )
                else:
                    model = self.client

                # Call API
                response = model.generate_content(
                    user_content,
                    generation_config=generation_config
                )

                # Extract translation
                translation = response.text

                # Get usage
                usage = self._get_usage(response)

                # Calculate cost
                cost = self._calculate_cost(usage)

                results.append(TranslationResult(
                    source=unit.text,
                    translation=translation,
                    index=unit.index,
                    cost=cost,
                    status=TranslationStatus.COMPLETED,
                    metadata={
                        'model': self.model,
                        **usage
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

    def _build_system_instruction(self, request: TranslationRequest) -> Optional[str]:
        """Build system instruction with examples for caching."""
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

    def _get_generation_config(self, request: TranslationRequest):
        """Get generation configuration."""
        return self.genai.types.GenerationConfig(
            temperature=request.parameters.get('temperature', 0.3),
            max_output_tokens=request.parameters.get('max_tokens', 2048),
            top_p=request.parameters.get('top_p', 0.9)
        )

    def _get_usage(self, response) -> Dict[str, int]:
        """Extract usage statistics from response."""
        usage = {
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'total_tokens': 0,
            'cached_tokens': 0
        }

        if hasattr(response, 'usage_metadata'):
            metadata = response.usage_metadata
            usage['prompt_tokens'] = getattr(metadata, 'prompt_token_count', 0)
            usage['completion_tokens'] = getattr(metadata, 'candidates_token_count', 0)
            usage['total_tokens'] = getattr(metadata, 'total_token_count', 0)
            usage['cached_tokens'] = getattr(metadata, 'cached_content_token_count', 0)

        return usage

    def _calculate_cost(self, usage: Dict[str, int]) -> float:
        """Calculate cost from usage statistics."""
        model_spec = self.MODELS[self.model]

        # Get token counts
        prompt_tokens = usage.get('prompt_tokens', 0)
        completion_tokens = usage.get('completion_tokens', 0)
        cached_tokens = usage.get('cached_tokens', 0)

        # Calculate costs
        regular_input_tokens = prompt_tokens - cached_tokens
        input_cost = (regular_input_tokens / 1_000_000) * model_spec['input_cost']

        if self.use_caching and cached_tokens > 0:
            cached_cost = (cached_tokens / 1_000_000) * model_spec.get('cached_input_cost', model_spec['input_cost'] * 0.25)
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
            cached_input_cost = (system_tokens * (request.num_units - 1) / 1_000_000) * model_spec.get('cached_input_cost', model_spec['input_cost'] * 0.25)
            system_first_cost = (system_tokens / 1_000_000) * model_spec['input_cost']

            input_cost = regular_input_cost + cached_input_cost + system_first_cost
        else:
            input_cost = (total_input_tokens / 1_000_000) * model_spec['input_cost']

        output_cost = (avg_output_tokens / 1_000_000) * model_spec['output_cost']

        return CostEstimate(
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=input_cost + output_cost,
            currency="INR",
            num_units=request.num_units,
            estimated_tokens=total_input_tokens + avg_output_tokens,
            metadata={
                'model': self.model,
                'caching': self.use_caching,
                'estimated_system_tokens': system_tokens
            }
        )

    def supports_batch(self) -> bool:
        """Gemini supports batch processing."""
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
            'requests_per_minute': 60,
            'tokens_per_minute': 1000000
        }
