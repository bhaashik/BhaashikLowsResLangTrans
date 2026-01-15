"""HuggingFace Causal LM models (Llama, Mistral, etc.) for translation."""

from typing import List, Optional, Union
from pathlib import Path
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig
)

from .base import AbstractTrainableMTModel
from .registry import ModelRegistry
from ..config import ModelConfig, LoRAConfig


@ModelRegistry.register("hf_llm", model_type="causal_lm", description="Generic HuggingFace Causal LM model")
class HuggingFaceLLMModel(AbstractTrainableMTModel):
    """
    Wrapper for HuggingFace causal LM models used for translation.

    Supports: Llama, Mistral, Mixtral, and other instruction-following LLMs.
    """

    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)
        self.generation_config = None
        self.peft_model = None
        self.instruction_template = None

    def load(self):
        """Load model and tokenizer from HuggingFace."""
        print(f"Loading LLM: {self.config.model_name_or_path}")

        # Prepare loading arguments
        load_kwargs = {
            "cache_dir": self.config.cache_dir,
            "revision": self.config.model_revision,
            "use_auth_token": self.config.use_auth_token,
            "trust_remote_code": self.config.trust_remote_code,
            "low_cpu_mem_usage": self.config.low_cpu_mem_usage,
        }

        # Add quantization config
        if self.config.load_in_8bit or self.config.load_in_4bit:
            from transformers import BitsAndBytesConfig

            quantization_config = BitsAndBytesConfig(
                load_in_8bit=self.config.load_in_8bit,
                load_in_4bit=self.config.load_in_4bit,
                bnb_4bit_compute_dtype=getattr(torch, self.config.bnb_4bit_compute_dtype),
                bnb_4bit_use_double_quant=self.config.bnb_4bit_use_double_quant,
                bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
            )
            load_kwargs["quantization_config"] = quantization_config

        # Add torch dtype
        if self.config.torch_dtype:
            load_kwargs["torch_dtype"] = getattr(torch, self.config.torch_dtype)

        # Add device map
        if self.config.device_map:
            load_kwargs["device_map"] = self.config.device_map

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name_or_path,
            **load_kwargs
        )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.tokenizer_name_or_path,
            use_fast=self.config.use_fast_tokenizer,
            cache_dir=self.config.cache_dir,
            revision=self.config.tokenizer_revision,
        )

        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Setup instruction template
        self._setup_instruction_template()

        # Setup generation config
        self._setup_generation_config()

        self._loaded = True
        print(f"✓ LLM loaded successfully")

    def _setup_instruction_template(self):
        """Setup instruction template based on model architecture."""
        if self.config.architecture == "llama":
            # Llama 3 format
            self.instruction_template = (
                "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                "You are a helpful translation assistant. Translate the given text accurately.<|eot_id|>"
                "<|start_header_id|>user<|end_header_id|>\n\n"
                "Translate from {source_lang} to {target_lang}:\n{text}<|eot_id|>"
                "<|start_header_id|>assistant<|end_header_id|>\n\n"
            )
        elif self.config.architecture == "mistral":
            # Mistral format
            self.instruction_template = (
                "<s>[INST] Translate the following text from {source_lang} to {target_lang}:\n"
                "{text} [/INST]"
            )
        else:
            # Generic format
            self.instruction_template = (
                "Translate from {source_lang} to {target_lang}:\n"
                "Source: {text}\n"
                "Translation:"
            )

    def _setup_generation_config(self):
        """Setup generation configuration."""
        self.generation_config = GenerationConfig(
            max_new_tokens=self.config.max_length,
            num_beams=self.config.num_beams,
            do_sample=False,  # Deterministic for translation
            temperature=1.0,
            top_p=1.0,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

    def translate(
        self,
        texts: Union[str, List[str]],
        source_lang: Optional[str] = None,
        target_lang: Optional[str] = None,
        **kwargs
    ) -> Union[str, List[str]]:
        """
        Translate text(s) using LLM.

        Args:
            texts: Single text or list of texts
            source_lang: Source language name (e.g., "Hindi", "English")
            target_lang: Target language name
            **kwargs: Additional generation parameters

        Returns:
            Translated text(s)
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Handle single text
        is_single = isinstance(texts, str)
        if is_single:
            texts = [texts]

        # Get language names
        src_lang = source_lang or self._get_language_name(self.config.source_lang)
        tgt_lang = target_lang or self._get_language_name(self.config.target_lang)

        # Format prompts
        prompts = [
            self.instruction_template.format(
                source_lang=src_lang,
                target_lang=tgt_lang,
                text=text
            )
            for text in texts
        ]

        # Tokenize
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512  # Prompt + source text
        )

        # Move to device
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        # Generate
        generation_kwargs = {
            "max_new_tokens": kwargs.get("max_new_tokens", self.config.max_length),
            "num_beams": kwargs.get("num_beams", self.config.num_beams),
            "do_sample": kwargs.get("do_sample", False),
            "temperature": kwargs.get("temperature", 1.0),
            "top_p": kwargs.get("top_p", 1.0),
        }

        with torch.no_grad():
            generated_tokens = self.model.generate(
                **inputs,
                **generation_kwargs
            )

        # Decode only the generated part (skip input prompt)
        translations = []
        for i, generated in enumerate(generated_tokens):
            # Skip input tokens
            input_length = inputs["input_ids"][i].shape[0]
            generated_part = generated[input_length:]

            # Decode
            translation = self.tokenizer.decode(
                generated_part,
                skip_special_tokens=True
            ).strip()

            translations.append(translation)

        return translations[0] if is_single else translations

    def _get_language_name(self, code: str) -> str:
        """Convert language code to full name."""
        lang_names = {
            "hi": "Hindi",
            "en": "English",
            "bho": "Bhojpuri",
            "mag": "Magahi",
            "awa": "Awadhi",
            "bra": "Braj",
            "mwr": "Marwari",
            "bns": "Bundeli",
            "mai": "Maithili",
            "bn": "Bengali",
            "ta": "Tamil",
            "te": "Telugu",
        }
        return lang_names.get(code, code.capitalize())

    def prepare_for_training(
        self,
        lora_config: Optional[LoRAConfig] = None,
        training_config: Optional[Any] = None
    ):
        """
        Prepare LLM for training with LoRA.

        Args:
            lora_config: LoRA configuration (recommended for LLMs)
            training_config: Training configuration
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        if lora_config is not None:
            # Apply LoRA (recommended for LLMs)
            print("Applying LoRA adapters to LLM...")
            try:
                from peft import get_peft_model, prepare_model_for_kbit_training
            except ImportError:
                raise ImportError("PEFT library required. Install: pip install peft")

            # Prepare model for k-bit training if quantized
            if self.config.load_in_8bit or self.config.load_in_4bit:
                self.model = prepare_model_for_kbit_training(self.model)

            # Set task type to CAUSAL_LM if not set
            if lora_config.task_type != "CAUSAL_LM":
                lora_config.task_type = "CAUSAL_LM"

            # Get PEFT config
            peft_config = lora_config.to_peft_config()

            # Apply LoRA
            self.model = get_peft_model(self.model, peft_config)
            self.peft_model = self.model

            print("✓ LoRA applied successfully")
            self.print_trainable_parameters()
        else:
            # Full fine-tuning (not recommended for large LLMs)
            print("WARNING: Full fine-tuning LLMs requires significant memory!")
            print("Consider using LoRA instead.")
            self.unfreeze_all_parameters()
            print("✓ All parameters unfrozen")
            self.print_trainable_parameters()

    def save(self, output_dir: str, save_full_model: bool = False):
        """
        Save model (full model or adapters only).

        Args:
            output_dir: Output directory
            save_full_model: Save full model or just adapters
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if self.peft_model is not None and not save_full_model:
            # Save LoRA adapters only
            print(f"Saving LoRA adapters to {output_dir}")
            self.peft_model.save_pretrained(output_dir)
        else:
            # Save full model
            print(f"Saving full model to {output_dir}")
            self.model.save_pretrained(output_dir)

        # Always save tokenizer
        self.tokenizer.save_pretrained(output_dir)

        print("✓ Model saved successfully")

    def load_adapter(self, adapter_path: str):
        """
        Load LoRA adapter weights.

        Args:
            adapter_path: Path to adapter weights
        """
        print(f"Loading adapter from {adapter_path}")
        try:
            from peft import PeftModel
        except ImportError:
            raise ImportError("PEFT library required. Install: pip install peft")

        if not self._loaded:
            self.load()

        self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.peft_model = self.model

        print("✓ Adapter loaded successfully")

    def format_training_example(
        self,
        source_text: str,
        target_text: str,
        source_lang: Optional[str] = None,
        target_lang: Optional[str] = None
    ) -> dict:
        """
        Format training example for causal LM training.

        Args:
            source_text: Source text
            target_text: Target translation
            source_lang: Source language name
            target_lang: Target language name

        Returns:
            Dict with "input_ids", "attention_mask", "labels"
        """
        src_lang = source_lang or self._get_language_name(self.config.source_lang)
        tgt_lang = target_lang or self._get_language_name(self.config.target_lang)

        # Format prompt (instruction + source)
        prompt = self.instruction_template.format(
            source_lang=src_lang,
            target_lang=tgt_lang,
            text=source_text
        )

        # Full text: prompt + target
        full_text = prompt + target_text + self.tokenizer.eos_token

        # Tokenize
        tokenized = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.config.max_length,
            padding=False,
        )

        # Create labels (mask prompt tokens, only compute loss on target)
        prompt_tokenized = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.config.max_length,
            padding=False,
        )

        labels = tokenized["input_ids"].copy()
        prompt_length = len(prompt_tokenized["input_ids"])

        # Mask prompt tokens with -100 (ignore in loss)
        labels[:prompt_length] = [-100] * prompt_length

        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": labels,
        }
