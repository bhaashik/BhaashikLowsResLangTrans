"""HuggingFace Seq2Seq models (NLLB, mBART, mT5, IndicTrans2)."""

from typing import List, Optional, Union
from pathlib import Path
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    GenerationConfig
)

from .base import AbstractTrainableMTModel
from .registry import ModelRegistry
from ..config import ModelConfig, LoRAConfig


@ModelRegistry.register("hf_seq2seq", model_type="seq2seq", description="Generic HuggingFace Seq2Seq model")
class HuggingFaceSeq2SeqModel(AbstractTrainableMTModel):
    """
    Generic wrapper for HuggingFace seq2seq translation models.

    Supports: NLLB, mBART, mT5, IndicTrans2, and other seq2seq architectures.
    """

    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)
        self.generation_config = None
        self.peft_model = None  # For LoRA

    def load(self):
        """Load model and tokenizer from HuggingFace."""
        print(f"Loading model: {self.config.model_name_or_path}")

        # Prepare loading arguments
        load_kwargs = {
            "cache_dir": self.config.cache_dir,
            "revision": self.config.model_revision,
            "use_auth_token": self.config.use_auth_token,
            "trust_remote_code": self.config.trust_remote_code,
            "low_cpu_mem_usage": self.config.low_cpu_mem_usage,
        }

        # Add quantization config if specified
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
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
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

        # Set generation config
        self._setup_generation_config()

        self._loaded = True
        print(f"✓ Model loaded successfully")

    def _setup_generation_config(self):
        """Setup generation configuration."""
        self.generation_config = GenerationConfig(
            max_length=self.config.max_length,
            min_length=self.config.min_length,
            num_beams=self.config.num_beams,
            length_penalty=self.config.length_penalty,
            forced_bos_token_id=self.config.forced_bos_token_id,
            forced_eos_token_id=self.config.forced_eos_token_id,
        )

    def translate(
        self,
        texts: Union[str, List[str]],
        source_lang: Optional[str] = None,
        target_lang: Optional[str] = None,
        **kwargs
    ) -> Union[str, List[str]]:
        """
        Translate text(s).

        Args:
            texts: Single text or list of texts
            source_lang: Source language code (overrides config)
            target_lang: Target language code (overrides config)
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

        # Get language codes
        src_lang = source_lang or self.config.source_lang
        tgt_lang = target_lang or self.config.target_lang

        # Prepare inputs based on model architecture
        inputs = self._prepare_inputs(texts, src_lang, tgt_lang)

        # Set forced tokens for target language if applicable
        generation_kwargs = self._prepare_generation_kwargs(tgt_lang, **kwargs)

        # Generate translations
        with torch.no_grad():
            generated_tokens = self.model.generate(**inputs, **generation_kwargs)

        # Decode translations
        translations = self.tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True
        )

        return translations[0] if is_single else translations

    def _prepare_inputs(self, texts: List[str], src_lang: str, tgt_lang: str) -> dict:
        """Prepare model inputs based on architecture."""
        # Add prefix for mT5-style models
        if self.config.add_prefix:
            if self.config.prefix_template:
                prefix = self.config.prefix_template.format(
                    source=src_lang, target=tgt_lang
                )
            else:
                prefix = f"translate {src_lang} to {tgt_lang}: "
            texts = [prefix + text for text in texts]

        # Tokenize
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_length
        )

        # Move to device
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        return inputs

    def _prepare_generation_kwargs(self, tgt_lang: str, **kwargs) -> dict:
        """Prepare generation kwargs with forced tokens if applicable."""
        generation_kwargs = {
            "max_length": self.generation_config.max_length,
            "num_beams": self.generation_config.num_beams,
            "length_penalty": self.generation_config.length_penalty,
        }

        # Override with user-provided kwargs
        generation_kwargs.update(kwargs)

        # Set forced BOS token for NLLB/mBART models
        if self.config.architecture in ["nllb", "mbart"]:
            tgt_lang_code = self._get_language_code(tgt_lang, is_target=True)
            if tgt_lang_code in self.tokenizer.lang_code_to_id:
                generation_kwargs["forced_bos_token_id"] = self.tokenizer.lang_code_to_id[tgt_lang_code]

        return generation_kwargs

    def _get_language_code(self, lang: str, is_target: bool = False) -> str:
        """Get model-specific language code."""
        # Use config-specified codes if available
        if is_target and self.config.target_lang_code:
            return self.config.target_lang_code
        elif not is_target and self.config.source_lang_code:
            return self.config.source_lang_code

        # Auto-convert based on architecture
        if self.config.architecture == "nllb":
            # NLLB format: "hin_Deva", "eng_Latn"
            lang_map = {
                "hi": "hin_Deva", "en": "eng_Latn", "bho": "bho_Deva",
                "mag": "mag_Deva", "awa": "awa_Deva", "mai": "mai_Deva",
                "bn": "ben_Beng", "ta": "tam_Taml", "te": "tel_Telu",
            }
            return lang_map.get(lang, lang)
        elif self.config.architecture == "indictrans2":
            # IndicTrans2 format: "hin_Deva", "eng_Latn"
            return self._get_language_code_indictrans2(lang)
        else:
            # Default: use as-is
            return lang

    def _get_language_code_indictrans2(self, lang: str) -> str:
        """Get IndicTrans2-specific language code."""
        lang_map = {
            "hi": "hin_Deva", "en": "eng_Latn",
            "bn": "ben_Beng", "gu": "guj_Gujr", "kn": "kan_Knda",
            "ml": "mal_Mlym", "mr": "mar_Deva", "or": "ori_Orya",
            "pa": "pan_Guru", "ta": "tam_Taml", "te": "tel_Telu",
            "ur": "urd_Arab", "as": "asm_Beng", "ks": "kas_Arab",
            "ne": "npi_Deva", "sa": "san_Deva", "sd": "snd_Arab",
        }
        return lang_map.get(lang, lang)

    def prepare_for_training(
        self,
        lora_config: Optional[LoRAConfig] = None,
        training_config: Optional[Any] = None
    ):
        """
        Prepare model for training with LoRA or full fine-tuning.

        Args:
            lora_config: LoRA configuration (None for full fine-tuning)
            training_config: Training configuration
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        if lora_config is not None:
            # Apply LoRA
            print("Applying LoRA adapters...")
            try:
                from peft import get_peft_model, prepare_model_for_kbit_training
            except ImportError:
                raise ImportError("PEFT library required. Install: pip install peft")

            # Prepare model for k-bit training if quantized
            if self.config.load_in_8bit or self.config.load_in_4bit:
                self.model = prepare_model_for_kbit_training(self.model)

            # Get PEFT config
            peft_config = lora_config.to_peft_config()

            # Apply LoRA
            self.model = get_peft_model(self.model, peft_config)
            self.peft_model = self.model

            print("✓ LoRA applied successfully")
            self.print_trainable_parameters()
        else:
            # Full fine-tuning
            print("Preparing for full fine-tuning...")
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
