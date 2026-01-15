"""IndicTrans2 translation pipeline."""

import torch
from typing import List, Optional, Union
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

from ..utils.config import Config
from ..utils.logger import get_logger


logger = get_logger(__name__)


class IndicTrans2Translator:
    """Translation pipeline using IndicTrans2 models."""

    # IndicTrans2 language codes
    FLORES_CODES = {
        'as': 'asm_Beng',  # Assamese
        'bn': 'ben_Beng',  # Bengali
        'gu': 'guj_Gujr',  # Gujarati
        'hi': 'hin_Deva',  # Hindi
        'kn': 'kan_Knda',  # Kannada
        'ks': 'kas_Arab',  # Kashmiri
        'kok': 'gom_Deva', # Konkani
        'ml': 'mal_Mlym',  # Malayalam
        'mni': 'mni_Mtei', # Manipuri
        'mr': 'mar_Deva',  # Marathi
        'ne': 'npi_Deva',  # Nepali
        'or': 'ory_Orya',  # Odia
        'pa': 'pan_Guru',  # Punjabi
        'sa': 'san_Deva',  # Sanskrit
        'sd': 'snd_Arab',  # Sindhi
        'ta': 'tam_Taml',  # Tamil
        'te': 'tel_Telu',  # Telugu
        'ur': 'urd_Arab',  # Urdu
        'brx': 'brx_Deva', # Bodo
        'sat': 'sat_Olck', # Santhali
        'mai': 'mai_Deva', # Maithili
        'doi': 'doi_Deva', # Dogri
        'en': 'eng_Latn',  # English
    }

    def __init__(
        self,
        model_name: Optional[str] = None,
        config: Optional[Config] = None,
        device: Optional[str] = None
    ):
        """
        Initialize IndicTrans2 translator.

        Args:
            model_name: HuggingFace model path. If None, uses config default.
            config: Configuration object. If None, creates a new one.
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
        """
        self.config = config or Config()

        # Set up device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Get model name
        if model_name is None:
            model_name = self.config.get_model_path('indictrans2', 'default')

        self.model_name = model_name
        logger.info(f"Initializing IndicTrans2Translator with {model_name}")
        logger.info(f"Device: {self.device}")

        # Load model and tokenizer
        self.tokenizer = None
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the model and tokenizer."""
        try:
            logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                cache_dir=self.config.transformers_cache
            )

            logger.info("Loading model...")
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                cache_dir=self.config.transformers_cache
            )

            self.model.to(self.device)
            self.model.eval()

            logger.success(f"✓ Model loaded successfully on {self.device}")

        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise

    def _get_flores_code(self, lang_code: str) -> str:
        """
        Convert language code to FLORES-200 format.

        Args:
            lang_code: Language code (e.g., 'hi', 'bn')

        Returns:
            FLORES-200 code (e.g., 'hin_Deva', 'ben_Beng')
        """
        if lang_code not in self.FLORES_CODES:
            raise ValueError(f"Unsupported language code: {lang_code}")
        return self.FLORES_CODES[lang_code]

    def translate(
        self,
        texts: Union[str, List[str]],
        src_lang: str,
        tgt_lang: str,
        batch_size: Optional[int] = None,
        max_length: Optional[int] = None,
        num_beams: Optional[int] = None,
        show_progress: bool = True
    ) -> Union[str, List[str]]:
        """
        Translate texts from source to target language.

        Args:
            texts: Text or list of texts to translate
            src_lang: Source language code (e.g., 'en', 'hi')
            tgt_lang: Target language code (e.g., 'hi', 'bn')
            batch_size: Batch size for processing. If None, uses config default.
            max_length: Maximum output length. If None, uses config default.
            num_beams: Number of beams for beam search. If None, uses config default.
            show_progress: Whether to show progress bar

        Returns:
            Translated text(s)
        """
        # Handle single string input
        single_input = isinstance(texts, str)
        if single_input:
            texts = [texts]

        # Get parameters
        batch_size = batch_size or self.config.batch_size
        max_length = max_length or self.config.max_length
        num_beams = num_beams or self.config.num_beams

        # Get FLORES codes
        src_flores = self._get_flores_code(src_lang)
        tgt_flores = self._get_flores_code(tgt_lang)

        logger.info(f"Translating {len(texts)} texts from {src_lang} to {tgt_lang}")
        logger.info(f"Batch size: {batch_size}, Max length: {max_length}, Beams: {num_beams}")

        translations = []

        # Process in batches
        num_batches = (len(texts) + batch_size - 1) // batch_size
        iterator = range(0, len(texts), batch_size)

        if show_progress:
            iterator = tqdm(iterator, total=num_batches, desc="Translating")

        with torch.no_grad():
            for i in iterator:
                batch = texts[i:i + batch_size]

                # Prepare inputs with source language tag
                inputs = [f"{src_flores}: {text}" for text in batch]

                # Tokenize
                encoded = self.tokenizer(
                    inputs,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt"
                ).to(self.device)

                # Generate translations with target language forcing
                generated = self.model.generate(
                    **encoded,
                    forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(tgt_flores),
                    max_length=max_length,
                    num_beams=num_beams,
                    num_return_sequences=1
                )

                # Decode
                batch_translations = self.tokenizer.batch_decode(
                    generated,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )

                translations.extend(batch_translations)

        logger.success(f"✓ Translated {len(translations)} texts")

        # Return single string if input was single string
        if single_input:
            return translations[0]

        return translations

    def translate_batch(
        self,
        texts: List[str],
        src_lang: str,
        tgt_lang: str,
        **kwargs
    ) -> List[str]:
        """
        Convenience method for batch translation.

        Args:
            texts: List of texts to translate
            src_lang: Source language code
            tgt_lang: Target language code
            **kwargs: Additional arguments passed to translate()

        Returns:
            List of translated texts
        """
        return self.translate(texts, src_lang, tgt_lang, **kwargs)

    def is_supported(self, lang_code: str) -> bool:
        """
        Check if a language is supported by IndicTrans2.

        Args:
            lang_code: Language code

        Returns:
            True if supported, False otherwise
        """
        return lang_code in self.FLORES_CODES

    def get_supported_languages(self) -> List[str]:
        """
        Get list of supported language codes.

        Returns:
            List of language codes
        """
        return list(self.FLORES_CODES.keys())
