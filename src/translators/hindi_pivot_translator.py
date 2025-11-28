"""Hindi pivot translation for unsupported languages."""

from typing import List, Optional, Union
from .indictrans2_translator import IndicTrans2Translator
from ..utils.config import Config
from ..utils.logger import get_logger


logger = get_logger(__name__)


class HindiPivotTranslator:
    """
    Translator that uses Hindi as a pivot language for unsupported languages.

    This is useful for Indo-Aryan languages like Bhojpuri, Magahi, Awadhi, etc.
    that are linguistically close to Hindi but not directly supported by IndicTrans2.

    Translation path: English → Hindi → Target Language
    or: Target Language → Hindi → English
    """

    # Languages that benefit from Hindi pivot
    HINDI_PIVOT_LANGUAGES = {
        'bho': 'Bhojpuri',
        'mag': 'Magahi',
        'awa': 'Awadhi',
        'bra': 'Braj',
        'mwr': 'Marwari',
        'bns': 'Bundeli'
    }

    def __init__(
        self,
        config: Optional[Config] = None,
        device: Optional[str] = None,
        model_name: Optional[str] = None
    ):
        """
        Initialize Hindi pivot translator.

        Args:
            config: Configuration object. If None, creates a new one.
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
            model_name: Model to use. If None, uses config default.
        """
        self.config = config or Config()
        self.device = device

        # Initialize IndicTrans2 translator for the pivoting
        logger.info("Initializing Hindi pivot translator...")
        self.translator = IndicTrans2Translator(
            model_name=model_name,
            config=self.config,
            device=self.device
        )

        logger.success("✓ Hindi pivot translator initialized")

    def translate_via_hindi(
        self,
        texts: Union[str, List[str]],
        src_lang: str,
        tgt_lang: str,
        batch_size: Optional[int] = None,
        max_length: Optional[int] = None,
        num_beams: Optional[int] = None,
        show_progress: bool = True,
        return_intermediate: bool = False
    ) -> Union[str, List[str], dict]:
        """
        Translate via Hindi pivot.

        Args:
            texts: Text or list of texts to translate
            src_lang: Source language code ('en' or pivot language)
            tgt_lang: Target language code (pivot language or 'en')
            batch_size: Batch size for processing
            max_length: Maximum output length
            num_beams: Number of beams for beam search
            show_progress: Whether to show progress bar
            return_intermediate: If True, returns intermediate Hindi translations

        Returns:
            Translated text(s), or dict with translations and intermediate if return_intermediate=True
        """
        # Handle single string input
        single_input = isinstance(texts, str)
        if single_input:
            texts = [texts]

        logger.info(f"Translating {len(texts)} texts: {src_lang} → hi → {tgt_lang}")

        # Step 1: Translate to/from Hindi
        if src_lang == 'en':
            # English → Hindi
            logger.info("Step 1: English → Hindi")
            hindi_texts = self.translator.translate(
                texts,
                src_lang='en',
                tgt_lang='hi',
                batch_size=batch_size,
                max_length=max_length,
                num_beams=num_beams,
                show_progress=show_progress
            )

            # Step 2: Hindi → Target (manual post-processing would go here)
            # For now, we return Hindi as it's linguistically close
            logger.info(f"Step 2: Hindi → {tgt_lang} (using Hindi as approximation)")
            logger.warning(
                f"Note: Hindi is used as approximation for {tgt_lang}. "
                "Consider post-editing or API enhancement for better quality."
            )
            final_translations = hindi_texts

        elif tgt_lang == 'en':
            # Source → Hindi → English
            logger.info(f"Step 1: {src_lang} → Hindi (using Hindi as approximation)")
            logger.warning(
                f"Note: Treating {src_lang} input as Hindi for translation. "
                "Consider using API for actual {src_lang} → Hindi translation."
            )

            # Treat input as Hindi (linguistically close) and translate to English
            logger.info("Step 2: Hindi → English")
            final_translations = self.translator.translate(
                texts,
                src_lang='hi',
                tgt_lang='en',
                batch_size=batch_size,
                max_length=max_length,
                num_beams=num_beams,
                show_progress=show_progress
            )
            hindi_texts = texts  # Original texts treated as Hindi

        else:
            raise ValueError(
                f"Hindi pivot requires either src_lang='en' or tgt_lang='en'. "
                f"Got src_lang='{src_lang}', tgt_lang='{tgt_lang}'"
            )

        logger.success(f"✓ Pivot translation complete: {len(final_translations)} texts")

        # Return format
        if return_intermediate:
            result = {
                'translations': final_translations[0] if single_input else final_translations,
                'intermediate_hindi': hindi_texts[0] if single_input else hindi_texts
            }
            return result
        else:
            return final_translations[0] if single_input else final_translations

    def translate(
        self,
        texts: Union[str, List[str]],
        src_lang: str,
        tgt_lang: str,
        **kwargs
    ) -> Union[str, List[str]]:
        """
        Main translation interface.

        Args:
            texts: Text or list of texts to translate
            src_lang: Source language code
            tgt_lang: Target language code
            **kwargs: Additional arguments passed to translate_via_hindi()

        Returns:
            Translated text(s)
        """
        return self.translate_via_hindi(texts, src_lang, tgt_lang, **kwargs)

    def is_pivot_language(self, lang_code: str) -> bool:
        """
        Check if a language is a Hindi pivot language.

        Args:
            lang_code: Language code

        Returns:
            True if it's a pivot language, False otherwise
        """
        return lang_code in self.HINDI_PIVOT_LANGUAGES

    def get_pivot_languages(self) -> dict:
        """
        Get dictionary of supported pivot languages.

        Returns:
            Dictionary mapping language codes to names
        """
        return self.HINDI_PIVOT_LANGUAGES.copy()

    def get_language_info(self, lang_code: str) -> dict:
        """
        Get information about a pivot language.

        Args:
            lang_code: Language code

        Returns:
            Dictionary with language information
        """
        unsupported_langs = self.config.get_unsupported_languages()

        if lang_code in unsupported_langs:
            return unsupported_langs[lang_code]
        else:
            return {
                'code': lang_code,
                'name': self.HINDI_PIVOT_LANGUAGES.get(lang_code, 'Unknown'),
                'pivot': 'hi',
                'family': 'Indo-Aryan'
            }
