#!/usr/bin/env python3
"""
Example 5: Corpus Translation with Directory Structure Preservation

This example demonstrates how to translate entire corpus structures,
including Universal Dependencies (CoNLL-U) and plain text corpora.
"""

import sys
from pathlib import Path
import tempfile
import shutil

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import setup_logger


def create_sample_corpus():
    """Create sample corpus for demonstration."""
    # Create temporary directory
    temp_dir = Path(tempfile.mkdtemp(prefix='corpus_example_'))

    # Create directory structure
    corpus_dir = temp_dir / 'sample_corpus'
    corpus_dir.mkdir()

    # Create subdirectories
    (corpus_dir / 'train').mkdir()
    (corpus_dir / 'dev').mkdir()
    (corpus_dir / 'test').mkdir()

    # Create sample plain text files
    train_texts = [
        "Hello, how are you?",
        "What is your name?",
        "I am learning a new language.",
        "Thank you very much!",
        "Have a great day!"
    ]

    dev_texts = [
        "Good morning!",
        "See you later.",
        "This is wonderful."
    ]

    test_texts = [
        "Welcome to the program.",
        "Let's get started."
    ]

    # Write plain text files
    with open(corpus_dir / 'train' / 'data.txt', 'w') as f:
        f.write('\n'.join(train_texts))

    with open(corpus_dir / 'dev' / 'data.txt', 'w') as f:
        f.write('\n'.join(dev_texts))

    with open(corpus_dir / 'test' / 'data.txt', 'w') as f:
        f.write('\n'.join(test_texts))

    # Create sample CoNLL-U file
    conllu_content = """# sent_id = 1
# text = Hello world
1\tHello\thello\tINTJ\t_\t_\t0\troot\t_\t_
2\tworld\tworld\tNOUN\t_\t_\t1\tvocative\t_\t_

# sent_id = 2
# text = How are you
1\tHow\thow\tADV\t_\t_\t2\tadvmod\t_\t_
2\tare\tbe\tVERB\t_\t_\t0\troot\t_\t_
3\tyou\tyou\tPRON\t_\t_\t2\tnsubj\t_\t_

"""

    with open(corpus_dir / 'train' / 'sample.conllu', 'w') as f:
        f.write(conllu_content)

    return corpus_dir


def main():
    # Setup logging
    setup_logger(level='INFO', console=True)

    print("="*70)
    print("EXAMPLE 5: CORPUS TRANSLATION")
    print("="*70)

    # Create sample corpus
    print("\nCreating sample corpus...")
    corpus_dir = create_sample_corpus()
    output_dir = corpus_dir.parent / 'translated_corpus'

    print(f"Sample corpus created at: {corpus_dir}")
    print(f"Output will be saved to: {output_dir}")

    # Show directory structure
    print("\nOriginal directory structure:")
    print("sample_corpus/")
    print("├── train/")
    print("│   ├── data.txt")
    print("│   └── sample.conllu")
    print("├── dev/")
    print("│   └── data.txt")
    print("└── test/")
    print("    └── data.txt")

    # Example 1: Translate plain text corpus
    print("\n" + "-"*70)
    print("Example 1: Translating Plain Text Corpus")
    print("-"*70)

    print("\nCommand to translate English → Hindi:")
    print(f"""
python scripts/translate_corpus.py \\
    --input-dir {corpus_dir} \\
    --output-dir {output_dir} \\
    --src en --tgt hi \\
    --format txt \\
    --indictrans2
""")

    print("\nThis will:")
    print("  1. Find all .txt files in the directory structure")
    print("  2. Translate each file using IndicTrans2 (FREE)")
    print("  3. Preserve the directory structure in output")
    print("  4. Create translation_metadata.json with statistics")

    # Example 2: Translate CoNLL-U corpus
    print("\n" + "-"*70)
    print("Example 2: Translating CoNLL-U Corpus (Universal Dependencies)")
    print("-"*70)

    print("\nCommand to translate UD corpus:")
    print("""
python scripts/translate_corpus.py \\
    --input-dir /path/to/UD_Hindi \\
    --output-dir /path/to/UD_Bhojpuri \\
    --src hi --tgt bho \\
    --format conllu \\
    --hindi-pivot
""")

    print("\nFor CoNLL-U files, the script:")
    print("  1. Preserves all token annotations (POS, dependencies, etc.)")
    print("  2. Updates the '# text = ...' metadata with translation")
    print("  3. Keeps original text as '# text_src = ...'")
    print("  4. Maintains sentence structure and IDs")

    # Example 3: Tiered translation for quality
    print("\n" + "-"*70)
    print("Example 3: Tiered Translation for Better Quality")
    print("-"*70)

    print("\nCommand for tiered strategy:")
    print("""
python scripts/translate_corpus.py \\
    --input-dir corpus/en \\
    --output-dir corpus/bho \\
    --src en --tgt bho \\
    --format txt \\
    --tiered \\
    --tier-free 80 \\
    --tier-enhancement 15 \\
    --tier-premium 5
""")

    print("\nThis uses:")
    print("  - 80% Hindi pivot (FREE)")
    print("  - 15% Claude Haiku 3.5 enhancement")
    print("  - 5% Claude Haiku 4.5 premium")

    # Example 4: Resume interrupted translation
    print("\n" + "-"*70)
    print("Example 4: Resume Interrupted Translation")
    print("-"*70)

    print("\nBy default, the script resumes automatically:")
    print("""
# First run (interrupted)
python scripts/translate_corpus.py --input-dir corpus --output-dir output ...

# Resume (skips already translated files)
python scripts/translate_corpus.py --input-dir corpus --output-dir output ...

# Force retranslation
python scripts/translate_corpus.py --input-dir corpus --output-dir output ... --no-resume
""")

    # Example 5: Pattern-based selection
    print("\n" + "-"*70)
    print("Example 5: Translate Specific Files Only")
    print("-"*70)

    print("\nTranslate only training files:")
    print("""
python scripts/translate_corpus.py \\
    --input-dir UD_Hindi \\
    --output-dir UD_Bhojpuri \\
    --src hi --tgt bho \\
    --format conllu \\
    --pattern 'train/*.conllu' \\
    --indictrans2
""")

    # Real-world examples
    print("\n" + "-"*70)
    print("REAL-WORLD USE CASES")
    print("-"*70)

    print("""
1. Universal Dependencies Translation:
   # Translate entire UD treebank
   python scripts/translate_corpus.py \\
       --input-dir ~/UD_Hindi-HDTB \\
       --output-dir ~/UD_Bhojpuri-HDTB \\
       --src hi --tgt bho \\
       --format conllu \\
       --tiered

2. Parallel Corpus Creation:
   # Create parallel corpus for MT training
   python scripts/translate_corpus.py \\
       --input-dir raw_data/en \\
       --output-dir parallel_data/hi \\
       --src en --tgt hi \\
       --format txt \\
       --indictrans2 \\
       --batch-size 64

3. Multi-domain Translation:
   # Translate corpus with multiple domains
   # Structure: corpus/{news,social,formal}/files.txt
   python scripts/translate_corpus.py \\
       --input-dir corpus \\
       --output-dir corpus_translated \\
       --src en --tgt bho \\
       --format txt \\
       --tiered

4. Low-Resource Language Expansion:
   # Expand UD coverage to low-resource language
   for src_lang in hi bn mr; do
       python scripts/translate_corpus.py \\
           --input-dir UD_${src_lang} \\
           --output-dir UD_Bhojpuri_from_${src_lang} \\
           --src ${src_lang} --tgt bho \\
           --format conllu \\
           --hindi-pivot
   done
""")

    # Output structure
    print("\n" + "-"*70)
    print("OUTPUT STRUCTURE")
    print("-"*70)

    print("""
After translation, the output directory will contain:

translated_corpus/
├── train/
│   ├── data.txt              # Translated sentences
│   └── sample.conllu         # Translated with preserved annotations
├── dev/
│   └── data.txt
├── test/
│   └── data.txt
└── translation_metadata.json # Translation statistics

The metadata file contains:
  - Source and target languages
  - Translation method used
  - File-by-file statistics
  - Total sentences translated
  - Timestamp and configuration
""")

    # CoNLL-U example
    print("\n" + "-"*70)
    print("CoNLL-U TRANSLATION EXAMPLE")
    print("-"*70)

    print("""
Original CoNLL-U:
    # sent_id = 1
    # text = Hello world
    1    Hello    hello    INTJ    _    _    0    root       _    _
    2    world    world    NOUN    _    _    1    vocative   _    _

Translated CoNLL-U (English → Hindi):
    # sent_id = 1
    # text_en = Hello world
    # text = नमस्ते दुनिया
    1    Hello    hello    INTJ    _    _    0    root       _    _
    2    world    world    NOUN    _    _    1    vocative   _    _

Note: Token annotations are preserved, only text is translated.
""")

    # Tips
    print("\n" + "-"*70)
    print("TIPS AND BEST PRACTICES")
    print("-"*70)

    print("""
1. Start with small samples:
   - Test on a subset first (--pattern 'test/*.txt')
   - Verify quality before translating entire corpus

2. Choose appropriate translator:
   - IndicTrans2: Best for 22 supported languages (FREE)
   - Hindi Pivot: For low-resource Indo-Aryan languages (FREE)
   - NLLB: Alternative with broader coverage (FREE)
   - Tiered: Best quality for low-resource (costs money)

3. Monitor progress:
   - Use --log-level INFO for detailed progress
   - Check translation_metadata.json for statistics
   - Resume capability allows interrupting large jobs

4. Batch size optimization:
   - GPU: --batch-size 32-64
   - CPU: --batch-size 8-16
   - API: --batch-size 10-20

5. Cost estimation for tiered:
   - Estimate first: use translate.py --estimate-only
   - Start with higher free percentage (90/8/2)
   - Adjust based on quality assessment
""")

    # Cleanup
    print("\n" + "-"*70)
    print(f"\nSample corpus created at: {corpus_dir}")
    print("To actually run the translation, use the commands shown above.")
    print(f"\nTo clean up: rm -rf {corpus_dir.parent}")

    print("\n" + "="*70)
    print("EXAMPLE COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
