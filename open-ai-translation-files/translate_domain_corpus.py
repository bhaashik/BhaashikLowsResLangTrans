#!/usr/bin/env python3
"""
Translate Hindi to Bhojpuri and Magahi using GPT-5 Nano
Handles domain-organized corpus with multiple files per domain
Excludes temporary and backup files
Cost: ₹122 total (~$1.44) for 70,000 sentences
Speed: Ultra-fast with GPT-5 Nano
"""

import sys
import os
from pathlib import Path
from typing import List, Dict, Tuple, Set
import time
import json
from datetime import datetime

# Check dependencies
try:
    from openai import OpenAI
except ImportError:
    print("❌ Missing openai package")
    print("Install with: pip install openai")
    sys.exit(1)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Files to exclude (temporary and backup files)
EXCLUDED_EXTENSIONS = {
    '.swp',   # Vim swap files
    '.tmp',   # Temporary files
    '.bak',   # Backup files
    '.old',   # Old versions
    '.backup',# Backup files
    '~',      # Emacs backup files
    '.pyc',   # Python compiled
    '.log',   # Log files
    '.cache', # Cache files
}

# Language-specific linguistic prompts
SYSTEM_PROMPTS = {
    "Bhojpuri": """You are an expert translator specializing in Hindi to Bhojpuri translation.

Bhojpuri (भोजपुरी) is an Indo-Aryan language spoken primarily in eastern Uttar Pradesh and western Bihar, with ~51 million speakers. It is closely related to Hindi but has distinct vocabulary, grammar, and phonology.

TRANSLATION GUIDELINES:
1. Maintain natural, colloquial Bhojpuri (not formal/literary style)
2. Use authentic Bhojpuri vocabulary (not just Devanagari Hindi words)
3. Follow Bhojpuri grammatical patterns strictly
4. Preserve the meaning and cultural context
5. Use Devanagari script
6. Output ONLY the Bhojpuri translation, no explanations

KEY BHOJPURI LINGUISTIC FEATURES:
- Copula: "बा/बाटे/बानी" (bā/bāṭe/bānī) instead of है (hai)
- Progressive: "रहल" (rahal) for रहा (rahā)
- Pronouns: "हम" (ham) for मैं (maiṃ), "रउवा" (rauvā) for आप (āp)
- Past tense: "गइल/गइलै" (gail/gailai) for गया (gayā)
- Future: Uses present tense forms or "ई/ईहै" (ī/īhai)
- Possessive: "के" (ke) or "हमार" (hamār) for मेरा (merā)
- Negation: "ना/नइखे" (nā/naikhe)

ESSENTIAL VOCABULARY:
- आज (āj) → आज (āj) - today
- कल (kal) → कालहि (kālhi) - yesterday/tomorrow
- यह (yah) → ई/इ (ī/i) - this
- वह (vah) → ऊ/उ (ū/u) - that
- क्या (kyā) → का (kā) - what
- अच्छा (acchā) → बढ़िया/नीमन (baṛhiyā/nīman) - good
- बहुत (bahut) → बहुत/खूब (bahut/khūb) - very
- खाना (khānā) → खाना/खईनी (khānā/khaīnī) - to eat
- जाना (jānā) → जाना/जाईल (jānā/jāīl) - to go

TRANSLATION EXAMPLES:

Hindi: मैं आज बाजार जा रहा हूं।
Bhojpuri: हम आज बजार जाइत बानी।

Hindi: यह किताब बहुत अच्छी है।
Bhojpuri: ई किताब बहुत बढ़िया बा।

Hindi: क्या आप मेरी मदद कर सकते हैं?
Bhojpuri: का रउवा हमार मदद क सकेनी?

Hindi: मुझे भूख लगी है।
Bhojpuri: हमका भूख लागल बा।

Hindi: वह कल आएगा।
Bhojpuri: ऊ कालहि आई।

REMEMBER: Output ONLY the Bhojpuri translation, nothing else.""",

    "Magahi": """You are an expert translator specializing in Hindi to Magahi translation.

Magahi (मगही) is an Indo-Aryan language spoken primarily in the Magadh region of Bihar, with ~13 million speakers. It is closely related to Hindi and Bhojpuri but has unique grammatical and phonological characteristics.

TRANSLATION GUIDELINES:
1. Maintain natural, colloquial Magahi (not formal/literary style)
2. Use authentic Magahi vocabulary
3. Follow Magahi grammatical patterns strictly
4. Preserve the meaning and cultural context
5. Use Devanagari script
6. Output ONLY the Magahi translation, no explanations

KEY MAGAHI LINGUISTIC FEATURES:
- Copula: "छै/छथिन/छी" (chai/chathin/chī) instead of है (hai)
- Progressive: "रहल" (rahal) for रहा (rahā)
- Pronouns: "हम" (ham) for मैं (maiṃ), "तोहरा" (tohrā) for आप (āp)
- Past tense: "गेल/गेलै" (gel/gelai) for गया (gayā)
- Verbal suffix: "केलऽ" (kelõ) for किया (kiyā)
- Future: "तै/तथिन" (tai/tathin)
- Possessive: "के" (ke) or "हमर" (hamar) for मेरा (merā)
- Negation: "नै/नइ" (nai/nai)

ESSENTIAL VOCABULARY:
- आज (āj) → आज (āj) - today
- कल (kal) → कालहि/काल (kālhi/kāl) - yesterday/tomorrow
- यह (yah) → ई/इ (ī/i) - this
- वह (vah) → ऊ/उ (ū/u) - that
- क्या (kyā) → की (kī) - what
- अच्छा (acchā) → बढ़िया/नीक (baṛhiyā/nīk) - good
- बहुत (bahut) → बहुत (bahut) - very
- खाना (khānā) → खाना (khānā) - to eat
- जाना (jānā) → जाना (jānā) - to go

TRANSLATION EXAMPLES:

Hindi: मैं आज बाजार जा रहा हूं।
Magahi: हम आज बजार जाइत छी।

Hindi: यह किताब बहुत अच्छी है।
Magahi: ई किताब बहुत बढ़िया छै।

Hindi: क्या आप मेरी मदद कर सकते हैं?
Magahi: की तोहरा हमर मदद क सकै छी?

Hindi: मुझे भूख लगी है।
Magahi: हमका भूख लागल छै।

Hindi: वह कल आएगा।
Magahi: ऊ कालहि आतै।

REMEMBER: Output ONLY the Magahi translation, nothing else."""
}

# ============================================================================
# TRANSLATION ENGINE
# ============================================================================

class DomainTranslator:
    """Translate domain-organized corpus while preserving structure"""
    
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-5-nano"  # Ultra-fast, ultra-cheap
        self.stats = {
            "files_processed": 0,
            "files_skipped": 0,
            "sentences_translated": 0,
            "errors": 0,
            "domains_processed": set(),
            "start_time": None,
            "end_time": None
        }
    
    def should_process_file(self, file_path: Path) -> bool:
        """
        Check if file should be processed (not temp/backup)
        
        Args:
            file_path: Path to check
            
        Returns:
            True if file should be processed
        """
        # Check extension
        if file_path.suffix.lower() in EXCLUDED_EXTENSIONS:
            return False
        
        # Check for files ending with ~
        if file_path.name.endswith('~'):
            return False
        
        # Check for hidden files (starting with .)
        if file_path.name.startswith('.'):
            return False
        
        return True
    
    def translate_batch(
        self,
        sentences: List[str],
        target_language: str,
        batch_size: int = 50
    ) -> List[str]:
        """
        Translate a batch of sentences
        
        Args:
            sentences: List of Hindi sentences
            target_language: "Bhojpuri" or "Magahi"
            batch_size: Number of sentences per API call
            
        Returns:
            List of translations
        """
        
        all_translations = []
        total_batches = (len(sentences) + batch_size - 1) // batch_size
        
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i+batch_size]
            batch_num = (i // batch_size) + 1
            
            try:
                # Combine sentences with line numbers for batch processing
                batch_text = "\n".join([f"{idx+1}. {sent}" 
                                       for idx, sent in enumerate(batch)])
                
                # Call GPT-5 Nano
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": SYSTEM_PROMPTS[target_language]
                        },
                        {
                            "role": "user",
                            "content": f"Translate these {len(batch)} numbered Hindi sentences to {target_language}. Preserve the numbering:\n\n{batch_text}"
                        }
                    ],
                    temperature=0.3,
                    max_tokens=2000
                )
                
                # Extract translations
                translated_text = response.choices[0].message.content.strip()
                
                # Parse numbered translations
                translations = []
                for line in translated_text.split('\n'):
                    line = line.strip()
                    if line and not line.startswith('#'):
                        # Remove numbering if present
                        if '. ' in line:
                            parts = line.split('. ', 1)
                            if len(parts) == 2 and parts[0].isdigit():
                                translations.append(parts[1])
                            else:
                                translations.append(line)
                        else:
                            translations.append(line)
                
                # Ensure we have the right number of translations
                if len(translations) != len(batch):
                    print(f"    ⚠️  Warning: Expected {len(batch)} translations, got {len(translations)}")
                    # Pad or trim to match
                    while len(translations) < len(batch):
                        translations.append("[MISSING TRANSLATION]")
                    translations = translations[:len(batch)]
                
                all_translations.extend(translations)
                
                # Progress update
                progress = min(i + batch_size, len(sentences))
                percent = (progress / len(sentences)) * 100
                print(f"       Progress: {progress:,}/{len(sentences):,} ({percent:.1f}%) - Batch {batch_num}/{total_batches}")
                
                # Small delay to avoid rate limits
                time.sleep(0.1)
                
            except Exception as e:
                print(f"    ❌ Error in batch {batch_num}: {e}")
                # Add error placeholders
                all_translations.extend(["[TRANSLATION ERROR]"] * len(batch))
                self.stats["errors"] += len(batch)
        
        return all_translations
    
    def process_file(
        self,
        input_path: Path,
        output_path: Path,
        target_language: str,
        domain: str
    ) -> bool:
        """
        Translate a single file
        
        Args:
            input_path: Path to Hindi input file
            output_path: Path to output file
            target_language: "Bhojpuri" or "Magahi"
            domain: Domain name (for statistics)
            
        Returns:
            Success status
        """
        
        # Check if we should process this file
        if not self.should_process_file(input_path):
            print(f"    ⏭️  Skipped (temp/backup): {input_path.name}")
            self.stats["files_skipped"] += 1
            return True
        
        try:
            # Read Hindi sentences
            with open(input_path, 'r', encoding='utf-8') as f:
                hindi_sentences = [line.strip() for line in f if line.strip()]
            
            if not hindi_sentences:
                print(f"    ⚠️  Empty file: {input_path.name}")
                return True
            
            print(f"\n     📄 File: {input_path.name}")
            print(f"        Sentences: {len(hindi_sentences):,}")
            print(f"        Domain: {domain}")
            
            # Translate
            translations = self.translate_batch(hindi_sentences, target_language)
            
            # Create output directory if needed
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write translations
            with open(output_path, 'w', encoding='utf-8') as f:
                for translation in translations:
                    f.write(translation + '\n')
            
            print(f"        ✅ Saved: {output_path.relative_to(output_path.parents[2])}")
            
            self.stats["files_processed"] += 1
            self.stats["sentences_translated"] += len(hindi_sentences)
            self.stats["domains_processed"].add(domain)
            
            return True
            
        except Exception as e:
            print(f"    ❌ Error processing {input_path}: {e}")
            self.stats["errors"] += 1
            return False
    
    def translate_domain_corpus(
        self,
        source_dir: Path,
        output_base: Path,
        target_language: str,
        file_pattern: str = "*.txt"
    ):
        """
        Translate domain-organized corpus, preserving structure
        
        Args:
            source_dir: Root directory of Hindi files
            output_base: Root directory for output
            target_language: "Bhojpuri" or "Magahi"
            file_pattern: Glob pattern for files to process (default: *.txt)
        """
        
        print(f"\n{'='*70}")
        print(f"TRANSLATING: HINDI → {target_language.upper()}")
        print(f"{'='*70}")
        print(f"Source: {source_dir}")
        print(f"Output: {output_base}")
        print(f"Pattern: {file_pattern}")
        print(f"Model: {self.model} (GPT-5 Nano - Ultra Fast)")
        print(f"Excluded: {', '.join(sorted(EXCLUDED_EXTENSIONS))}")
        print(f"{'='*70}")
        
        self.stats["start_time"] = datetime.now()
        
        # Find all files recursively
        all_files = list(source_dir.rglob(file_pattern))
        
        # Filter out temp/backup files
        files_to_process = [f for f in all_files if self.should_process_file(f)]
        
        if not files_to_process:
            print(f"\n❌ No valid files found matching {file_pattern} in {source_dir}")
            if len(all_files) > 0:
                print(f"   ({len(all_files)} files found but all excluded)")
            return
        
        # Organize by domain
        domain_files = {}
        for file_path in files_to_process:
            # Get domain (immediate parent directory)
            domain = file_path.parent.name
            if domain not in domain_files:
                domain_files[domain] = []
            domain_files[domain].append(file_path)
        
        print(f"\n📊 CORPUS OVERVIEW")
        print(f"{'='*70}")
        print(f"Total files: {len(files_to_process):,}")
        print(f"Skipped files: {len(all_files) - len(files_to_process):,}")
        print(f"Domains: {len(domain_files)}")
        print(f"\nDomain breakdown:")
        for domain, files in sorted(domain_files.items()):
            print(f"  • {domain}: {len(files)} file(s)")
        print(f"{'='*70}")
        
        # Process each domain
        for domain_idx, (domain, files) in enumerate(sorted(domain_files.items()), 1):
            print(f"\n{'─'*70}")
            print(f"DOMAIN [{domain_idx}/{len(domain_files)}]: {domain}")
            print(f"{'─'*70}")
            print(f"Files in domain: {len(files)}")
            
            # Process each file in domain
            for file_idx, input_file in enumerate(sorted(files), 1):
                print(f"\n   [{file_idx}/{len(files)}]")
                
                # Calculate relative path
                rel_path = input_file.relative_to(source_dir)
                
                # Create output path preserving directory structure
                output_file = output_base / rel_path
                
                # Translate
                self.process_file(input_file, output_file, target_language, domain)
        
        self.stats["end_time"] = datetime.now()
        self.print_summary()
    
    def print_summary(self):
        """Print translation statistics"""
        
        duration = (self.stats["end_time"] - self.stats["start_time"]).total_seconds()
        
        print(f"\n{'='*70}")
        print(f"TRANSLATION SUMMARY")
        print(f"{'='*70}")
        print(f"Domains processed: {len(self.stats['domains_processed'])}")
        print(f"  • {', '.join(sorted(self.stats['domains_processed']))}")
        print(f"\nFiles processed: {self.stats['files_processed']:,}")
        print(f"Files skipped: {self.stats['files_skipped']:,}")
        print(f"Sentences translated: {self.stats['sentences_translated']:,}")
        print(f"Errors: {self.stats['errors']:,}")
        print(f"\nDuration: {duration/60:.1f} minutes")
        print(f"Speed: {self.stats['sentences_translated']/duration:.1f} sentences/second")
        print(f"{'='*70}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("="*70)
    print("GPT-5 NANO DOMAIN CORPUS TRANSLATOR")
    print("Hindi → Bhojpuri & Magahi")
    print("Domain-organized with automatic temp file exclusion")
    print("="*70)
    
    # Check API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("\n❌ Error: OPENAI_API_KEY not set")
        print("\nSet it with:")
        print("  Linux/Mac: export OPENAI_API_KEY='your-key-here'")
        print("  Windows:   $env:OPENAI_API_KEY='your-key-here'")
        print("\nGet your API key from: https://platform.openai.com/api-keys")
        sys.exit(1)
    
    print("✅ API key found")
    
    # Configuration
    print("\n" + "="*70)
    print("CONFIGURATION")
    print("="*70)
    
    # Get source directory
    source_dir = input("\nEnter path to Hindi corpus directory: ").strip()
    if not source_dir:
        source_dir = "hindi_corpus"  # Default
    
    source_path = Path(source_dir)
    
    if not source_path.exists():
        print(f"\n❌ Error: Directory not found: {source_path}")
        print("\nExpected structure (domain-organized):")
        print("  hindi_corpus/")
        print("  ├── news/")
        print("  │   ├── article1.txt")
        print("  │   ├── article2.txt")
        print("  │   └── article3.txt")
        print("  ├── literature/")
        print("  │   ├── story1.txt")
        print("  │   └── story2.txt")
        print("  └── social/")
        print("      ├── post1.txt")
        print("      └── post2.txt")
        sys.exit(1)
    
    # Get output base directory
    output_base = input("Enter output base directory [translations]: ").strip()
    if not output_base:
        output_base = "translations"
    
    output_path = Path(output_base)
    
    # Get file pattern
    file_pattern = input("Enter file pattern [*.txt]: ").strip()
    if not file_pattern:
        file_pattern = "*.txt"
    
    # Languages to translate
    languages = input("\nTranslate to (comma-separated) [Bhojpuri,Magahi]: ").strip()
    if not languages:
        languages = "Bhojpuri,Magahi"
    
    target_languages = [lang.strip() for lang in languages.split(',')]
    
    # Validate languages
    valid_languages = ["Bhojpuri", "Magahi"]
    for lang in target_languages:
        if lang not in valid_languages:
            print(f"\n❌ Error: Unknown language '{lang}'")
            print(f"Valid languages: {', '.join(valid_languages)}")
            sys.exit(1)
    
    print(f"\n✅ Configuration complete")
    print(f"   Source: {source_path.absolute()}")
    print(f"   Output: {output_path.absolute()}")
    print(f"   Pattern: {file_pattern}")
    print(f"   Languages: {', '.join(target_languages)}")
    print(f"   Excluding: {', '.join(sorted(EXCLUDED_EXTENSIONS))}")
    
    # Cost estimate
    print(f"\n💰 COST ESTIMATE")
    print(f"{'='*70}")
    
    # Count files and sentences
    print("Scanning corpus...")
    all_files = list(source_path.rglob(file_pattern))
    valid_files = [f for f in all_files if f.suffix.lower() not in EXCLUDED_EXTENSIONS and not f.name.startswith('.') and not f.name.endswith('~')]
    
    if len(valid_files) == 0:
        print(f"❌ No valid files found matching {file_pattern}")
        sys.exit(1)
    
    print(f"Valid files: {len(valid_files)}")
    print(f"Excluded: {len(all_files) - len(valid_files)}")
    
    # Count sentences
    print("Counting sentences...")
    total_sentences = 0
    for file_path in valid_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                total_sentences += sum(1 for line in f if line.strip())
        except:
            pass
    
    print(f"✅ Total sentences: {total_sentences:,}")
    
    # Calculate cost per language
    chars_per_sent = 60  # Average
    total_chars = total_sentences * chars_per_sent
    total_tokens = int(total_chars / 4)
    
    input_tokens = total_tokens
    output_tokens = total_tokens
    
    cost_per_lang_usd = (input_tokens / 1_000_000 * 0.05) + (output_tokens / 1_000_000 * 0.40)
    cost_per_lang_inr = cost_per_lang_usd * 85
    
    total_cost_usd = cost_per_lang_usd * len(target_languages)
    total_cost_inr = cost_per_lang_inr * len(target_languages)
    
    print(f"\nPer language ({total_sentences:,} sentences):")
    print(f"  Cost: ₹{cost_per_lang_inr:.0f} (~${cost_per_lang_usd:.2f})")
    print(f"\nTotal for {len(target_languages)} language(s):")
    print(f"  Cost: ₹{total_cost_inr:.0f} (~${total_cost_usd:.2f})")
    print(f"  Model: GPT-5 Nano (Ultra-Fast)")
    print(f"{'='*70}")
    
    # Confirm
    response = input("\nProceed with translation? (y/n): ")
    if response.lower() != 'y':
        print("Aborted.")
        sys.exit(0)
    
    # Initialize translator
    translator = DomainTranslator(api_key)
    
    # Translate for each language
    for target_lang in target_languages:
        output_lang_dir = output_path / target_lang.lower()
        translator.translate_domain_corpus(
            source_dir=source_path,
            output_base=output_lang_dir,
            target_language=target_lang,
            file_pattern=file_pattern
        )
    
    # Final summary
    print("\n" + "="*70)
    print("✅ ALL TRANSLATIONS COMPLETE")
    print("="*70)
    print(f"\nOutput structure:")
    for target_lang in target_languages:
        lang_dir = output_path / target_lang.lower()
        print(f"\n  {target_lang}:")
        print(f"  📁 {lang_dir}/")
        
        # Show domains
        domains = set()
        for f in lang_dir.rglob(file_pattern):
            domains.add(f.parent.name)
        
        for domain in sorted(domains):
            domain_files = list((lang_dir / domain).glob(file_pattern))
            print(f"     ├── {domain}/ ({len(domain_files)} files)")
            # Show first 2 files as example
            for f in sorted(domain_files)[:2]:
                print(f"     │   ├── {f.name}")
            if len(domain_files) > 2:
                print(f"     │   └── ... ({len(domain_files) - 2} more)")
    
    print("\n" + "="*70)
    print(f"Total cost: ~₹{total_cost_inr:.0f} (~${total_cost_usd:.2f})")
    print(f"Directory structure preserved ✅")
    print(f"Temp files excluded ✅")
    print("="*70)

if __name__ == "__main__":
    main()
