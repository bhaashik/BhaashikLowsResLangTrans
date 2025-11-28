# Full Translation Guide - OpenAI GPT-4o-mini

## Quick Start - Translate All Files

### Single Language

```bash
# Set your API key
export OPENAI_API_KEY="sk-..."

# Translate to Bhojpuri (all 1,432 files)
python scripts/translate_hindi_to_lowres_openai.py --target-lang bho
```

### All 3 Configured Languages

```bash
# Sequential translation (one after another)
for lang in bho mag mai; do
    echo "Starting translation to $lang..."
    python scripts/translate_hindi_to_lowres_openai.py --target-lang $lang
    echo "Completed $lang!"
    echo "---"
done
```

### Parallel Translation (3 terminals)

Open 3 separate terminal windows and run:

**Terminal 1 (Bhojpuri):**
```bash
export OPENAI_API_KEY="sk-..."
python scripts/translate_hindi_to_lowres_openai.py --target-lang bho
```

**Terminal 2 (Magahi):**
```bash
export OPENAI_API_KEY="sk-..."
python scripts/translate_hindi_to_lowres_openai.py --target-lang mag
```

**Terminal 3 (Maithili):**
```bash
export OPENAI_API_KEY="sk-..."
python scripts/translate_hindi_to_lowres_openai.py --target-lang mai
```

**Note:** Parallel runs will hit rate limits faster. Monitor your OpenAI usage dashboard.

## Cost and Time Estimates

### Current Setup (3 Languages)

| Metric | Per Language | All 3 Languages |
|--------|-------------|-----------------|
| **Files** | 1,432 | 4,296 |
| **Sentences** | 66,317 | 198,951 |
| **Cost (GPT-4o-mini)** | ~$4.00 | ~$12.00 |
| **Time (sequential)** | ~8-10 hours | ~24-30 hours |
| **Time (parallel)** | ~8-10 hours | ~8-10 hours* |

*Parallel time assumes no rate limiting. Actual time may be longer if you hit API limits.

### Extended Setup (15 Languages)

If you extend to **15 low-resource Indic languages**:

**Languages to add (examples):**
1. Bhojpuri (bho) ✓ configured
2. Magahi (mag) ✓ configured
3. Maithili (mai) ✓ configured
4. Awadhi (awa)
5. Braj (bra)
6. Bundeli (bns)
7. Marwari (mwr)
8. Chhattisgarhi (hne)
9. Rajasthani (raj)
10. Haryanvi (bgc)
11. Kumaoni (kfy)
12. Garhwali (gbm)
13. Kangri (xnr)
14. Bagri (bgq)
15. Malvi (mup)

#### Cost Estimates (15 Languages)

| Metric | Value |
|--------|-------|
| **Total Files** | 21,480 (1,432 × 15) |
| **Total Sentences** | 994,755 (66,317 × 15) |
| **Cost per language** | ~$4.00 |
| **Total Cost** | **~$60.00 USD** |

**Breakdown:**
- Input tokens (with caching): ~$1.50 × 15 = $22.50
- Output tokens: ~$2.50 × 15 = $37.50
- **Total: ~$60.00 USD**

#### Time Estimates (15 Languages)

**Sequential (one after another):**
- Per language: ~8-10 hours
- All 15 languages: **~120-150 hours (5-6 days)**

**Parallel (multiple simultaneous):**
- 3 at a time: ~40-50 hours (~2 days)
- 5 at a time: ~24-30 hours (~1-1.5 days)*
- 15 at a time: ~8-10 hours (but will hit rate limits)*

*Highly dependent on your OpenAI account tier's rate limits

## Rate Limits

### Typical Free Tier Limits
- **Requests per minute (RPM):** 3
- **Tokens per minute (TPM):** 40,000

This would be **too slow** for large-scale translation. Consider upgrading.

### Tier 1 Limits (after $5 spent)
- **RPM:** 500
- **TPM:** 200,000

This is sufficient for 1-3 parallel translations.

### Tier 2+ Limits
Higher limits allow more parallel processing. Check: https://platform.openai.com/account/limits

## Monitoring Progress

### Real-time Progress
The script shows:
```
[15/1432] AGRICULTURE/file_015.txt
  Sentences: 51
  Translating batch 1-10/51... Done (cost: $0.0004)
  Translating batch 11-20/51... Done (cost: $0.0004)
  ...
  ✓ Written: output/Bhojpuri/.../file_015.txt
```

### Check Files Completed
```bash
# Count translated files
find output/Bhojpuri/plain-text/by_file -name "*.txt" | wc -l

# Compare with input
find input/converted/Hindi/plain-text/by_file -name "*.txt" | wc -l
```

### Monitor Costs
Check OpenAI usage dashboard: https://platform.openai.com/usage

## Resuming Interrupted Translation

If translation is interrupted, simply re-run the command:
```bash
python scripts/translate_hindi_to_lowres_openai.py --target-lang bho
```

The script will:
- ✅ Skip already translated files
- ✅ Continue from where it stopped
- ✅ Add only incremental cost

## Adding New Languages

To add one of the 12 additional languages:

### 1. Create Prompt Configuration

Copy and modify an existing prompt config:
```bash
cd universal_translate/config/prompts/

# Copy Bhojpuri config as template
cp hi_to_bho_openai.yaml hi_to_awa_openai.yaml

# Edit for Awadhi
nano hi_to_awa_openai.yaml
```

Update:
- `name`: "hindi_to_awadhi_openai"
- `description`: Change language name
- `system_prompt`: Update linguistic features for Awadhi
- `examples_file`: Point to Awadhi examples (if available)

### 2. Prepare Monolingual Examples (Optional)

If you have Awadhi corpus data:
```bash
# Create examples file
python scripts/prepare_monolingual_examples.py \
  --input-file input/PV-BMM-Public-Data/awadhi-all.txt \
  --language awa \
  --language-name Awadhi \
  --output universal_translate/data/examples/hi_to_awa_monolingual.json
```

### 3. Update Script

Add language to choices in `scripts/translate_hindi_to_lowres_openai.py`:
```python
parser.add_argument(
    '--target-lang',
    required=True,
    choices=['bho', 'mag', 'mai', 'awa', 'bra', ...],  # Add new language codes
    help="Target language"
)
```

And add to language names dict:
```python
lang_names = {
    'bho': 'Bhojpuri',
    'mag': 'Magahi',
    'mai': 'Maithili',
    'awa': 'Awadhi',
    'bra': 'Braj',
    # ... etc
}
```

### 4. Run Translation
```bash
python scripts/translate_hindi_to_lowres_openai.py --target-lang awa
```

## Best Practices for 15 Languages

### 1. Start Small
Test each language with 5 files first:
```bash
for lang in bho mag mai awa bra bns mwr hne raj bgc kfy gbm xnr bgq mup; do
    echo "Testing $lang..."
    python scripts/translate_hindi_to_lowres_openai.py \
      --target-lang $lang \
      --max-files 5
done
```

### 2. Check Quality
Manually review sample translations from each language to ensure quality.

### 3. Batch by Priority
Group languages by importance:
```bash
# High priority (3 languages)
for lang in bho mag mai; do
    python scripts/translate_hindi_to_lowres_openai.py --target-lang $lang
done

# Medium priority (6 languages)
for lang in awa bra bns mwr hne raj; do
    python scripts/translate_hindi_to_lowres_openai.py --target-lang $lang
done

# Lower priority (6 languages)
for lang in bgc kfy gbm xnr bgq mup; do
    python scripts/translate_hindi_to_lowres_openai.py --target-lang $lang
done
```

### 4. Use Screen/Tmux for Long Runs
For 5-6 day runs, use `screen` or `tmux` to prevent interruption:
```bash
# Start screen session
screen -S translation

# Run translation
for lang in bho mag mai ...; do
    python scripts/translate_hindi_to_lowres_openai.py --target-lang $lang
done

# Detach: Ctrl+A, then D
# Reattach: screen -r translation
```

### 5. Log Everything
```bash
for lang in bho mag mai ...; do
    echo "Starting $lang at $(date)" | tee -a translation.log
    python scripts/translate_hindi_to_lowres_openai.py --target-lang $lang 2>&1 | tee -a translation_${lang}.log
    echo "Completed $lang at $(date)" | tee -a translation.log
done
```

## Cost Optimization Tips

### 1. Use Larger Batch Sizes
```bash
python scripts/translate_hindi_to_lowres_openai.py \
  --target-lang bho \
  --batch-size 20  # Default is 10
```

Larger batches:
- ✅ Fewer API calls
- ✅ Better prompt caching utilization
- ✅ Slightly lower cost
- ⚠️ Might hit token limits with very long sentences

### 2. Maximize Prompt Caching
The system already uses 50 monolingual examples for caching. After the first batch, subsequent batches get ~50% discount on system prompt tokens.

### 3. Consider GPT-3.5-turbo for Draft Translation
If quality isn't critical initially:
```bash
python scripts/translate_hindi_to_lowres_openai.py \
  --target-lang bho \
  --model gpt-3.5-turbo
```

Cost: ~$2.50/language instead of $4.00 (but lower quality)

## Final Recommendations

### For 3 Languages (Current Setup)
- **Cost:** ~$12.00 USD ✓ Very affordable
- **Time:** 1-2 days (sequential) or ~8-10 hours (parallel)
- **Recommendation:** ✅ Run parallel if you have Tier 1+ account

### For 15 Languages (Extended)
- **Cost:** ~$60.00 USD ✓ Still very affordable
- **Time:** 5-6 days (sequential) or 1-2 days (parallel)
- **Recommendations:**
  1. ✅ Start with 3 languages to validate quality
  2. ✅ Use `screen`/`tmux` for long runs
  3. ✅ Run 3-5 languages in parallel (based on rate limits)
  4. ✅ Monitor costs on OpenAI dashboard
  5. ✅ Test each new language with 5 files first

## Verification After Translation

```bash
# Check all files were translated
for lang in Bhojpuri Magahi Maithili; do
    input_count=$(find input/converted/Hindi/plain-text/by_file -name "*.txt" | wc -l)
    output_count=$(find output/$lang/plain-text/by_file -name "*.txt" | wc -l)
    echo "$lang: $output_count / $input_count files"
done

# Spot-check translations
head -5 input/converted/Hindi/plain-text/by_file/AGRICULTURE/mor-hin_agriculture_set6-pos-chunk-1-50-posn-name.txt
echo "---"
head -5 output/Bhojpuri/plain-text/by_file/AGRICULTURE/mor-hin_agriculture_set6-pos-chunk-1-50-posn-name.txt
```

## Summary

| Scenario | Files | Sentences | Cost | Time (Sequential) | Time (Parallel) |
|----------|-------|-----------|------|-------------------|-----------------|
| **3 Languages** | 4,296 | 198,951 | ~$12 | 24-30 hrs | 8-10 hrs |
| **15 Languages** | 21,480 | 994,755 | ~$60 | 120-150 hrs | 24-40 hrs |

**Ready to start?**
```bash
export OPENAI_API_KEY="sk-..."
python scripts/translate_hindi_to_lowres_openai.py --target-lang bho
```
