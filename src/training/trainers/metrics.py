"""Metrics computation for translation evaluation."""

from typing import List, Dict, Any, Optional
import numpy as np


def compute_metrics(eval_preds, tokenizer, metric_type: str = "bleu") -> Dict[str, float]:
    """
    Compute evaluation metrics for translation.

    Args:
        eval_preds: EvalPrediction object with predictions and label_ids
        tokenizer: Tokenizer for decoding
        metric_type: Type of metric ("bleu", "chrf", "both")

    Returns:
        Dictionary of metrics
    """
    preds, labels = eval_preds

    # Decode predictions
    if isinstance(preds, tuple):
        preds = preds[0]

    # Replace -100 in labels (used for padding)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    # Decode
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Strip whitespace
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [label.strip() for label in decoded_labels]

    # Compute metrics
    metrics = {}

    if metric_type in ["bleu", "both"]:
        bleu_score = compute_bleu(decoded_preds, decoded_labels)
        metrics["bleu"] = bleu_score

    if metric_type in ["chrf", "both"]:
        chrf_score = compute_chrf(decoded_preds, decoded_labels)
        metrics["chrf"] = chrf_score

    # Add length statistics
    pred_lens = [len(pred.split()) for pred in decoded_preds]
    label_lens = [len(label.split()) for label in decoded_labels]

    metrics["gen_len"] = np.mean(pred_lens)
    metrics["ref_len"] = np.mean(label_lens)

    return metrics


def compute_bleu(predictions: List[str], references: List[str]) -> float:
    """
    Compute corpus BLEU score.

    Args:
        predictions: List of predicted translations
        references: List of reference translations

    Returns:
        BLEU score (0-100)
    """
    try:
        import sacrebleu
    except ImportError:
        print("Warning: sacrebleu not installed. Install: pip install sacrebleu")
        return 0.0

    # Format references for sacrebleu (list of lists)
    references_formatted = [[ref] for ref in references]

    # Compute BLEU
    bleu = sacrebleu.corpus_bleu(predictions, references_formatted)

    return bleu.score


def compute_chrf(predictions: List[str], references: List[str]) -> float:
    """
    Compute corpus chrF score.

    Args:
        predictions: List of predicted translations
        references: List of reference translations

    Returns:
        chrF score (0-100)
    """
    try:
        import sacrebleu
    except ImportError:
        print("Warning: sacrebleu not installed. Install: pip install sacrebleu")
        return 0.0

    # Format references for sacrebleu
    references_formatted = [[ref] for ref in references]

    # Compute chrF
    chrf = sacrebleu.corpus_chrf(predictions, references_formatted)

    return chrf.score


def compute_ter(predictions: List[str], references: List[str]) -> float:
    """
    Compute Translation Error Rate (TER).

    Args:
        predictions: List of predicted translations
        references: List of reference translations

    Returns:
        TER score (lower is better)
    """
    try:
        import sacrebleu
    except ImportError:
        print("Warning: sacrebleu not installed. Install: pip install sacrebleu")
        return 100.0

    # Format references for sacrebleu
    references_formatted = [[ref] for ref in references]

    # Compute TER
    ter = sacrebleu.corpus_ter(predictions, references_formatted)

    return ter.score


def compute_all_metrics(
    predictions: List[str],
    references: List[str]
) -> Dict[str, float]:
    """
    Compute all available metrics.

    Args:
        predictions: List of predicted translations
        references: List of reference translations

    Returns:
        Dictionary with all metrics
    """
    return {
        "bleu": compute_bleu(predictions, references),
        "chrf": compute_chrf(predictions, references),
        "ter": compute_ter(predictions, references),
    }


def print_metrics(metrics: Dict[str, float], title: str = "Evaluation Metrics"):
    """
    Print metrics in a formatted table.

    Args:
        metrics: Dictionary of metrics
        title: Title for the table
    """
    print("\n" + "=" * 50)
    print(f"{title:^50}")
    print("=" * 50)

    for metric_name, score in metrics.items():
        print(f"{metric_name.upper():15} : {score:6.2f}")

    print("=" * 50 + "\n")
