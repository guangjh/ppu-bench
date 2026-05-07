import re
import string
from collections import Counter
from math import exp, log

try:
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
except Exception:  # pragma: no cover
    SmoothingFunction = None
    sentence_bleu = None


def _safe_text(value):
    if value is None:
        return ""
    return str(value).strip()


def normalize_text(text):
    text = _safe_text(text).lower()
    text = "".join(char for char in text if char not in set(string.punctuation))
    return " ".join(text.split())


def parse_choice_label(prediction):
    text = _safe_text(prediction)
    if not text:
        return None

    matches = re.findall(r"\b([ABCD])\b", text.upper())
    if not matches:
        stripped = text.strip().upper()
        if stripped in {"A", "B", "C", "D"}:
            return stripped
        return None

    unique_matches = []
    for match in matches:
        if match not in unique_matches:
            unique_matches.append(match)

    if len(unique_matches) != 1:
        return None
    return unique_matches[0]


def classification_accuracy(rows):
    total = len(rows)
    if total == 0:
        return 0.0

    correct = 0
    for row in rows:
        predicted_label = parse_choice_label(row.get("prediction"))
        if predicted_label and predicted_label == _safe_text(row.get("answer_label")).upper():
            correct += 1
    return correct / total


def exact_match(prediction, answer):
    return float(_safe_text(prediction) == _safe_text(answer))


def normalized_exact_match(prediction, answer):
    return float(normalize_text(_safe_text(prediction)) == normalize_text(_safe_text(answer)))


def aggregate_cloze(rows):
    total = len(rows)
    if total == 0:
        return _empty_cloze_metrics()

    totals = _empty_cloze_metrics()
    for row in rows:
        prediction = row.get("prediction")
        answer = row.get("answer_text")
        totals["exact_match"] += exact_match(prediction, answer)
        totals["normalized_exact_match"] += normalized_exact_match(prediction, answer)
        _add_rouge_totals(totals, answer, prediction)

    for key in totals:
        totals[key] /= total
    return totals


def _empty_cloze_metrics():
    return {
        "exact_match": 0.0,
        "normalized_exact_match": 0.0,
        **_empty_rouge_metrics(),
    }


def _ngram_counts(tokens, n):
    if len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1))


def _f1_from_precision_recall(precision, recall):
    if precision == 0.0 or recall == 0.0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def rouge_n_scores(reference, prediction, n):
    reference_tokens = normalize_text(_safe_text(reference)).split()
    prediction_tokens = normalize_text(_safe_text(prediction)).split()
    if not reference_tokens and not prediction_tokens:
        return 1.0, 1.0, 1.0
    if not reference_tokens or not prediction_tokens:
        return 0.0, 0.0, 0.0

    reference_counts = _ngram_counts(reference_tokens, n)
    prediction_counts = _ngram_counts(prediction_tokens, n)
    if not reference_counts or not prediction_counts:
        return 0.0, 0.0, 0.0

    overlap = reference_counts & prediction_counts
    overlap_total = sum(overlap.values())
    precision = overlap_total / sum(prediction_counts.values())
    recall = overlap_total / sum(reference_counts.values())
    return precision, recall, _f1_from_precision_recall(precision, recall)


def rouge_n_f1(reference, prediction, n):
    return rouge_n_scores(reference, prediction, n)[2]


def _lcs_length(reference_tokens, prediction_tokens):
    if not reference_tokens or not prediction_tokens:
        return 0

    previous = [0] * (len(prediction_tokens) + 1)
    for reference_token in reference_tokens:
        current = [0]
        for index, prediction_token in enumerate(prediction_tokens, start=1):
            if reference_token == prediction_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def rouge_l_scores(reference, prediction):
    reference_tokens = normalize_text(_safe_text(reference)).split()
    prediction_tokens = normalize_text(_safe_text(prediction)).split()
    if not reference_tokens and not prediction_tokens:
        return 1.0, 1.0, 1.0
    if not reference_tokens or not prediction_tokens:
        return 0.0, 0.0, 0.0

    lcs = _lcs_length(reference_tokens, prediction_tokens)
    precision = lcs / len(prediction_tokens)
    recall = lcs / len(reference_tokens)
    return precision, recall, _f1_from_precision_recall(precision, recall)


def rouge_l_f1(reference, prediction):
    return rouge_l_scores(reference, prediction)[2]


def _fallback_bleu(reference_tokens, prediction_tokens):
    max_order = min(4, len(reference_tokens), len(prediction_tokens))
    if max_order == 0:
        return 0.0

    precisions = []
    for n in range(1, max_order + 1):
        prediction_counts = _ngram_counts(prediction_tokens, n)
        if not prediction_counts:
            return 0.0
        reference_counts = _ngram_counts(reference_tokens, n)
        overlap_total = sum((prediction_counts & reference_counts).values())
        precisions.append((overlap_total + 1.0) / (sum(prediction_counts.values()) + 1.0))

    geometric_mean = exp(sum(log(value) for value in precisions) / max_order)
    brevity_penalty = 1.0
    if len(prediction_tokens) < len(reference_tokens):
        brevity_penalty = exp(1 - len(reference_tokens) / len(prediction_tokens))
    return brevity_penalty * geometric_mean


def bleu_score(reference, prediction):
    reference_text = _safe_text(reference)
    prediction_text = _safe_text(prediction)
    if not reference_text and not prediction_text:
        return 1.0
    if not reference_text or not prediction_text:
        return 0.0

    reference_tokens = normalize_text(reference_text).split()
    prediction_tokens = normalize_text(prediction_text).split()
    if not reference_tokens or not prediction_tokens:
        return 0.0
    if sentence_bleu is None or SmoothingFunction is None:
        return _fallback_bleu(reference_tokens, prediction_tokens)
    return sentence_bleu([reference_tokens], prediction_tokens, smoothing_function=SmoothingFunction().method1)


def _empty_rouge_metrics():
    metrics = {}
    metrics["rougeL_recall"] = 0.0
    metrics["rougeL_f1"] = 0.0
    # for rouge_name in ("rougeL"): # "rouge1", "rouge2", 
    #     # metrics[f"{rouge_name}_precision"] = 0.0
    #     metrics[f"{rouge_name}_recall"] = 0.0
    #     metrics[f"{rouge_name}_f1"] = 0.0
    return metrics


def _add_rouge_totals(totals, answer, prediction):
    rouge1_precision, rouge1_recall, rouge1_f1 = rouge_n_scores(answer, prediction, 1)
    rouge2_precision, rouge2_recall, rouge2_f1 = rouge_n_scores(answer, prediction, 2)
    rouge_l_precision, rouge_l_recall, rouge_l_f1_score = rouge_l_scores(answer, prediction)
    # totals["rouge1_precision"] += rouge1_precision
    # totals["rouge1_recall"] += rouge1_recall
    # totals["rouge1_f1"] += rouge1_f1
    # totals["rouge2_precision"] += rouge2_precision
    # totals["rouge2_recall"] += rouge2_recall
    # totals["rouge2_f1"] += rouge2_f1
    # totals["rougeL_precision"] += rouge_l_precision
    totals["rougeL_recall"] += rouge_l_recall
    totals["rougeL_f1"] += rouge_l_f1_score


def _empty_generation_metrics():
    metrics = _empty_rouge_metrics()
    # metrics["bleu"] = 0.0
    return metrics


def aggregate_generation(rows):
    total = len(rows)
    if total == 0:
        return _empty_generation_metrics()

    totals = _empty_generation_metrics()
    # bleu_total = 0.0
    for row in rows:
        prediction = row.get("prediction")
        answer = row.get("answer_text")
        _add_rouge_totals(totals, answer, prediction)
        # bleu_total += bleu_score(answer, prediction)

    # totals["bleu"] = bleu_total / total
    for key in totals:
        if key != "bleu":
            totals[key] /= total
    return totals


def summarize_task_metrics(task_type, rows):
    if task_type == "classification":
        return {"accuracy": classification_accuracy(rows)}
    if task_type == "cloze":
        return aggregate_cloze(rows)
    if task_type == "generation":
        return aggregate_generation(rows)
    raise ValueError(f"Unsupported task type: {task_type}")
