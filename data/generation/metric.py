from __future__ import annotations

import re
import string
from collections import Counter


def normalize_text(text: str) -> str:
    regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
    text = text.lower()
    text = "".join(char for char in text if char not in set(string.punctuation))
    text = re.sub(regex, " ", text)
    return " ".join(text.split())


def exact_match_score(prediction: str, reference: str) -> float:
    return float(normalize_text(prediction) == normalize_text(reference))


def token_f1_score(prediction: str, reference: str) -> float:
    prediction_tokens = normalize_text(prediction).split()
    reference_tokens = normalize_text(reference).split()

    if not prediction_tokens or not reference_tokens:
        return float(prediction_tokens == reference_tokens)

    common = Counter(prediction_tokens) & Counter(reference_tokens)
    num_common = sum(common.values())
    if num_common == 0:
        return 0.0

    precision = num_common / len(prediction_tokens)
    recall = num_common / len(reference_tokens)
    return (2 * precision * recall) / (precision + recall)
