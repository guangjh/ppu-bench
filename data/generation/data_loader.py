from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

from PIL import Image


def load_records_from_parquet(data_path: Path) -> list[dict[str, Any]]:
    try:
        import pandas as pd
    except ImportError:
        pd = None

    if pd is not None:
        dataframe = pd.read_parquet(data_path)
        return dataframe.to_dict(orient="records")

    try:
        from datasets import load_dataset
    except ImportError:
        load_dataset = None

    if load_dataset is not None:
        dataset = load_dataset("parquet", data_files=str(data_path), split="train")
        return [dict(row) for row in dataset]

    raise RuntimeError(
        "Reading parquet requires `pandas` with a parquet backend or `datasets`. "
        "Install one of them first, for example: `pip install pandas pyarrow`."
    )


def load_records(data_path: Path) -> list[dict[str, Any]]:
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    suffix = data_path.suffix.lower()
    if suffix == ".parquet":
        return load_records_from_parquet(data_path)
    if suffix == ".json":
        data = json.loads(data_path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"Expected JSON array dataset: {data_path}")
        return data
    if suffix == ".jsonl":
        records = []
        with data_path.open("r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    raise ValueError(f"Unsupported dataset format: {data_path.suffix}")


def load_hf_unlearning_target() -> list[dict[str, Any]]:
    from datasets import load_dataset as hf_load_dataset

    ds = hf_load_dataset("closerG/ppu-bench", "unlearning_target")["train"]
    return [dict(row) for row in ds]


def resolve_image_path(raw_path: str | None, dataset_path: Path) -> Path | None:
    if not raw_path:
        return None

    raw_image_path = Path(raw_path)
    candidates = []

    if raw_image_path.is_absolute():
        candidates.append(raw_image_path)
    else:
        repo_root = Path(__file__).resolve().parents[1]
        candidates.append(dataset_path.parent / raw_image_path)
        candidates.append(repo_root / "data" / raw_image_path)
        candidates.append(Path.cwd() / raw_image_path)
        candidates.append(repo_root / raw_image_path)

    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.exists():
            return candidate

    return candidates[0].resolve() if candidates else None


def flatten_qa_records(
    records: list[dict[str, Any]],
    dataset_path: Path,
    *,
    subject_limit: int | None,
    qa_limit: int | None,
) -> list[dict[str, Any]]:
    flattened: list[dict[str, Any]] = []

    for subject_index, record in enumerate(records):
        if subject_limit is not None and subject_index >= subject_limit:
            break

        record_id = str(record.get("subject_id") or record.get("id") or subject_index)
        subject = str(record.get("subject", "")).strip()
        image_path = resolve_image_path(record.get("image_path"), dataset_path)
        image_blob = record.get("image")
        qas = record.get("cloze_probes") or record.get("qas")
        if isinstance(qas, str):
            qas = json.loads(qas)
        if hasattr(qas, "tolist"):
            qas = qas.tolist()
        elif isinstance(qas, tuple):
            qas = list(qas)

        if not isinstance(qas, list):
            raise ValueError(
                f"Expected `qas` to be a list for record `{record_id}`, got {type(qas).__name__}."
            )

        for qa_index, qa in enumerate(qas):
            if qa_limit is not None and len(flattened) >= qa_limit:
                return flattened

            if not isinstance(qa, dict):
                raise ValueError(
                    f"Expected QA item to be a dict for record `{record_id}`, got {type(qa).__name__}."
                )

            question = str(qa.get("question") or qa.get("query") or "").strip()
            answer = str(qa.get("answer", "")).strip()
            qa_source = str(qa.get("from", "")).strip()
            qa_id = str(qa.get("sample_id") or qa.get("id") or qa.get("qa_id") or "").strip()
            sample_id = qa_id or f"{record_id}_{qa_index:02d}"

            if not question or not answer:
                continue

            flattened.append(
                {
                    "id": sample_id,
                    "subject_id": record_id,
                    "record_id": record_id,
                    "subject": subject,
                    "image_path": image_path.as_posix() if image_path else None,
                    "image_blob": image_blob,
                    "question": question,
                    "ground_truth": answer,
                    "qa_source": qa_source,
                }
            )

    return flattened


def load_evaluation_samples(
    data_path: Path,
    *,
    hf_config: str = "",
    subject_limit: int | None,
    qa_limit: int | None,
) -> list[dict[str, Any]]:
    if hf_config:
        records = load_hf_unlearning_target()
        return flatten_qa_records(
            records,
            Path("."),
            subject_limit=subject_limit,
            qa_limit=qa_limit,
        )

    records = load_records(data_path)
    return flatten_qa_records(
        records,
        data_path,
        subject_limit=subject_limit,
        qa_limit=qa_limit,
    )


def load_pil_image(
    sample: dict[str, Any],
    cache: dict[str, Image.Image],
    *,
    require_image: bool = True,
) -> tuple[Image.Image | None, str | None]:
    cache_key = sample["record_id"]
    if cache_key in cache:
        cached = cache[cache_key]
        return cached.copy(), sample.get("image_path")

    image_blob = sample.get("image_blob")
    if isinstance(image_blob, Image.Image):
        cache[cache_key] = image_blob
        return image_blob.copy(), sample.get("image_path")
    if isinstance(image_blob, dict):
        image_bytes = image_blob.get("bytes")
        blob_path = image_blob.get("path")
        if image_bytes:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            cache[cache_key] = image
            return image.copy(), sample.get("image_path") or blob_path
        if blob_path:
            image = Image.open(blob_path).convert("RGB")
            cache[cache_key] = image
            return image.copy(), blob_path

    image_path = sample.get("image_path")
    if image_path:
        image = Image.open(image_path).convert("RGB")
        cache[cache_key] = image
        return image.copy(), image_path

    if not require_image:
        return None, None

    sample_identifier = sample.get("id") or sample.get("sample_id") or "<unknown>"
    raise ValueError(f"No usable image found for sample `{sample_identifier}`.")
