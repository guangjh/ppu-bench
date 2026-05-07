import json
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_COMPLETE_SPLIT_FILE = REPO_ROOT / "eval" / "split" / "CompleteUnlearning" / "forget_subject_ids.json"
DEFAULT_SELECTIVE_SPLIT_FILE = REPO_ROOT / "eval" / "split" / "SelectiveUnlearning" / "sensitive_info_subject_id_id.json"
DEFAULT_PERSONA_SPLIT_FILE = REPO_ROOT / "eval" / "split" / "PersonaUnlearning" / "majority_ids.json"

SPLIT_FILE_NAMES = {
    "complete": "forget_subject_ids.json",
    "selective": "sensitive_info_subject_id_id.json",
    "persona": "majority_ids.json",
}
SPLIT_DIR_NAMES = {
    "complete": "CompleteUnlearning",
    "selective": "SelectiveUnlearning",
    "persona": "PersonaUnlearning",
}
DEFAULT_SPLIT_FILES = {
    "complete": DEFAULT_COMPLETE_SPLIT_FILE,
    "selective": DEFAULT_SELECTIVE_SPLIT_FILE,
    "persona": DEFAULT_PERSONA_SPLIT_FILE,
}


class FlatPruneDataset(Dataset):
    """Flattened QA/VQA records used by the incremental pruning runner."""

    def __init__(self, records: Sequence[Dict[str, Any]], use_images: bool):
        self.records = list(records)
        self.use_images = use_images

    def __len__(self) -> int:
        return len(self.records)

    def _load_image_from_path(self, image_path: str) -> Image.Image:
        with Image.open(image_path) as image:
            return image.convert("RGB")

    def _load_image_from_bytes(self, image_bytes: bytes) -> Image.Image:
        with Image.open(BytesIO(image_bytes)) as image:
            return image.convert("RGB")

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        record = self.records[idx]
        image = None
        if self.use_images:
            image_pil = record.get("image")
            if isinstance(image_pil, Image.Image):
                image = image_pil
            else:
                image_bytes = record.get("image_bytes")
                if image_bytes:
                    image = self._load_image_from_bytes(image_bytes)
                else:
                    image = self._load_image_from_path(record["image_path"])

        return {
            "image": image,
            "question": record["question"],
            "answer": record["generated_answer"],
            "sample_id": record["sample_id"],
            "subject_id": record.get("subject_id", ""),
            "subject": record.get("subject", ""),
            "image_path": record.get("image_path", ""),
        }


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _read_records(path: str) -> List[Dict[str, Any]]:
    dataset_path = Path(path).expanduser().resolve()
    if dataset_path.suffix == ".json":
        data = _read_json(dataset_path)
    elif dataset_path.suffix == ".parquet":
        data = pd.read_parquet(dataset_path).to_dict(orient="records")
    else:
        raise ValueError(f"Unsupported dataset format: {dataset_path}. Expected .json or .parquet")

    if not isinstance(data, list):
        raise ValueError(f"Dataset must resolve to a list of flattened records: {dataset_path}")
    return data


def load_hf_records(config_name: str) -> List[Dict[str, Any]]:
    from datasets import load_dataset as hf_load_dataset

    ds = hf_load_dataset("closerG/ppu-bench", config_name)["train"]
    return [ds[idx] for idx in range(len(ds))]


def _extract_image_bytes(record: Dict[str, Any]) -> Optional[bytes]:
    image = record.get("image")
    if not isinstance(image, dict):
        return None

    image_bytes = image.get("bytes")
    if image_bytes is None:
        return None
    if isinstance(image_bytes, bytes):
        return image_bytes
    return bytes(image_bytes)


def _normalize_multimodal_record(record: Dict[str, Any], index: int) -> Dict[str, Any]:
    image = record.get("image")
    image_is_pil = isinstance(image, Image.Image)

    normalized = {
        "sample_id": record.get("sample_id", record.get("id")),
        "subject_id": record.get("subject_id"),
        "subject": record.get("subject"),
        "image_path": record.get("image_path"),
        "question": record.get("question"),
        "generated_answer": record.get("generated_answer"),
        "image_bytes": _extract_image_bytes(record) if not image_is_pil else None,
        "image": image if image_is_pil else None,
    }

    missing = [
        key
        for key in ("sample_id", "subject_id", "subject", "image_path", "question", "generated_answer")
        if normalized.get(key) in (None, "")
    ]
    if image_is_pil:
        missing = [f for f in missing if f not in ("image_path",)]
    if missing:
        raise ValueError(f"Multimodal record {index} is missing required fields: {missing}")

    for key in ("sample_id", "subject_id", "subject", "question", "generated_answer"):
        normalized[key] = str(normalized[key])
    normalized["image_path"] = str(normalized["image_path"]) if normalized["image_path"] is not None else ""
    return normalized


def _normalize_unimodal_record(
    record: Dict[str, Any],
    index: int,
    sample_to_subject_id: Dict[str, str],
    subject_to_subject_id: Dict[str, str],
) -> Dict[str, Any]:
    sample_id = str(record.get("sample_id", record.get("id", "")))
    subject = str(record.get("subject", ""))
    subject_id = record.get("subject_id")
    if subject_id in (None, ""):
        subject_id = sample_to_subject_id.get(sample_id) or subject_to_subject_id.get(subject)

    normalized = {
        "sample_id": sample_id,
        "subject_id": str(subject_id) if subject_id not in (None, "") else "",
        "subject": subject,
        "image_path": str(record.get("image_path", "")),
        "question": record.get("question"),
        "generated_answer": record.get("generated_answer"),
        "image_bytes": None,
    }

    missing = [
        key
        for key in ("sample_id", "subject", "question", "generated_answer")
        if normalized.get(key) in (None, "")
    ]
    if missing:
        raise ValueError(f"Unimodal record {index} is missing required fields: {missing}")

    if not normalized["subject_id"]:
        raise ValueError(
            f"Unimodal record {index} ({sample_id}) has no subject_id and could not be "
            "matched from the multimodal dataset. Complete unlearning needs subject_id."
        )

    normalized["question"] = str(normalized["question"])
    normalized["generated_answer"] = str(normalized["generated_answer"])
    return normalized


def _resolve_split_path(task: str, data_split_dir: Optional[str]) -> Path:
    if task not in DEFAULT_SPLIT_FILES:
        raise ValueError(f"Unsupported split task: {task}")

    if not data_split_dir:
        return DEFAULT_SPLIT_FILES[task]

    candidate = Path(data_split_dir).expanduser()
    if candidate.suffix == ".json":
        return candidate.resolve()

    direct_candidate = candidate / SPLIT_FILE_NAMES[task]
    if direct_candidate.exists():
        return direct_candidate.resolve()

    nested_candidate = candidate / SPLIT_DIR_NAMES[task] / SPLIT_FILE_NAMES[task]
    if nested_candidate.exists():
        return nested_candidate.resolve()

    return direct_candidate.resolve()


def _split_entries_to_sample_ids(entries: Any, split_path: Path, key: str) -> set:
    if not isinstance(entries, list):
        raise ValueError(f"Split key {key!r} must be a JSON array in {split_path}")

    sample_ids = set()
    for index, entry in enumerate(entries):
        sample_id = entry.get("id", entry.get("sample_id")) if isinstance(entry, dict) else entry
        if sample_id in (None, ""):
            raise ValueError(f"Split entry {key}[{index}] is missing id/sample_id in {split_path}")
        sample_ids.add(str(sample_id))
    return sample_ids


def _split_records(
    records: Iterable[Dict[str, Any]],
    task: str,
    split_path: Path,
    ratio: int,
    retain_ids: Optional[set] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    records = list(records)
    split_data = _read_json(split_path)

    if task == "complete":
        if ratio not in {5, 10, 15, 30}:
            raise ValueError(f"Complete unlearning ratio must be one of 5/10/15/30, got {ratio}")
        forget_key = f"forget{ratio}"
        if forget_key not in split_data:
            raise KeyError(f"Split file {split_path} does not contain key {forget_key}")
        forget_subject_ids = {str(subject_id) for subject_id in split_data[forget_key]}
        forget_records = [record for record in records if record["subject_id"] in forget_subject_ids]
        retain_records = [record for record in records if record["subject_id"] not in forget_subject_ids]
        return forget_records, retain_records

    if task == "selective":
        if not isinstance(split_data, list):
            raise ValueError(f"Selective split file must be a JSON array: {split_path}")
        forget_sample_ids = {str(sample["id"]) for sample in split_data if "id" in sample}
        forget_records = [record for record in records if record["sample_id"] in forget_sample_ids]
        retain_records = [record for record in records if record["sample_id"] not in forget_sample_ids]
        return forget_records, retain_records

    if task == "persona":
        if not isinstance(split_data, dict):
            raise ValueError(f"Persona split file must be a JSON object: {split_path}")
        forget_sample_ids = _split_entries_to_sample_ids(split_data["delete_ids"], split_path, "delete_ids")
        keep_sample_ids = retain_ids or _split_entries_to_sample_ids(split_data["keep_ids"], split_path, "keep_ids")
        forget_records = [record for record in records if record["sample_id"] in forget_sample_ids]
        retain_records = [record for record in records if record["sample_id"] in keep_sample_ids]
        return forget_records, retain_records

    raise ValueError(f"Unsupported task: {task}")


def build_flat_prune_datasets(
    multimodal_train_dataset: str = "",
    unimodal_train_dataset: str = "",
    multimodal_hf_config: str = "",
    unimodal_hf_config: str = "",
    task: str = "complete",
    ratio: int = 5,
    data_split_dir: Optional[str] = None,
) -> Tuple[FlatPruneDataset, FlatPruneDataset, FlatPruneDataset, FlatPruneDataset, Path]:
    if multimodal_hf_config:
        multimodal_raw = load_hf_records(multimodal_hf_config)
        multimodal_records = [
            _normalize_multimodal_record(record, index)
            for index, record in enumerate(multimodal_raw)
        ]
    elif multimodal_train_dataset:
        multimodal_records = [
            _normalize_multimodal_record(record, index)
            for index, record in enumerate(_read_records(multimodal_train_dataset))
        ]
    else:
        raise ValueError("Either --multimodal_hf_config or --multimodal_train_dataset is required")

    sample_to_subject_id = {record["sample_id"]: record["subject_id"] for record in multimodal_records}
    subject_to_subject_id = {record["subject"]: record["subject_id"] for record in multimodal_records}

    if unimodal_hf_config:
        unimodal_raw = load_hf_records(unimodal_hf_config)
        unimodal_records = [
            _normalize_unimodal_record(record, index, sample_to_subject_id, subject_to_subject_id)
            for index, record in enumerate(unimodal_raw)
        ]
    elif unimodal_train_dataset:
        unimodal_records = [
            _normalize_unimodal_record(record, index, sample_to_subject_id, subject_to_subject_id)
            for index, record in enumerate(_read_records(unimodal_train_dataset))
        ]
    else:
        raise ValueError("Either --unimodal_hf_config or --unimodal_train_dataset is required")

    split_path = _resolve_split_path(task, data_split_dir)
    multimodal_forget, multimodal_retain = _split_records(multimodal_records, task, split_path, ratio)
    unimodal_forget, unimodal_retain = _split_records(unimodal_records, task, split_path, ratio)

    return (
        FlatPruneDataset(multimodal_forget, use_images=True),
        FlatPruneDataset(unimodal_forget, use_images=False),
        FlatPruneDataset(multimodal_retain, use_images=True),
        FlatPruneDataset(unimodal_retain, use_images=False),
        split_path,
    )
