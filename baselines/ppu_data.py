import json
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image
from torch.utils.data import Dataset


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_COMPLETE_SPLIT_FILE = REPO_ROOT /  "split" / "complete_split.json"
DEFAULT_SELECTIVE_SPLIT_FILE = REPO_ROOT /  "split" / "selective_split.json"
DEFAULT_PERSONA_SPLIT_FILE = REPO_ROOT /  "split" / "persona_split.json"

DEFAULT_SPLIT_FILES = {
    "complete": DEFAULT_COMPLETE_SPLIT_FILE,
    "selective": DEFAULT_SELECTIVE_SPLIT_FILE,
    "persona": DEFAULT_PERSONA_SPLIT_FILE,
}

REQUIRED_FIELDS = (
    "sample_id",
    "subject_id",
    "subject",
    "image_path",
    "question",
    "generated_answer",
)


class LLAVA_multimodal_Dataset(Dataset):
    """
    Dataset for train_pair JSON/Parquet files used by the MLLMU GA baseline.

    The input dataset is expected to be flattened to per-sample QA items. This
    dataset only normalizes field names and optionally loads the associated image
    for VQA.
    """

    def __init__(
        self,
        records: Sequence[Dict[str, Any]],
        vqa: bool = True,
        target_size: Optional[Tuple[int, int]] = None,
        sort_json_key: bool = True,
    ):
        super().__init__()
        self.records = list(records)
        self.vqa = vqa
        self.target_size = target_size
        self.sort_json_key = sort_json_key

    def __len__(self) -> int:
        return len(self.records)

    def json2token(self, obj: Any, sort_json_key: bool = True) -> str:
        if isinstance(obj, dict):
            if len(obj) == 1 and "text_sequence" in obj:
                return obj["text_sequence"]

            output = ""
            keys = sorted(obj.keys(), reverse=True) if sort_json_key else obj.keys()
            for key in keys:
                output += f"<s_{key}>" + self.json2token(obj[key], sort_json_key) + f"</s_{key}>"
            return output
        if isinstance(obj, list):
            return "<sep/>".join(self.json2token(item, sort_json_key) for item in obj)
        return str(obj)

    def _load_image(self, image_path: str) -> Image.Image:
        with Image.open(image_path) as image:
            loaded_image = image.convert("RGB")
        if self.target_size is not None:
            loaded_image = loaded_image.resize(self.target_size, Image.Resampling.LANCZOS)
        return loaded_image

    def _load_image_from_bytes(self, image_bytes: bytes) -> Image.Image:
        with Image.open(BytesIO(image_bytes)) as image:
            loaded_image = image.convert("RGB")
        if self.target_size is not None:
            loaded_image = loaded_image.resize(self.target_size, Image.Resampling.LANCZOS)
        return loaded_image

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        record = self.records[idx]
        image = None
        if self.vqa:
            image_pil = record.get("image")
            if isinstance(image_pil, Image.Image):
                image = image_pil
                if self.target_size is not None:
                    image = image.resize(self.target_size, Image.Resampling.LANCZOS)
            else:
                image_bytes = record.get("image_bytes")
                if image_bytes:
                    image = self._load_image_from_bytes(image_bytes)
                else:
                    image = self._load_image(record["image_path"])

        return {
            "image": image,
            "question": self.json2token(record["question"], sort_json_key=self.sort_json_key),
            "answer": self.json2token(record["generated_answer"], sort_json_key=self.sort_json_key),
            "sample_id": record["sample_id"],
            "subject_id": record["subject_id"],
            "subject": record["subject"],
            "image_path": record["image_path"],
        }


def _read_json(json_path: Path) -> Any:
    with json_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _read_parquet(parquet_path: Path) -> List[Dict[str, Any]]:
    import pandas as pd

    dataframe = pd.read_parquet(parquet_path)
    return dataframe.to_dict(orient="records")


def _extract_image_bytes(record: Dict[str, Any]) -> Optional[bytes]:
    image = record.get("image")
    if isinstance(image, dict):
        image_bytes = image.get("bytes")
        if isinstance(image_bytes, bytes):
            return image_bytes
        if image_bytes is not None:
            return bytes(image_bytes)
    return None


def _normalize_record(record: Dict[str, Any], index: int) -> Dict[str, Any]:
    sample_id = record.get("sample_id", record.get("id"))
    image = record.get("image")
    image_is_pil = isinstance(image, Image.Image)

    normalized = {
        "sample_id": sample_id,
        "subject_id": record.get("subject_id"),
        "subject": record.get("subject"),
        "image_path": record.get("image_path"),
        "question": record.get("question"),
        "generated_answer": record.get("generated_answer"),
        "image_bytes": _extract_image_bytes(record) if not image_is_pil else None,
        "image": image if image_is_pil else None,
    }

    missing = [field for field in REQUIRED_FIELDS if normalized.get(field) in (None, "")]
    if image_is_pil:
        missing = [f for f in missing if f not in ("image_path",)]
    if missing:
        raise ValueError(f"Record {index} in train dataset is missing required fields: {missing}")

    normalized["sample_id"] = str(normalized["sample_id"])
    normalized["subject_id"] = str(normalized["subject_id"])
    normalized["subject"] = str(normalized["subject"])
    normalized["image_path"] = str(normalized["image_path"]) if normalized["image_path"] is not None else ""
    normalized["question"] = str(normalized["question"])
    normalized["generated_answer"] = str(normalized["generated_answer"])
    return normalized


def load_train_records(train_ds_path: str) -> List[Dict[str, Any]]:
    dataset_path = Path(train_ds_path).expanduser().resolve()
    if dataset_path.suffix == ".json":
        raw_data = _read_json(dataset_path)
    elif dataset_path.suffix == ".parquet":
        raw_data = _read_parquet(dataset_path)
    else:
        raise ValueError(f"Unsupported train dataset format: {dataset_path}. Expected .json or .parquet")

    if not isinstance(raw_data, list):
        raise ValueError(f"Train dataset must resolve to a list of flattened records: {dataset_path}")

    return [_normalize_record(record, index) for index, record in enumerate(raw_data)]


def load_hf_train_records(config_name: str) -> List[Dict[str, Any]]:
    from datasets import load_dataset as hf_load_dataset

    ds = hf_load_dataset("closerG/ppu-bench", config_name)["train"]
    records = []
    for idx in range(len(ds)):
        records.append(_normalize_record(ds[idx], idx))
    return records


def _split_entries_to_sample_ids(entries: Any, split_path: Path, key: str) -> set:
    if not isinstance(entries, list):
        raise ValueError(f"Persona split key {key!r} must be a JSON array in {split_path}")

    sample_ids = set()
    for index, entry in enumerate(entries):
        if isinstance(entry, dict):
            sample_id = entry.get("id", entry.get("sample_id"))
        else:
            sample_id = entry
        if sample_id in (None, ""):
            raise ValueError(f"Persona split entry {key}[{index}] is missing id/sample_id in {split_path}")
        sample_ids.add(str(sample_id))
    return sample_ids


def _load_persona_split_ids(split_path: Path) -> Tuple[set, set]:
    split_data = _read_json(split_path)
    if not isinstance(split_data, dict):
        raise ValueError(f"Persona split file must be a JSON object: {split_path}")
    for key in ("delete_ids", "keep_ids"):
        if key not in split_data:
            raise KeyError(f"Persona split file {split_path} does not contain key {key}")

    forget_sample_ids = _split_entries_to_sample_ids(split_data["delete_ids"], split_path, "delete_ids")
    retain_sample_ids = _split_entries_to_sample_ids(split_data["keep_ids"], split_path, "keep_ids")
    return forget_sample_ids, retain_sample_ids


def _build_dataset_pair(
    forget_records: Iterable[Dict[str, Any]],
    retain_records: Iterable[Dict[str, Any]],
    vqa: bool,
) -> Tuple[LLAVA_multimodal_Dataset, LLAVA_multimodal_Dataset]:
    return LLAVA_multimodal_Dataset(list(forget_records), vqa=vqa), LLAVA_multimodal_Dataset(list(retain_records), vqa=vqa)


def get_complete_ds(
    train_ds_path: str = "",
    hf_config: str = "",
    ratio: int = 5,
    vqa: bool = True,
) -> Tuple[LLAVA_multimodal_Dataset, LLAVA_multimodal_Dataset]:
    if ratio not in {5, 10, 15, 30}:
        raise ValueError(f"Complete unlearning ratio must be one of 5/10/15/30, got {ratio}")

    split_path = DEFAULT_SPLIT_FILES["complete"]
    split_data = _read_json(split_path)
    forget_key = f"forget{ratio}"
    if forget_key not in split_data:
        raise KeyError(f"Split file {split_path} does not contain key {forget_key}")

    forget_subject_ids = {str(subject_id) for subject_id in split_data[forget_key]}

    if hf_config:
        records = load_hf_train_records(hf_config)
    else:
        records = load_train_records(train_ds_path)

    forget_records = [record for record in records if record["subject_id"] in forget_subject_ids]
    retain_records = [record for record in records if record["subject_id"] not in forget_subject_ids]
    return _build_dataset_pair(forget_records, retain_records, vqa=vqa)


def get_selective_ds(
    train_ds_path: str = "",
    hf_config: str = "",
    vqa: bool = True,
) -> Tuple[LLAVA_multimodal_Dataset, LLAVA_multimodal_Dataset]:
    split_path = DEFAULT_SPLIT_FILES["selective"]
    split_data = _read_json(split_path)
    if not isinstance(split_data, list):
        raise ValueError(f"Selective split file must be a JSON array: {split_path}")

    forget_sample_ids = {str(sample["id"]) for sample in split_data if "id" in sample}

    if hf_config:
        records = load_hf_train_records(hf_config)
    else:
        records = load_train_records(train_ds_path)

    forget_records = [record for record in records if record["sample_id"] in forget_sample_ids]
    retain_records = [record for record in records if record["sample_id"] not in forget_sample_ids]
    return _build_dataset_pair(forget_records, retain_records, vqa=vqa)


def get_persona_ds(
    train_ds_path: str = "",
    hf_config: str = "",
    vqa: bool = True,
) -> Tuple[LLAVA_multimodal_Dataset, LLAVA_multimodal_Dataset]:
    split_path = DEFAULT_SPLIT_FILES["persona"]
    forget_sample_ids, retain_sample_ids = _load_persona_split_ids(split_path)

    if hf_config:
        records = load_hf_train_records(hf_config)
    else:
        records = load_train_records(train_ds_path)

    forget_records = [record for record in records if record["sample_id"] in forget_sample_ids]
    retain_records = [record for record in records if record["sample_id"] in retain_sample_ids]
    return _build_dataset_pair(forget_records, retain_records, vqa=vqa)


__all__ = [
    "DEFAULT_COMPLETE_SPLIT_FILE",
    "DEFAULT_PERSONA_SPLIT_FILE",
    "DEFAULT_SELECTIVE_SPLIT_FILE",
    "LLAVA_multimodal_Dataset",
    "get_complete_ds",
    "get_persona_ds",
    "get_selective_ds",
    "load_hf_train_records",
    "load_train_records",
]
