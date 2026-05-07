import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CANONICAL_FILE = REPO_ROOT / "eval" / "cache" / "ppu_eval_canonical.parquet"
DEFAULT_COMPLETE_SPLIT_FILE = REPO_ROOT / "data" / "split" / "complete_split.json"
DEFAULT_PERSONA_SPLIT_FILE = REPO_ROOT / "data" / "split" / "persona_split.json"
DEFAULT_SELECTIVE_SPLIT_FILE = REPO_ROOT / "data" / "split" / "selective_split.json"
TASK_ALIASES = {
    "qa": "generation",
    "class": "classification",
    "classification": "classification",
    "cloze": "cloze",
    "generation": "generation",
}


def normalize_task_name(task):
    normalized_task = str(task).strip().lower()
    if normalized_task not in TASK_ALIASES:
        raise ValueError(f"Unsupported task: {task}")
    return TASK_ALIASES[normalized_task]


class PPUEvalData:
    def __init__(self, canonical_path=None, complete_split_path=None, selective_split_path=None, persona_split_path=None):
        self.canonical_path = self._resolve_repo_path(canonical_path or DEFAULT_CANONICAL_FILE)
        self.complete_split_path = self._resolve_split_path(complete_split_path, DEFAULT_COMPLETE_SPLIT_FILE)
        self.selective_split_path = self._resolve_split_path(selective_split_path, DEFAULT_SELECTIVE_SPLIT_FILE)
        self.persona_split_path = self._resolve_split_path(persona_split_path, DEFAULT_PERSONA_SPLIT_FILE)
        self._dataframe = None
        self._complete_split = None
        self._selective_split_ids = None
        self._persona_split_ids = None

    def _resolve_repo_path(self, candidate):
        path = Path(candidate).expanduser()
        if not path.is_absolute():
            path = REPO_ROOT / path
        return path.resolve()

    def _resolve_split_path(self, candidate, default_path):
        if candidate is None:
            return default_path
        path = self._resolve_repo_path(candidate)
        if path.suffix == ".json":
            return path
        nested_name = None
        if default_path.name == "complete_split.json":
            file_name = "complete_split.json"
        elif default_path.name == "persona_split.json":
            file_name = "persona_split.json"
        elif default_path.name == "selective_split.json":
            file_name = "selective_split.json"
        elif default_path.name == "forget_subject_ids.json":
            nested_name = "CompleteUnlearning"
            file_name = "forget_subject_ids.json"
        elif default_path.name == "sensitive_info_subject_id_id.json":
            nested_name = "SelectiveUnlearning"
            file_name = "sensitive_info_subject_id_id.json"
        elif default_path.name == "majority_ids.json":
            nested_name = "PersonaUnlearning"
            file_name = "majority_ids.json"
        else:
            raise ValueError(f"Unsupported split default path: {default_path}")
        direct_candidate = path / file_name
        if direct_candidate.exists():
            return direct_candidate.resolve()
        if nested_name is not None:
            nested_candidate = path / nested_name / file_name
            if nested_candidate.exists():
                return nested_candidate.resolve()
        return direct_candidate.resolve()

    def _load_dataframe(self):
        if self._dataframe is None:
            if not self.canonical_path.exists():
                raise FileNotFoundError(
                    f"Canonical eval data not found: {self.canonical_path}. Run eval/prepare_eval_data.py first."
                )
            self._dataframe = pd.read_parquet(self.canonical_path)
            self._dataframe["sample_id"] = self._dataframe["sample_id"].astype(str)
            self._dataframe["subject_id"] = self._dataframe["subject_id"].astype(str)
            self._dataframe["task_type"] = self._dataframe["task_type"].astype(str)
            self._dataframe["modality"] = self._dataframe["modality"].astype(str)
        return self._dataframe

    def set_dataframe(self, df):
        self._dataframe = df.copy()
        self._dataframe["sample_id"] = self._dataframe["sample_id"].astype(str)
        self._dataframe["subject_id"] = self._dataframe["subject_id"].astype(str)
        self._dataframe["task_type"] = self._dataframe["task_type"].astype(str)
        self._dataframe["modality"] = self._dataframe["modality"].astype(str)

    def _load_complete_split(self):
        if self._complete_split is None:
            with self.complete_split_path.open("r", encoding="utf-8") as handle:
                self._complete_split = json.load(handle)
        return self._complete_split

    def _load_selective_split_ids(self):
        if self._selective_split_ids is None:
            dataframe = self._load_dataframe()
            if "source_from" in dataframe.columns:
                selective_rows = dataframe[dataframe["source_from"] == "sensitive_info"]
                self._selective_split_ids = set(selective_rows["sample_id"].astype(str).tolist())
            else:
                with self.selective_split_path.open("r", encoding="utf-8") as handle:
                    selective_data = json.load(handle)
                if not isinstance(selective_data, list):
                    raise ValueError(f"Selective split file must be a JSON array: {self.selective_split_path}")
                self._selective_split_ids = set()
                for entry in selective_data:
                    sample_id = entry.get("id", entry.get("sample_id"))
                    if sample_id:
                        self._selective_split_ids.add(str(sample_id))
        return self._selective_split_ids

    def _split_entries_to_sample_ids(self, entries, key):
        if not isinstance(entries, list):
            raise ValueError(f"Persona split key {key!r} must be a JSON array in {self.persona_split_path}")

        sample_ids = set()
        for index, entry in enumerate(entries):
            if isinstance(entry, dict):
                sample_id = entry.get("id", entry.get("sample_id"))
            else:
                sample_id = entry
            if sample_id in (None, ""):
                raise ValueError(
                    f"Persona split entry {key}[{index}] is missing id/sample_id in {self.persona_split_path}"
                )
            sample_ids.add(str(sample_id))
        return sample_ids

    def _load_persona_split_ids(self):
        if self._persona_split_ids is None:
            with self.persona_split_path.open("r", encoding="utf-8") as handle:
                split_data = json.load(handle)
            if not isinstance(split_data, dict):
                raise ValueError(f"Persona split file must be a JSON object: {self.persona_split_path}")
            for key in ("delete_ids", "keep_ids"):
                if key not in split_data:
                    raise KeyError(f"Persona split file {self.persona_split_path} does not contain key {key}")
            self._persona_split_ids = (
                self._split_entries_to_sample_ids(split_data["delete_ids"], "delete_ids"),
                self._split_entries_to_sample_ids(split_data["keep_ids"], "keep_ids"),
            )
        return self._persona_split_ids

    def get_rows(self, setting, task, subset, ratio=None, modality=None):
        normalized_setting = str(setting).strip().lower()
        normalized_subset = str(subset).strip().lower()
        normalized_task = normalize_task_name(task)

        dataframe = self._load_dataframe()
        rows = dataframe[dataframe["task_type"] == normalized_task].copy()
        if modality:
            rows = rows[rows["modality"] == str(modality).strip().lower()].copy()

        if normalized_setting == "complete":
            if ratio not in {5, 10, 15, 30}:
                raise ValueError(f"Complete ratio must be one of 5/10/15/30, got {ratio}")
            split_data = self._load_complete_split()
            forget_subject_ids = {str(value) for value in split_data[f"forget{ratio}"]}
            if normalized_subset == "forget":
                rows = rows[rows["subject_id"].isin(forget_subject_ids)].copy()
            elif normalized_subset == "retain":
                rows = rows[~rows["subject_id"].isin(forget_subject_ids)].copy()
            else:
                raise ValueError(f"Unsupported subset: {subset}")
        elif normalized_setting == "selective":
            selective_split_ids = self._load_selective_split_ids()
            if normalized_subset == "forget":
                rows = rows[rows["sample_id"].isin(selective_split_ids)].copy()
            elif normalized_subset == "retain":
                rows = rows[~rows["sample_id"].isin(selective_split_ids)].copy()
            else:
                raise ValueError(f"Unsupported subset: {subset}")
        elif normalized_setting == "persona":
            forget_sample_ids, retain_sample_ids = self._load_persona_split_ids()
            if normalized_subset == "forget":
                rows = rows[rows["sample_id"].isin(forget_sample_ids)].copy()
            elif normalized_subset == "retain":
                rows = rows[rows["sample_id"].isin(retain_sample_ids)].copy()
            else:
                raise ValueError(f"Unsupported subset: {subset}")
        else:
            raise ValueError(f"Unsupported setting: {setting}")

        return rows.reset_index(drop=True)

    def get_modalities(self, setting, task, subset, ratio=None):
        rows = self.get_rows(setting=setting, task=task, subset=subset, ratio=ratio)
        return sorted(rows["modality"].dropna().unique().tolist())

    def sample_few_shot(self, task, modality, shot_num, few_shot_data=None, exclude_sample_ids=None, setting=None, ratio=None):
        if shot_num <= 0:
            return []

        source = PPUEvalData(
            canonical_path=few_shot_data or self.canonical_path,
            complete_split_path=self.complete_split_path,
            selective_split_path=self.selective_split_path,
            persona_split_path=self.persona_split_path,
        )
        if setting is None:
            rows = source._load_dataframe().copy()
            rows = rows[rows["task_type"] == normalize_task_name(task)]
            rows = rows[rows["modality"] == str(modality).strip().lower()]
        else:
            rows = source.get_rows(setting=setting, task=task, subset="retain", ratio=ratio, modality=modality)

        if exclude_sample_ids:
            excluded = {str(value) for value in exclude_sample_ids}
            rows = rows[~rows["sample_id"].isin(excluded)].copy()

        if rows.empty:
            return []

        sample_count = min(int(shot_num), len(rows))
        sampled = rows.sample(n=sample_count, random_state=10)
        return sampled.to_dict(orient="records")
