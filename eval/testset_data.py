import re
import unicodedata
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TESTSET_IMAGE_ROOT = REPO_ROOT / "data" / "testset_images"
DEFAULT_TESTSET_OUTPUT_ROOT = REPO_ROOT / "eval" / "cache" / "testsets"
TESTSET_IDS = {2, 3, 4}
TESTSET_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def parse_testset_id(value):
    if value in (None, ""):
        return None
    testset_id = int(value)
    if testset_id not in TESTSET_IDS:
        raise ValueError(f"--testset must be one of 2/3/4, got {value}")
    return testset_id


def format_testset_name(testset_id):
    testset_id = parse_testset_id(testset_id)
    return f"testset-{testset_id:03d}"


def setting_file_stem(setting, ratio=None):
    normalized_setting = str(setting).strip().lower()
    if normalized_setting == "complete":
        if ratio is None:
            raise ValueError("Complete testset data requires a ratio")
        return f"complete_r{int(ratio)}"
    if normalized_setting in {"selective", "persona"}:
        return normalized_setting
    raise ValueError(f"Unsupported setting for testset data: {setting}")


def resolve_testset_data_file(setting, ratio, testset, output_root=None):
    output_root = _resolve_repo_path(output_root or DEFAULT_TESTSET_OUTPUT_ROOT)
    testset_name = format_testset_name(testset)
    return output_root / testset_name / f"{setting_file_stem(setting, ratio)}.parquet"


def _resolve_repo_path(candidate):
    path = Path(candidate).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def normalize_subject_name(value, drop_parenthetical=False):
    text = str(value)
    text = "".join(ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch))
    text = text.casefold()
    if drop_parenthetical:
        text = re.sub(r"\([^)]*\)", "", text)
    text = text.replace("&", "and")
    text = re.sub(r"[\"“”‘’']", "", text)
    text = re.sub(r"\bjr\b", "jr", text)
    text = re.sub(r"\bsr\b", "sr", text)
    text = re.sub(r"[^\w]+", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def resolve_subject_image_dir(subject, image_root=None):
    image_root = _resolve_repo_path(image_root or DEFAULT_TESTSET_IMAGE_ROOT)
    exact_dir = image_root / str(subject)
    if exact_dir.is_dir():
        return exact_dir

    directories = [path for path in image_root.iterdir() if path.is_dir()]
    for drop_parenthetical in (False, True):
        target = normalize_subject_name(subject, drop_parenthetical=drop_parenthetical)
        matches = [
            path
            for path in directories
            if normalize_subject_name(path.name, drop_parenthetical=drop_parenthetical) == target
        ]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            candidates = ", ".join(path.as_posix() for path in matches)
            raise ValueError(f"Ambiguous testset image directory for subject {subject!r}: {candidates}")

    raise FileNotFoundError(f"Missing testset image directory for subject {subject!r} under {image_root}")


def resolve_subject_testset_image(subject, testset, image_root=None):
    testset_id = parse_testset_id(testset)
    subject_dir = resolve_subject_image_dir(subject, image_root=image_root)
    stems = (f"{testset_id:03d}", f"{testset_id:02d}", str(testset_id))
    matches = [
        subject_dir / f"{stem}{extension}"
        for stem in stems
        for extension in TESTSET_IMAGE_EXTENSIONS
        if (subject_dir / f"{stem}{extension}").exists()
    ]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        candidates = ", ".join(path.as_posix() for path in matches)
        raise ValueError(f"Ambiguous testset image for subject {subject!r}, testset {testset_id}: {candidates}")
    raise FileNotFoundError(
        f"Missing testset image for subject {subject!r}, testset {testset_id} in {subject_dir}; "
        f"expected one of {', '.join(stems)} with .jpg/.jpeg/.png"
    )


def repo_relative_path(path):
    resolved_path = _resolve_repo_path(path)
    try:
        return resolved_path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return resolved_path.as_posix()
