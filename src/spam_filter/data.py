from __future__ import annotations

import io
from pathlib import Path
import urllib.request
import zipfile

UCI_SMS_SPAM_ZIP_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
DEFAULT_DATA_DIR = Path("data")
DATASET_FILENAME = "SMSSpamCollection"


def ensure_dataset(data_dir: Path = DEFAULT_DATA_DIR, timeout: float = 30.0) -> Path:
    """Download and extract SMSSpamCollection into data_dir if not already present."""
    data_dir.mkdir(parents=True, exist_ok=True)
    target_path = data_dir / DATASET_FILENAME

    if target_path.exists():
        return target_path

    with urllib.request.urlopen(UCI_SMS_SPAM_ZIP_URL, timeout=timeout) as response:
        archive_bytes = response.read()

    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as zf:
        member_name = next(
            (name for name in zf.namelist() if Path(name).name == DATASET_FILENAME),
            None,
        )
        if member_name is None:
            raise FileNotFoundError(
                f"{DATASET_FILENAME} not found inside archive members: {zf.namelist()}"
            )
        target_path.write_bytes(zf.read(member_name))

    return target_path


def load_sms_spam(
    data_dir: Path = DEFAULT_DATA_DIR,
    *,
    binary_labels: bool = True,
) -> tuple[list[str], list[int] | list[str]]:
    """Load SMS messages and labels from the cached or newly downloaded dataset."""
    path = ensure_dataset(data_dir=data_dir)
    raw_text = path.read_text(encoding="utf-8", errors="replace")

    texts: list[str] = []
    labels: list[int] | list[str] = []

    for line_no, line in enumerate(raw_text.splitlines(), start=1):
        if not line.strip():
            continue
        label_str, sep, message = line.partition("\t")
        if not sep:
            raise ValueError(f"Malformed line {line_no} in {path}: missing tab separator")

        label_clean = label_str.strip().lower()
        if label_clean not in {"ham", "spam"}:
            raise ValueError(f"Unexpected label {label_str!r} on line {line_no}")

        texts.append(message)
        if binary_labels:
            labels.append(1 if label_clean == "spam" else 0)
        else:
            labels.append(label_clean)

    return texts, labels