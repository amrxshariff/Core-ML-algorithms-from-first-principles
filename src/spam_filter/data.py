from __future__ import annotations

import http.client
import io
import os
import urllib.request
import zipfile
from pathlib import Path

MIRROR_URLS: tuple[str, ...] = (
    "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip",
    "http://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip",
)
DATASET_FILENAME = "SMSSpamCollection"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = Path(os.environ.get("SPAM_FILTER_DATA_DIR", REPO_ROOT / "data"))

EXPECTED_TOTAL = 5574
EXPECTED_SPAM = 747


def _download_archive(urls: tuple[str, ...] = MIRROR_URLS, timeout: float = 30.0) -> bytes:
    """Fetch the raw zip bytes, trying each source in order.

    Isolated from ensure_dataset so tests can replace it without touching urllib.
    """
    errors: list[str] = []
    for url in urls:
        request = urllib.request.Request(
            url, headers={"User-Agent": "spam-filter/0.1 (coursework)"}
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        # OSError covers URLError, HTTPError, TimeoutError and connection resets;
        # HTTPException covers truncated or malformed responses.
        except (OSError, http.client.HTTPException) as exc:
            errors.append(f"{url}: {exc}")
    raise ConnectionError(
        "Could not download the dataset from any source:\n  " + "\n  ".join(errors)
    )


def _extract_member(archive_bytes: bytes) -> bytes:
    """Pull SMSSpamCollection out of the zip, failing loudly if it is not there."""
    try:
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as zf:
            member_name = next(
                (n for n in zf.namelist() if Path(n).name == DATASET_FILENAME), None
            )
            if member_name is None:
                raise FileNotFoundError(
                    f"{DATASET_FILENAME} not in archive members: {zf.namelist()}"
                )
            return zf.read(member_name)
    except zipfile.BadZipFile as exc:
        raise ValueError("Downloaded archive is not a valid zip file") from exc


def ensure_dataset(data_dir: Path = DEFAULT_DATA_DIR, timeout: float = 30.0) -> Path:
    """Return the path to the local dataset, downloading it only if absent."""
    data_dir = Path(data_dir)
    target_path = data_dir / DATASET_FILENAME
    if target_path.exists():
        return target_path

    data_dir.mkdir(parents=True, exist_ok=True)
    payload = _extract_member(_download_archive(timeout=timeout))

    # Write to a sibling temp file, then rename: an interrupted write must never
    # leave a truncated file that the exists() check above would cache forever.
    tmp_path = target_path.with_name(target_path.name + ".part")
    tmp_path.write_bytes(payload)
    tmp_path.replace(target_path)
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


def check_integrity(data_dir: Path = DEFAULT_DATA_DIR) -> None:
    """Fail loudly if the local file is not the canonical corpus."""
    _, labels = load_sms_spam(data_dir=data_dir)
    spam = sum(labels)
    if (len(labels), spam) != (EXPECTED_TOTAL, EXPECTED_SPAM):
        raise ValueError(
            f"Unexpected dataset: {len(labels)} messages, {spam} spam "
            f"(expected {EXPECTED_TOTAL}, {EXPECTED_SPAM})"
        )


if __name__ == "__main__":
    texts, labels = load_sms_spam()
    spam_count = sum(1 for label in labels if label == 1)
    print(f"Loaded {len(texts)} messages ({spam_count} spam, {len(texts) - spam_count} ham).")
