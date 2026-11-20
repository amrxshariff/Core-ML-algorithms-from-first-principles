import io
import zipfile

import pytest

from spam_filter import data as data_module

SAMPLE = "ham\tWhat you doing?\nspam\tWIN a brand new car! Txt WIN to 80086\n"


def _fake_zip(payload: str = SAMPLE, name: str = "SMSSpamCollection") -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("readme", "irrelevant")  # mirrors the real archive
        zf.writestr(name, payload)
    return buf.getvalue()


@pytest.fixture
def no_network(monkeypatch):
    """Turn any download attempt into a test failure, not a slow test."""

    def _boom(*args, **kwargs):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(data_module, "_download_archive", _boom)


def test_downloads_and_extracts(tmp_path, monkeypatch):
    monkeypatch.setattr(data_module, "_download_archive", lambda **kw: _fake_zip())
    path = data_module.ensure_dataset(data_dir=tmp_path)
    assert path.name == "SMSSpamCollection"
    assert path.read_text(encoding="utf-8").startswith("ham\t")
    assert not (tmp_path / "SMSSpamCollection.part").exists()


def test_uses_local_cache(tmp_path, no_network):
    (tmp_path / "SMSSpamCollection").write_text(SAMPLE, encoding="utf-8")
    texts, labels = data_module.load_sms_spam(data_dir=tmp_path)
    assert labels == [0, 1]
    assert texts[1].startswith("WIN")


def test_string_labels(tmp_path, no_network):
    (tmp_path / "SMSSpamCollection").write_text(SAMPLE, encoding="utf-8")
    _, labels = data_module.load_sms_spam(data_dir=tmp_path, binary_labels=False)
    assert labels == ["ham", "spam"]


def test_bad_archive_raises_value_error(tmp_path, monkeypatch):
    monkeypatch.setattr(data_module, "_download_archive", lambda **kw: b"not a zip")
    with pytest.raises(ValueError):
        data_module.ensure_dataset(data_dir=tmp_path)


def test_missing_member_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(
        data_module, "_download_archive", lambda **kw: _fake_zip(name="other-file")
    )
    with pytest.raises(FileNotFoundError):
        data_module.ensure_dataset(data_dir=tmp_path)


def test_malformed_line_raises(tmp_path, no_network):
    (tmp_path / "SMSSpamCollection").write_text("ham no tab here\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing tab"):
        data_module.load_sms_spam(data_dir=tmp_path)


def test_unknown_label_raises(tmp_path, no_network):
    (tmp_path / "SMSSpamCollection").write_text("maybe\thello\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unexpected label"):
        data_module.load_sms_spam(data_dir=tmp_path)


def test_integrity_check_rejects_wrong_corpus(tmp_path, no_network):
    (tmp_path / "SMSSpamCollection").write_text(SAMPLE, encoding="utf-8")
    with pytest.raises(ValueError, match="Unexpected dataset"):
        data_module.check_integrity(data_dir=tmp_path)


@pytest.mark.network
def test_real_download_smoke(tmp_path):
    """Opt-in only: pytest -m network. Deselected in CI."""
    path = data_module.ensure_dataset(data_dir=tmp_path)
    assert path.stat().st_size > 400_000
