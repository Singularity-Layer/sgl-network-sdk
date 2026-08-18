import io
import tarfile
import pytest
from pathlib import Path

from singularity_grid.vault import (
    VaultError, _aad_bytes, _pack_dir, _safe_extract,
    decrypt_envelope, encrypt_envelope,
)

AAD = _aad_bytes("0xw", "11111111-2222-4333-8444-555566667777", "99999999-8888-4777-8666-555544443333")


def test_envelope_roundtrip():
    blob = encrypt_envelope(b"soul", "pass", AAD)
    assert decrypt_envelope(blob, "pass", AAD) == b"soul"


def test_wrong_passphrase_and_tampered_aad_fail():
    blob = encrypt_envelope(b"soul", "pass", AAD)
    with pytest.raises(VaultError):
        decrypt_envelope(blob, "wrong", AAD)
    other = _aad_bytes("0xw", "11111111-2222-4333-8444-555566667777", "aaaaaaaa-8888-4777-8666-555544443333")
    with pytest.raises(VaultError):
        decrypt_envelope(blob, "pass", other)


def test_hostile_scrypt_params_refused():
    import json
    blob = encrypt_envelope(b"x", "p", AAD)
    hlen = int.from_bytes(blob[:4], "big")
    header = json.loads(blob[4:4 + hlen])
    header["kdfParams"]["N"] = 1 << 26  # 8 GiB — resource exhaustion attempt
    hj = json.dumps(header, separators=(",", ":")).encode()
    forged = len(hj).to_bytes(4, "big") + hj + blob[4 + hlen:]
    with pytest.raises(VaultError, match="scrypt"):
        decrypt_envelope(forged, "p", AAD)


def test_pack_excludes_junk(tmp_path):
    agent = tmp_path / "bot"
    (agent / "memories").mkdir(parents=True)
    (agent / "memories" / "core.md").write_text("42")
    (agent / "node_modules" / "junk").mkdir(parents=True)
    (agent / "node_modules" / "junk" / "x.js").write_text("junk")
    tarball = _pack_dir(agent, ("node_modules",))
    names = tarfile.open(fileobj=io.BytesIO(tarball)).getnames()
    assert any("memories/core.md" in n for n in names)
    assert not any("node_modules" in n for n in names)
    assert "manifest.json" in names


def test_safe_extract_rejects_traversal(tmp_path):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        evil = tarfile.TarInfo("../../evil.txt")
        evil.size = 4
        tf.addfile(evil, io.BytesIO(b"boom"))
    with pytest.raises(VaultError, match="unsafe path"):
        _safe_extract(buf.getvalue(), tmp_path / "out")


def test_safe_extract_skips_links(tmp_path):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        link = tarfile.TarInfo("sneaky")
        link.type = tarfile.SYMTYPE
        link.linkname = "/etc/passwd"
        tf.addfile(link)
        ok = tarfile.TarInfo("fine.txt")
        ok.size = 2
        tf.addfile(ok, io.BytesIO(b"ok"))
    _safe_extract(buf.getvalue(), tmp_path / "out")
    assert (tmp_path / "out" / "fine.txt").read_text() == "ok"
    assert not (tmp_path / "out" / "sneaky").exists()
