"""Agent Vault — zero-knowledge encrypted agent backup/restore.

Wire-compatible with the agentvault CLI and the pod runner: the envelope is
``[4-byte BE header length][JSON header][AES-256-GCM body, tag appended]``
with an scrypt-derived KEK (N=2**17, r=8, p=1) wrapping a random DEK, and the
snapshot identity bound into the GCM AAD. Blobs encrypted here restore with
the CLI and vice versa. Argon2id blobs (very early CLI versions) are refused
with a pointer to the CLI, which carries the native argon2 dependency.

Auth is a Singularity compute API key (X-API-Key). Nothing here ever sends a
passphrase or plaintext to the network.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import secrets
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import httpx
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

VAULT_URL = "https://compute.x402layer.cc"
SCRYPT = {"N": 1 << 17, "r": 8, "p": 1}
DEFAULT_EXCLUDES = ("node_modules", "__pycache__", "**/*.log", "tmp", "cache", ".venv", ".git")


class VaultError(Exception):
    """Raised for API failures and integrity violations."""


def _uuid(v: str) -> str:
    import re
    if not re.fullmatch(r"[0-9a-fA-F-]{36}", v or ""):
        raise VaultError(f"not a snapshot id: {v!r}")
    return v.lower()


@dataclass
class VaultAgent:
    id: str
    name: str
    framework: str
    source: str


@dataclass
class VaultSnapshot:
    id: str
    agent_id: str
    size_bytes: int
    sha256: Optional[str]
    created_at: str


# ─── Envelope crypto (must stay byte-identical to agentvault-core) ──────────

def _aad_bytes(user_id: str, agent_id: str, backup_id: str) -> bytes:
    # Canonical key order + compact separators — byte-identical to
    # JSON.stringify({agentId, backupId, formatVersion, userId}) in core.
    return json.dumps(
        {"agentId": agent_id, "backupId": backup_id, "formatVersion": 1, "userId": user_id},
        separators=(",", ":"),
    ).encode("utf-8")


def _derive_kek(passphrase: str, salt: bytes) -> bytes:
    return hashlib.scrypt(
        passphrase.encode("utf-8"), salt=salt, n=SCRYPT["N"], r=SCRYPT["r"], p=SCRYPT["p"],
        maxmem=512 * 1024 * 1024, dklen=32,
    )


def encrypt_envelope(plaintext: bytes, passphrase: str, aad: bytes) -> bytes:
    salt = secrets.token_bytes(16)
    kek = _derive_kek(passphrase, salt)
    dek = secrets.token_bytes(32)
    dek_nonce = secrets.token_bytes(12)
    wrapped = AESGCM(kek).encrypt(dek_nonce, dek, aad)  # ciphertext||tag
    blob_nonce = secrets.token_bytes(12)
    body = AESGCM(dek).encrypt(blob_nonce, plaintext, aad)
    import base64
    header = {
        "formatVersion": 1,
        "kdf": "scrypt",
        "kdfParams": {"N": SCRYPT["N"], "r": SCRYPT["r"], "p": SCRYPT["p"],
                      "salt": base64.b64encode(salt).decode()},
        "cipher": "aes-256-gcm",
        "wrappedDek": {"nonce": base64.b64encode(dek_nonce).decode(),
                       "ciphertext": base64.b64encode(wrapped).decode()},
        "blobNonce": base64.b64encode(blob_nonce).decode(),
        "aad": json.loads(aad),
    }
    hj = json.dumps(header, separators=(",", ":")).encode("utf-8")
    return len(hj).to_bytes(4, "big") + hj + body


def decrypt_envelope(blob: bytes, passphrase: str, aad: bytes) -> bytes:
    import base64
    if len(blob) < 4:
        raise VaultError("malformed blob: too short")
    hlen = int.from_bytes(blob[:4], "big")
    if 4 + hlen > len(blob):
        raise VaultError("malformed blob: header length out of bounds")
    try:
        header = json.loads(blob[4:4 + hlen])
    except Exception as e:
        raise VaultError("malformed blob: invalid header JSON") from e
    kdf = header.get("kdf")
    if kdf != "scrypt":
        raise VaultError(f"this backup uses {kdf!r} key derivation — restore it with the agentvault CLI")
    p = header.get("kdfParams") or {}
    n, r, par = p.get("N"), p.get("r"), p.get("p")
    if not (isinstance(n, int) and 1 << 15 <= n <= 1 << 17 and (n & (n - 1)) == 0
            and isinstance(r, int) and 8 <= r <= 16 and isinstance(par, int) and 1 <= par <= 4):
        raise VaultError("malformed blob: unsupported scrypt parameters")
    salt = base64.b64decode(p["salt"])
    if len(salt) < 16:
        raise VaultError("malformed blob: salt too short")
    kek = hashlib.scrypt(passphrase.encode(), salt=salt, n=n, r=r, p=par,
                         maxmem=512 * 1024 * 1024, dklen=32)
    try:
        dek = AESGCM(kek).decrypt(
            base64.b64decode(header["wrappedDek"]["nonce"]),
            base64.b64decode(header["wrappedDek"]["ciphertext"]), aad)
        return AESGCM(dek).decrypt(base64.b64decode(header["blobNonce"]), blob[4 + hlen:], aad)
    except Exception as e:
        raise VaultError("incorrect passphrase or corrupted backup") from e


def _parse_aad_from_key(r2_key: str) -> bytes:
    parts = r2_key.split("/")
    if (len(parts) != 5 or parts[0] != "backups" or parts[4] != "blob.enc"
            or not parts[1] or not parts[2] or not parts[3]):
        raise VaultError(f"malformed r2 key: {r2_key}")
    return _aad_bytes(parts[1], parts[2], parts[3])


# ─── Packing / unpacking ────────────────────────────────────────────────────

def _pack_dir(path: Path, excludes: tuple[str, ...]) -> bytes:
    """Gzipped tar of the directory (by basename) + a manifest, junk excluded."""
    def keep(member: tarfile.TarInfo) -> Optional[tarfile.TarInfo]:
        rel = member.name.split("/", 1)[1] if "/" in member.name else ""
        segments = rel.split("/")
        for pat in excludes:
            needle = pat.replace("**/", "").replace("*", "")
            if needle and any(needle in seg for seg in segments):
                return None
        return member

    buf = io.BytesIO()
    manifest = json.dumps({
        "agentName": path.name, "framework": path.name, "frameworkVersion": None,
        "createdAt": __import__("datetime").datetime.now(__import__("datetime").timezone.utc)
            .isoformat(), "formatVersion": 1,
    }).encode()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        tf.add(str(path), arcname=path.name, filter=keep)
        mi = tarfile.TarInfo("manifest.json")
        mi.size = len(manifest)
        tf.addfile(mi, io.BytesIO(manifest))
    return buf.getvalue()


def _safe_extract(tarball: bytes, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    root = dest.resolve()
    with tarfile.open(fileobj=io.BytesIO(tarball), mode="r:gz") as tf:
        members = []
        # Preflight EVERY member before writing anything, so a hostile entry
        # late in the archive cannot leave a partial extraction behind.
        for m in tf.getmembers():
            if not (m.isreg() or m.isdir()):
                continue  # no links, devices, or FIFOs — ever
            target = (root / m.name).resolve()
            if not str(target).startswith(str(root) + os.sep) and target != root:
                raise VaultError(f"archive contains unsafe path: {m.name}")
            members.append(m)
        for m in members:
            m.mode &= ~0o7022  # strip setuid/setgid/sticky + group/world write
            tf.extract(m, root)


# ─── Client ─────────────────────────────────────────────────────────────────

class VaultClient:
    """Agent Vault operations with a compute API key.

    >>> vault = VaultClient(api_key="x402c_...")
    >>> vault.backup_dir("~/my-agent", passphrase="...")  # doctest: +SKIP
    >>> vault.restore(snapshot_id, passphrase="...", dest="./restored")  # doctest: +SKIP
    """

    def __init__(self, api_key: str, base_url: str = VAULT_URL, timeout: float = 300.0):
        from urllib.parse import urlsplit
        u = urlsplit(base_url)
        local = (u.hostname or "") in ("localhost", "127.0.0.1", "::1")
        if u.scheme != "https" and not local:
            raise VaultError("base_url must be https (the API key travels in a header)")
        if u.username or u.password or u.query or u.fragment:
            raise VaultError("base_url must be a bare origin")
        self._base = base_url.rstrip("/")
        self._http = httpx.Client(timeout=timeout, headers={"x-api-key": api_key})

    def _call(self, method: str, path: str, **kw) -> dict:
        res = self._http.request(method, self._base + path, **kw)
        try:
            data = res.json()
        except Exception:
            data = {}
        if res.status_code >= 400:
            raise VaultError(data.get("error") or f"request failed: {res.status_code}")
        return data

    # — metadata —
    def agents(self) -> list[VaultAgent]:
        return [VaultAgent(a["id"], a["name"], a["framework"], a["source"])
                for a in self._call("GET", "/backups/agents")["agents"]]

    def snapshots(self, agent_id: Optional[str] = None) -> list[VaultSnapshot]:
        from urllib.parse import quote
        q = f"?agentId={quote(str(agent_id), safe='')}" if agent_id else ""
        return [VaultSnapshot(b["id"], b["agent_id"], b["size_bytes"], b.get("sha256"), b["created_at"])
                for b in self._call("GET", f"/backups{q}")["backups"]]

    def usage(self) -> dict:
        return self._call("GET", "/backups/usage")

    def subscribe_pro(self) -> dict:
        """Activate Vault Pro ($3/mo from credits)."""
        return self._call("POST", "/backups/subscribe")

    def delete_snapshot(self, snapshot_id: str) -> None:
        self._call("DELETE", f"/backups/{_uuid(snapshot_id)}")

    # — the real work —
    def backup_dir(self, path: str, passphrase: str, name: Optional[str] = None,
                   excludes: tuple[str, ...] = DEFAULT_EXCLUDES) -> str:
        """Encrypt + upload a directory as an agent snapshot. Returns the snapshot id."""
        p = Path(path).expanduser().resolve()
        if not p.is_dir() or p.is_symlink():
            raise VaultError(f"--path must be a real directory: {p}")
        if str(p) in ("/", str(Path.home().resolve())):
            raise VaultError("refusing to back up the filesystem root or your entire home directory")
        display = (name or p.name).strip()
        import re
        slug = re.sub(r"-+", "-", re.sub(r"[^a-z0-9-]+", "-", display.lower())).strip("-")
        if not slug:
            raise VaultError("cannot derive a framework label — pass name=")
        digest = hashlib.sha256(str(p).encode()).hexdigest()[:6]
        framework = (f"custom-{slug}"[:25]).rstrip("-") + f"-{digest}"

        agent = self._call("POST", "/backups/agents",
                           json={"name": display, "framework": framework})["agent"]
        tarball = _pack_dir(p, excludes)
        res = self._call("POST", "/backups", json={"agentId": agent["id"], "sizeBytes": len(tarball)})
        aad = _parse_aad_from_key(res["r2Key"])
        blob = encrypt_envelope(tarball, passphrase, aad)
        up = httpx.put(res["uploadUrl"], content=blob,
                       headers={"content-type": "application/octet-stream"}, timeout=600.0)
        if up.status_code >= 400:
            raise VaultError(f"upload failed: {up.status_code}")
        self._call("POST", f"/backups/{res['backupId']}/complete",
                   json={"sha256": hashlib.sha256(blob).hexdigest()})
        return res["backupId"]

    def restore(self, snapshot_id: str, passphrase: str, dest: str) -> Path:
        """Download, decrypt, and safely unpack a snapshot into ``dest``."""
        info = self._call("GET", f"/backups/{_uuid(snapshot_id)}/restore")
        dl = httpx.get(info["downloadUrl"], timeout=600.0)
        if dl.status_code >= 400:
            raise VaultError(f"download failed: {dl.status_code}")
        tarball = decrypt_envelope(dl.content, passphrase, _parse_aad_from_key(info["r2Key"]))
        out = Path(dest).expanduser()
        _safe_extract(tarball, out)
        return out
