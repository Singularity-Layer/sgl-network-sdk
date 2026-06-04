"""
End-to-end encryption for the SGL grid (client side).

Must match the node, orchestrator, and browser/TS clients byte-for-byte:
  X25519 ECDH -> HKDF-SHA256 -> XChaCha20-Poly1305 (24-byte nonce), AAD-bound.
Sealed blob layout: nonce(24) || ciphertext, base58.

The orchestrator only ever relays ciphertext — it never sees the prompt or reply.
"""

from __future__ import annotations

import base58
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from nacl import bindings

ALGO_V2 = "x25519-xchacha20poly1305-hkdf-v2"
ALGO_V2_STREAM = "x25519-xchacha20poly1305-hkdf-v2-stream"

_SALT = b"sgl-e2e-v2-salt"
_INFO_INPUT = b"sgl-e2e-v2-input"
_INFO_OUTPUT = b"sgl-e2e-v2-output"


def _hkdf(shared: bytes, info: bytes) -> bytes:
    return HKDF(algorithm=hashes.SHA256(), length=32, salt=_SALT, info=info).derive(shared)


def _aad_input(node_b58: str, eph_b58: str, resp_b58: str) -> bytes:
    return f"sgl-aad/v2/input|node={node_b58}|eph={eph_b58}|resp={resp_b58}".encode()


def _aad_output(resp_b58: str, eph_b58: str) -> bytes:
    return f"sgl-aad/v2/output|resp={resp_b58}|eph={eph_b58}".encode()


def _aad_stream(resp_b58: str, eph_b58: str, nonce_b58: str, seq: int, final: bool) -> bytes:
    f = 1 if final else 0
    return (
        f"sgl-aad/v2/stream|resp={resp_b58}|eph={eph_b58}|nonce={nonce_b58}|seq={seq}|final={f}"
    ).encode()


def _b58e(b: bytes) -> str:
    return base58.b58encode(b).decode()


def _b58d(s: str) -> bytes:
    return base58.b58decode(s)


def _gen_keypair() -> tuple[bytes, bytes]:
    sk = bindings.randombytes(32)
    pk = bindings.crypto_scalarmult_base(sk)
    return sk, pk


def new_response_keypair() -> tuple[bytes, str]:
    """The caller's response keypair — the node seals its reply to this."""
    sk, pk = _gen_keypair()
    return sk, _b58e(pk)


def random_nonce_b58() -> str:
    """A per-request nonce bound into every stream chunk's AAD."""
    return _b58e(bindings.randombytes(16))


def seal_input(node_pub_b58: str, resp_pub_b58: str, plaintext: bytes) -> tuple[str, str]:
    """Seal the prompt to the node's X25519 key. Returns (ciphertext_b58, ephemeral_pub_b58)."""
    node_pub = _b58d(node_pub_b58)
    eph_sk, eph_pk = _gen_keypair()
    eph_b58 = _b58e(eph_pk)
    shared = bindings.crypto_scalarmult(eph_sk, node_pub)
    key = _hkdf(shared, _INFO_INPUT)
    nonce = bindings.randombytes(24)
    aad = _aad_input(node_pub_b58, eph_b58, resp_pub_b58)
    ct = bindings.crypto_aead_xchacha20poly1305_ietf_encrypt(plaintext, aad, nonce, key)
    return _b58e(nonce + ct), eph_b58


def open_output(resp_sk: bytes, resp_pub_b58: str, node_eph_b58: str, ct_b58: str) -> bytes:
    """Open the node's (non-stream) reply sealed to our response key."""
    shared = bindings.crypto_scalarmult(resp_sk, _b58d(node_eph_b58))
    key = _hkdf(shared, _INFO_OUTPUT)
    aad = _aad_output(resp_pub_b58, node_eph_b58)
    blob = _b58d(ct_b58)
    return bindings.crypto_aead_xchacha20poly1305_ietf_decrypt(blob[24:], aad, blob[:24], key)


def stream_out_key(resp_sk: bytes, node_stream_eph_b58: str) -> bytes:
    """Derive the stream output key once from the node's stream ephemeral (chunk 0)."""
    shared = bindings.crypto_scalarmult(resp_sk, _b58d(node_stream_eph_b58))
    return _hkdf(shared, _INFO_OUTPUT)


def open_stream_chunk(
    out_key: bytes,
    resp_pub_b58: str,
    eph_b58: str,
    nonce_b58: str,
    seq: int,
    final: bool,
    ct_b58: str,
) -> bytes:
    """Open one stream chunk with the precomputed key + nonce/seq/final-bound AAD."""
    aad = _aad_stream(resp_pub_b58, eph_b58, nonce_b58, seq, final)
    blob = _b58d(ct_b58)
    return bindings.crypto_aead_xchacha20poly1305_ietf_decrypt(blob[24:], aad, blob[:24], out_key)


if __name__ == "__main__":
    # Cross-language vector from the node's encryption.rs: proves this module's
    # X25519+HKDF+XChaCha20+AAD+base58 match the Rust/TS/browser clients byte-for-byte.
    import hashlib
    import json

    node_secret = bytes([0x42] * 32)
    node_x25519_sk = hashlib.sha256(b"sgl-x25519-derive:" + node_secret).digest()
    node_pub_b58 = _b58e(bindings.crypto_scalarmult_base(node_x25519_sk))

    eph_b58 = "DQFdwcBsqukJEBn9UNfQruaTHKHxHFMVRA2B5qZuFdfB"
    resp_b58 = "2L54SXdEHm5mraF2X2GPid3m4PSkwVehEvhk487mWTx8"
    ct_b58 = ("gbEA6dFFVPxdar6e8QsjPKWj7xHcBo32nAqweQC5arnt4M5LmHhjREKoUTdVZsU6mmkxKu1Xmv"
              "Eo4oG8EUySndq2ytTyzDgyfMjyBSmPE2fqjdPDzKYtdrC2kZbAfCXv227GczHgmtQBqchA5qMB5"
              "ydxgxYnk9V8jb8sifTjHM61iEQkisdwYCqna")

    shared = bindings.crypto_scalarmult(node_x25519_sk, _b58d(eph_b58))
    key = _hkdf(shared, _INFO_INPUT)
    aad = _aad_input(node_pub_b58, eph_b58, resp_b58)
    blob = _b58d(ct_b58)
    pt = bindings.crypto_aead_xchacha20poly1305_ietf_decrypt(blob[24:], aad, blob[:24], key)
    got = json.loads(pt)
    expected = {"messages": [{"role": "user", "content": "cross-lang v2 test"}],
                "temperature": 0.7, "max_tokens": 512}
    assert got == expected, f"MISMATCH: {got}"
    print("OK: singularity-grid E2E crypto matches the cross-language vector")
