from __future__ import annotations

import hashlib
import hmac
import os


SCHEME = "pbkdf2_sha256"
DEFAULT_ITERATIONS = 200_000


def _pbkdf2_sha256(password: str, salt: bytes, iterations: int) -> bytes:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)


def hash_password(password: str, iterations: int = DEFAULT_ITERATIONS) -> str:
    """
    Encodes password as:
      pbkdf2_sha256$<iterations>$<salt_hex>$<digest_hex>
    """
    salt = os.urandom(16)
    digest = _pbkdf2_sha256(password, salt, iterations)
    return f"{SCHEME}${iterations}${salt.hex()}${digest.hex()}"


def is_password_hashed(value: str) -> bool:
    return isinstance(value, str) and value.startswith(f"{SCHEME}$")


def verify_password(password: str, encoded: str) -> bool:
    try:
        scheme, iterations_s, salt_hex, digest_hex = encoded.split("$", 3)
        if scheme != SCHEME:
            return False
        iterations = int(iterations_s)
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(digest_hex)
    except Exception:
        return False

    computed = _pbkdf2_sha256(password, salt, iterations)
    return hmac.compare_digest(computed, expected)


def needs_rehash(encoded: str, min_iterations: int = DEFAULT_ITERATIONS) -> bool:
    if not is_password_hashed(encoded):
        return True
    try:
        _, iterations_s, _, _ = encoded.split("$", 3)
        return int(iterations_s) < min_iterations
    except Exception:
        return True
