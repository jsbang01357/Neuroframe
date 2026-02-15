from __future__ import annotations

import unittest

from storage.security import hash_password, is_password_hashed, needs_rehash, verify_password


class PasswordSecurityTests(unittest.TestCase):
    def test_hash_and_verify_roundtrip(self):
        encoded = hash_password("correct horse battery staple")
        self.assertTrue(is_password_hashed(encoded))
        self.assertTrue(verify_password("correct horse battery staple", encoded))
        self.assertFalse(verify_password("wrong-password", encoded))

    def test_needs_rehash_for_plaintext_and_legacy_iterations(self):
        self.assertTrue(needs_rehash("plaintext-password"))

        encoded = hash_password("pw", iterations=1_000)
        self.assertTrue(needs_rehash(encoded, min_iterations=200_000))
        self.assertFalse(needs_rehash(encoded, min_iterations=1_000))


if __name__ == "__main__":
    unittest.main()

