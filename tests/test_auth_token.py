"""Gateway API token digest (Bearer auth)."""

import hashlib

from middleware.auth import _bearer_digest


def test_bearer_digest_stable():
    t = "air_testtoken123"
    assert _bearer_digest(t) == hashlib.sha256(t.encode("utf-8")).hexdigest()
