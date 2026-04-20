class ProviderRateLimitError(Exception):
    """Raised when a provider returns HTTP 429 (rate limited)."""
