"""Magppie voice demo.

This machine's TLS is intercepted (corporate proxy / AV), so the interceptor's
root CA lives in the Windows cert store and not in certifi. Without this, every
outbound call to Sarvam and OpenAI dies with CERTIFICATE_VERIFY_FAILED, which
reads exactly like a bad API key. Must run before any SSLContext is built.
"""

import truststore

truststore.inject_into_ssl()
