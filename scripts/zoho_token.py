"""One-time helper: exchange a Zoho Self Client GRANT CODE for a refresh token,
and write it straight into .env — the token is never printed.

Why this exists: a Zoho grant code is single-use and expires in ~10 min. It is
NOT a refresh token; pasting it into ZOHO_REFRESH_TOKEN gives {'error':
'invalid_code'}. This does the exchange the right way.

Steps:
  1. api-console.zoho.in -> your Self Client -> Generate Code tab.
     Scope: ZohoCRM.modules.READ   Duration: 10 min   -> copy the grant code.
  2. Make sure ZOHO_CLIENT_ID and ZOHO_CLIENT_SECRET are already in .env, and
     ZOHO_ACCOUNTS_DOMAIN matches the data centre you generated the code on
     (default https://accounts.zoho.in).
  3. Run within 10 minutes:
        uv run python scripts/zoho_token.py PASTE_THE_GRANT_CODE_HERE

On success it updates ZOHO_REFRESH_TOKEN in .env in place. Restart the server.
"""

import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import PROJECT_ROOT, settings  # noqa: E402

ENV_PATH = PROJECT_ROOT / ".env"


def _write_env(key: str, value: str) -> None:
    """Set key=value in .env, replacing an existing line or appending."""
    lines = ENV_PATH.read_text(encoding="utf-8").splitlines() if ENV_PATH.exists() else []
    out, found = [], False
    for line in lines:
        if line.strip().startswith(f"{key}=") or line.strip().startswith(f"{key} ="):
            out.append(f"{key}={value}")
            found = True
        else:
            out.append(line)
    if not found:
        out.append(f"{key}={value}")
    ENV_PATH.write_text("\n".join(out) + "\n", encoding="utf-8")


def main() -> int:
    if len(sys.argv) != 2 or not sys.argv[1].strip():
        print("usage: uv run python scripts/zoho_token.py <GRANT_CODE>")
        return 2
    code = sys.argv[1].strip()

    if not settings.zoho_client_id or not settings.zoho_client_secret:
        print("ERROR: set ZOHO_CLIENT_ID and ZOHO_CLIENT_SECRET in .env first.")
        return 1

    url = f"{settings.zoho_accounts_domain}/oauth/v2/token"
    print(f"exchanging grant code at {url} ...")
    try:
        resp = httpx.post(
            url,
            data={
                "grant_type": "authorization_code",
                "client_id": settings.zoho_client_id,
                "client_secret": settings.zoho_client_secret,
                "code": code,
            },
            timeout=15,
        )
    except Exception as exc:
        print(f"ERROR: request failed ({exc}). Check ZOHO_ACCOUNTS_DOMAIN / network.")
        return 1

    data = {}
    try:
        data = resp.json()
    except Exception:
        pass

    refresh = data.get("refresh_token")
    if not refresh:
        err = data.get("error", resp.text[:200])
        hint = {
            "invalid_code": "grant code is wrong, already used, or expired (>10 min). Generate a fresh one.",
            "invalid_client": "client id/secret mismatch, or wrong data centre (.in vs .com).",
        }.get(err, "")
        print(f"FAILED: {err}. {hint}".rstrip())
        return 1

    _write_env("ZOHO_REFRESH_TOKEN", refresh)
    print("OK: refresh token written to .env (value not shown). Restart the server.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
