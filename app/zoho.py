"""Zoho CRM — fetch the freshest lead so Shubh can run a real outbound call.

Server-to-server OAuth: a long-lived refresh token (set once in .env) is
exchanged for short-lived access tokens, which are cached in memory until they
expire. No per-call OAuth round trip.

Every failure here is SOFT. If Zoho is unconfigured, unreachable, slow, or
returns nothing, `latest_lead()` returns None and the caller falls back to the
generic greeting. Zoho must never block, delay past its timeout, or crash a
call — personalisation is a bonus on top of a flow that already works.
"""

import logging
import time
from dataclasses import dataclass

import httpx

from app.config import settings

log = logging.getLogger(__name__)

# access_token cached in-process. Zoho tokens live 1h; we refresh a minute early.
_token: dict[str, float | str | None] = {"value": None, "expires_at": 0.0}


@dataclass(frozen=True)
class Lead:
    """One CRM lead, only the parts a sales call cares about. Every field but the
    name is optional — a real CRM record is patchy."""

    id: str
    name: str
    interest: str | None = None
    city: str | None = None
    budget: str | None = None
    timeline: str | None = None
    source: str | None = None
    status: str | None = None

    @property
    def first_name(self) -> str:
        """Just the first name — 'Arun Of Jesus' -> 'Arun'. What you say aloud."""
        parts = self.name.strip().split()
        return parts[0] if parts else self.name.strip()

    def context_block(self) -> str:
        """The facts, as a compact note the model can carry through the whole call.

        Only non-empty fields are listed, so the model is never told 'Budget:
        unknown' and start inventing around it.
        """
        rows = [("Name", self.name)]
        if self.city:
            rows.append(("City", self.city))
        if self.interest:
            rows.append(("Interested in", self.interest))
        if self.budget:
            rows.append(("Estimated budget", self.budget))
        if self.timeline:
            rows.append(("Timeline to buy", self.timeline))
        if self.source:
            rows.append(("Lead source", self.source))
        if self.status:
            rows.append(("Lead status", self.status))
        return "\n".join(f"- {k}: {v}" for k, v in rows)


def enabled() -> bool:
    """True only when the three secrets are all present."""
    return bool(
        settings.zoho_refresh_token
        and settings.zoho_client_id
        and settings.zoho_client_secret
    )


async def _access_token(client: httpx.AsyncClient) -> str | None:
    now = time.monotonic()
    if _token["value"] and now < float(_token["expires_at"]):
        return str(_token["value"])

    # Credentials go in the POST body, NOT the query string: httpx logs the full
    # request URL at INFO, so query params would write the client secret and
    # refresh token into the log file in plaintext. The body is never logged.
    resp = await client.post(
        f"{settings.zoho_accounts_domain}/oauth/v2/token",
        data={
            "refresh_token": settings.zoho_refresh_token,
            "client_id": settings.zoho_client_id,
            "client_secret": settings.zoho_client_secret,
            "grant_type": "refresh_token",
        },
    )
    resp.raise_for_status()
    data = resp.json()
    tok = data.get("access_token")
    if not tok:
        # Zoho returns 200 with an "error" body for a bad/expired refresh token.
        log.warning("zoho: token refresh returned no access_token: %s", data)
        return None

    _token["value"] = tok
    # Refresh 60s early so we never present a token that expires mid-request.
    _token["expires_at"] = now + float(data.get("expires_in", 3600)) - 60
    return tok


def _pick(row: dict, field: str) -> str | None:
    """Read one field from a CRM record, tolerant of Zoho's shapes.

    A field can come back as a plain string, or as a lookup/picklist dict
    ({"name": ...}). Empty/whitespace becomes None.
    """
    if not field:
        return None
    val = row.get(field)
    if isinstance(val, dict):
        val = val.get("name") or val.get("value")
    val = (str(val).strip() if val is not None else "")
    return val or None


def _extract(row: dict) -> Lead | None:
    """Build a Lead from one CRM record. No usable name -> None (useless to call)."""
    name = _pick(row, settings.zoho_name_field)
    if not name:  # fall back to First + Last
        first = _pick(row, "First_Name") or ""
        last = _pick(row, "Last_Name") or ""
        name = f"{first} {last}".strip() or None
    if not name:
        return None

    return Lead(
        id=str(row.get("id", "")),
        name=name,
        interest=_pick(row, settings.zoho_interest_field),
        city=_pick(row, settings.zoho_city_field),
        budget=_pick(row, settings.zoho_budget_field),
        timeline=_pick(row, settings.zoho_timeline_field),
        source=_pick(row, "Lead_Source"),
        status=_pick(row, "Lead_Status"),
    )


async def latest_lead() -> Lead | None:
    """The most recently created lead, or None if unavailable for any reason."""
    if not enabled():
        return None
    try:
        async with httpx.AsyncClient(timeout=settings.zoho_timeout) as client:
            token = await _access_token(client)
            if not token:
                return None

            fields = ",".join(
                dict.fromkeys(  # de-dupe, keep order, drop blanks
                    f
                    for f in (
                        settings.zoho_name_field,
                        settings.zoho_interest_field,
                        settings.zoho_city_field,
                        settings.zoho_budget_field,
                        settings.zoho_timeline_field,
                        "First_Name",
                        "Last_Name",
                        "Lead_Source",
                        "Lead_Status",
                    )
                    if f
                )
            )
            resp = await client.get(
                f"{settings.zoho_api_domain}/crm/v2/{settings.zoho_module}",
                params={
                    "sort_by": "Created_Time",
                    "sort_order": "desc",
                    "per_page": 1,
                    "fields": fields,
                },
                headers={"Authorization": f"Zoho-oauthtoken {token}"},
            )
            if resp.status_code == 204:  # Zoho's "no records" — empty body
                return None
            resp.raise_for_status()
            rows = resp.json().get("data") or []
            if not rows:
                return None
            return _extract(rows[0])
    except Exception as exc:
        # Timeout, network error, auth failure, unexpected shape — all soft.
        log.warning("zoho: could not fetch latest lead (%s) — generic greeting", exc)
        return None
