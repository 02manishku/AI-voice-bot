"""Zoho lead -> outbound "Lead Call". The parts that carry real logic:

  - Lead.first_name / Lead.context_block(): what Shubh says vs. what he silently
    knows; only populated facts appear (no "Budget: unknown").
  - outbound_greeting(): opens like a caller ("this is Shubh calling from ..."),
    first name only, and a name-less fallback — never "how can I help you".
  - lead_call_context(): the pinned note tells him it's outbound and not to
    re-ask known facts.
  - zoho._extract(): tolerant of Zoho's record shapes; "no usable name" -> None.
  - zoho.enabled(): off unless all three secrets are present.

The live HTTP round trip is verified by ear once real credentials are in .env;
here we pin the pure logic so it can't regress.

Run: uv run python tests/test_zoho.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app import zoho
from app.config import settings
from app.prompts import ASSISTANT_GREETING, lead_call_context, outbound_greeting
from app.zoho import Lead


def check(label, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}")
    return cond


ok = True

# --- Lead.first_name / context_block ----------------------------------------
print("lead fields:")
lead = Lead(
    id="1", name="Arun Of Jesus", city="Bangalore",
    budget="Between 11-15 Lakhs", timeline="After 6 Months",
    interest=None, source="Adglobal", status="Not Contacted Yet",
)
ok &= check("first_name is just the first token", lead.first_name == "Arun")
block = lead.context_block()
ok &= check("context block lists the name", "Arun Of Jesus" in block)
ok &= check("context block lists city + budget + timeline",
           "Bangalore" in block and "11-15" in block and "After 6 Months" in block)
ok &= check("empty fields are omitted (no 'Interested in')", "Interested in" not in block)

# --- outbound_greeting ------------------------------------------------------
print("\noutbound greeting:")
g = outbound_greeting("Arun")
ok &= check("opens as the caller ('calling from')", "calling from Magppie" in g)
ok &= check("uses the first name", "Arun" in g)
ok &= check("does NOT say 'how can I help you'", "how can i help" not in g.lower())
ok &= check("name-less fallback still greets", "Magppie" in outbound_greeting(None))
ok &= check("name-less fallback drops the name line", "speaking with" not in outbound_greeting("").lower())

# --- lead_call_context ------------------------------------------------------
print("\nlead-call context note:")
ctx = lead_call_context(block, outbound_greeting("Arun"))
ok &= check("marks the call outbound", "OUTBOUND" in ctx)
ok &= check("tells him not to re-ask known facts", "never ask" in ctx.lower())
ok &= check("says he already greeted (won't greet twice)", "ALREADY opened" in ctx)
ok &= check("carries the facts", "Bangalore" in ctx)

# --- assistant greeting is distinct + inbound -------------------------------
print("\nassistant (inbound) greeting:")
ok &= check("assistant greeting offers help", "how can i help" in ASSISTANT_GREETING.lower())

# --- _extract: Zoho record shapes -------------------------------------------
print("\nlead extraction from CRM record shapes:")
settings.zoho_name_field = "Full_Name"
settings.zoho_interest_field = "Description"
settings.zoho_city_field = "City"
settings.zoho_budget_field = "Est_Budget"
settings.zoho_timeline_field = "How_Soon_Do_You_Require_Magppie"

row = {
    "id": "1", "Full_Name": "Neha Gupta", "City": "Pune",
    "Est_Budget": "6-10 Lakhs", "How_Soon_Do_You_Require_Magppie": "Immediately",
    "Lead_Source": "Instagram", "Lead_Status": "Contacted",
}
ld = zoho._extract(row)
ok &= check("name + city + budget + timeline extracted",
           ld and ld.name == "Neha Gupta" and ld.city == "Pune"
           and ld.budget == "6-10 Lakhs" and ld.timeline == "Immediately")
ok &= check("source + status extracted", ld and ld.source == "Instagram" and ld.status == "Contacted")

ld = zoho._extract({"id": "2", "Full_Name": "  ", "First_Name": "Vikram", "Last_Name": "Rao"})
ok &= check("blank Full_Name -> First+Last fallback", ld and ld.name == "Vikram Rao")

ld = zoho._extract({"id": "3", "Full_Name": {"name": "Owner Lookup"}, "City": None})
ok &= check("lookup-dict name unwrapped", ld and ld.name == "Owner Lookup")
ok &= check("null city -> None", ld and ld.city is None)

ok &= check("no usable name -> None", zoho._extract({"id": "4", "City": "x"}) is None)

# --- llm message assembly: context is pinned WITHOUT breaking the cache -------
print("\nllm context threading (Task 1 — remember the lead, keep the cache):")
from app import llm  # noqa: E402

msgs = llm._messages("KB TEXT HERE", "hi there", [{"role": "user", "content": "prev"}],
                     "en-IN", context="LEAD NOTE: Arun, Bangalore")
ok &= check("first message is still the big KB system (cache prefix intact)",
           msgs[0]["role"] == "system" and "KB TEXT HERE" in msgs[0]["content"])
ok &= check("context rides as a 2nd system message", msgs[1]["role"] == "system" and "Arun" in msgs[1]["content"])
ok &= check("history + user still follow", msgs[2]["content"] == "prev" and msgs[-1]["role"] == "user")

no_ctx = llm._messages("KB", "q", [], "en-IN")
ok &= check("no context -> no extra system message (assistant mode unchanged)",
           sum(1 for m in no_ctx if m["role"] == "system") == 1)

# --- enabled() gating -------------------------------------------------------
print("\nenabled() gating:")
saved = (settings.zoho_client_id, settings.zoho_client_secret, settings.zoho_refresh_token)
settings.zoho_client_id = settings.zoho_client_secret = settings.zoho_refresh_token = ""
ok &= check("all secrets blank -> disabled", zoho.enabled() is False)
settings.zoho_client_id, settings.zoho_client_secret, settings.zoho_refresh_token = "a", "b", "c"
ok &= check("all secrets set -> enabled", zoho.enabled() is True)
settings.zoho_client_id = ""
ok &= check("one secret missing -> disabled", zoho.enabled() is False)
settings.zoho_client_id, settings.zoho_client_secret, settings.zoho_refresh_token = saved

print("\nOK" if ok else "\nFAILED")
sys.exit(0 if ok else 1)
