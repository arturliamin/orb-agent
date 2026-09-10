"""
rh_mcp.py — Robinhood Agentic Trading MCP client for the ORB agent.

One-time login (run as the service user so the token file is readable by the agent):
    sudo -u orb /opt/orb-agent/venv/bin/python /opt/orb-agent/rh_mcp.py login
  -> prints a URL. Open it on your DESKTOP, sign in, approve. The browser will end up on a
     http://localhost:8765/callback?... page that fails to load — that's expected. Copy that
     full URL from the address bar and paste it into the terminal.

Read-only probe (proves auth + lists accounts, buying power, a quote):
    sudo -u orb /opt/orb-agent/venv/bin/python /opt/orb-agent/rh_mcp.py probe

Library use from orb_agent.py:
    from rh_mcp import RH
    rh = RH()                          # loads saved tokens; refreshes automatically
    rh.call("get_accounts")
    rh.call("review_equity_order", account_number=..., side="buy", symbol="XYZ", type="limit", ...)

Tokens live in $STATE_DIR/rh_tokens.json (default /var/lib/orb-agent). Nothing else is stored.
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from mcp import ClientSession
from mcp.client.auth import OAuthClientProvider, TokenStorage
from mcp.client.streamable_http import streamable_http_client
from mcp.shared._httpx_utils import create_mcp_http_client
from mcp.shared.auth import (AuthorizationCodeResult, OAuthClientInformationFull,
                             OAuthClientMetadata, OAuthToken)

MCP_URL = os.environ.get("RH_MCP_URL", "https://agent.robinhood.com/mcp/trading")
STATE_DIR = Path(os.environ.get("STATE_DIR", "/var/lib/orb-agent"))
TOKEN_FILE = STATE_DIR / "rh_tokens.json"
REDIRECT_URI = "http://localhost:8765/callback"


class FileTokenStorage(TokenStorage):
    """Persists OAuth tokens + registered client info to a 0600 JSON file."""

    def _load(self) -> dict:
        if TOKEN_FILE.exists():
            return json.loads(TOKEN_FILE.read_text())
        return {}

    def _save(self, d: dict) -> None:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        TOKEN_FILE.write_text(json.dumps(d, indent=1))
        os.chmod(TOKEN_FILE, 0o600)

    async def get_tokens(self) -> OAuthToken | None:
        d = self._load().get("tokens")
        return OAuthToken.model_validate(d) if d else None

    async def set_tokens(self, tokens: OAuthToken) -> None:
        d = self._load()
        d["tokens"] = tokens.model_dump(exclude_none=True)
        self._save(d)

    async def get_client_info(self) -> OAuthClientInformationFull | None:
        d = self._load().get("client")
        return OAuthClientInformationFull.model_validate(d) if d else None

    async def set_client_info(self, info: OAuthClientInformationFull) -> None:
        d = self._load()
        d["client"] = info.model_dump(exclude_none=True, mode="json")
        self._save(d)


async def _redirect_handler(url: str) -> None:
    print("\n=== OPEN THIS URL ON YOUR DESKTOP BROWSER ===\n")
    print(url)
    print("\nSign in and approve. The browser will land on a localhost page that fails to load.")
    print("Copy the FULL URL from the address bar and paste it below.\n")


async def _callback_handler() -> AuthorizationCodeResult:
    pasted = await asyncio.get_event_loop().run_in_executor(None, input, "Paste redirected URL: ")
    q = parse_qs(urlparse(pasted.strip()).query)
    code = (q.get("code") or [None])[0]
    state = (q.get("state") or [None])[0]
    if not code:
        raise SystemExit("No ?code= found in what you pasted. Try the login again.")
    return AuthorizationCodeResult(code=code, state=state)


def _provider() -> OAuthClientProvider:
    return OAuthClientProvider(
        server_url=MCP_URL,
        client_metadata=OAuthClientMetadata(
            client_name="ORB alert agent",
            redirect_uris=[REDIRECT_URI],
            grant_types=["authorization_code", "refresh_token"],
            response_types=["code"],
            token_endpoint_auth_method="none",
        ),
        storage=FileTokenStorage(),
        redirect_handler=_redirect_handler,
        callback_handler=_callback_handler,
    )


async def _with_session(fn):
    async with create_mcp_http_client(auth=_provider()) as http:
        async with streamable_http_client(MCP_URL, http_client=http) as (read, write, *_):
            async with ClientSession(read, write) as session:
                await session.initialize()
                return await fn(session)


def _unwrap(result):
    """Tool results come back as content blocks; return parsed JSON when possible."""
    if getattr(result, "structuredContent", None):
        return result.structuredContent
    texts = [c.text for c in result.content if getattr(c, "type", "") == "text"]
    joined = "\n".join(texts)
    try:
        return json.loads(joined)
    except Exception:  # noqa: BLE001
        return joined


class RH:
    """Synchronous wrapper used by the agent."""

    def call(self, tool: str, **args):
        async def go(session):
            res = await session.call_tool(tool, args)
            if getattr(res, "isError", False):
                raise RuntimeError(f"{tool}: {_unwrap(res)}")
            return _unwrap(res)
        return asyncio.run(_with_session(go))

    def tools(self):
        async def go(session):
            return [t.name for t in (await session.list_tools()).tools]
        return asyncio.run(_with_session(go))


# ---------------------------------------------------------------- CLI
def _cmd_login():
    rh = RH()
    names = rh.tools()
    print(f"\nLogged in. {len(names)} tools available. Tokens saved to {TOKEN_FILE}")


def _cmd_probe():
    rh = RH()
    accts = rh.call("get_accounts")
    data = accts.get("data", accts) if isinstance(accts, dict) else accts
    agentic = None
    for a in data.get("accounts", []):
        flag = "AGENTIC" if a.get("agentic_allowed") else "read-only"
        print(f"account ••••{a['account_number'][-4:]}  {a.get('nickname') or a.get('brokerage_account_type')}  {a.get('type')}  {flag}")
        if a.get("agentic_allowed"):
            agentic = a["account_number"]
    if not agentic:
        print("No agentic-enabled account visible to this client."); return
    port = rh.call("get_portfolio", account_number=agentic)
    p = port.get("data", port)
    print(f"agentic buying power: ${p['buying_power']['buying_power']}  pending deposits: ${p.get('pending_deposits', '0')}")
    q = rh.call("get_equity_quotes", symbols=["SPY"])
    print("quote SPY:", json.dumps(q.get("data", q))[:200])
    print("\nProbe OK.")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "probe"
    {"login": _cmd_login, "probe": _cmd_probe}.get(cmd, _cmd_probe)()
