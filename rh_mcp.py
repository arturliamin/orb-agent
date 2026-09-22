"""
rh_mcp.py — Robinhood Agentic Trading MCP client for the ORB agent.

One-time login (run as the service user so the token file is readable by the agent):
    sudo -u orb /opt/orb-agent/venv/bin/python /opt/orb-agent/rh_mcp.py login
  -> prints a URL. Open it on your DESKTOP, sign in, approve. The browser will end up on a
     http://localhost:8765/callback?... page that fails to load — that's expected. Copy that
     full URL from the address bar and paste it into the terminal.

Read-only probe (proves auth + lists accounts, buying power, a quote):
    sudo -u orb /opt/orb-agent/venv/bin/python /opt/orb-agent/rh_mcp.py probe

Force a token refresh without a browser (a weekly cron can keep the login alive):
    sudo -u orb /opt/orb-agent/venv/bin/python /opt/orb-agent/rh_mcp.py refresh

Pre-open health check (cron, weekdays 7:00 AM ET): renews if possible, and PUSHES you
if the login needs a browser re-authorization or is close to expiry. Needs PUSHOVER_TOKEN
and PUSHOVER_USER in the environment (the cron line loads /etc/orb-agent.env):
    rh_mcp.py check

Check how long the current login has left (exit code 1 if it expires within 2 days):
    sudo -u orb /opt/orb-agent/venv/bin/python /opt/orb-agent/rh_mcp.py status

Library use from orb_agent.py:
    from rh_mcp import RH
    rh = RH()                          # loads saved tokens; refreshes automatically
    rh.call("get_accounts")
    rh.call("review_equity_order", account_number=..., side="buy", symbol="XYZ", type="limit", ...)

Tokens live in $STATE_DIR/rh_tokens.json (default /var/lib/orb-agent). Nothing else is stored.
"""

import asyncio
import json
import logging
import os
import sys
import time
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
INTERACTIVE = False          # set True only by the `login` CLI command
WARN_WITHIN_DAYS = 2

# Every tool the agent calls, with every argument it sends. Verified against the live
# schemas on 2026-09-22. If Robinhood renames a tool or an argument during the beta,
# the pre-open check fails loudly instead of the agent failing mid-trade.
REQUIRED_TOOLS = {
    "get_accounts": [],
    "get_portfolio": ["account_number"],
    "review_equity_order": ["account_number", "side", "symbol", "type", "quantity",
                            "limit_price", "time_in_force"],
    "place_equity_order": ["account_number", "side", "symbol", "type", "quantity",
                           "limit_price", "stop_price", "time_in_force", "ref_id"],
    "get_equity_orders": ["account_number", "order_id"],
    "cancel_equity_order": ["account_number", "order_id"],
}

_sdk_msgs: list[str] = []


class _Capture(logging.Handler):
    """Keeps SDK auth warnings so a failure can report why, not just that."""

    def emit(self, record):
        _sdk_msgs.append(record.getMessage())


logging.getLogger("mcp.client.auth").addHandler(_Capture())
logging.getLogger("mcp.client.auth").setLevel(logging.WARNING)


class LoginExpired(RuntimeError):
    """Raised when the saved login can no longer be refreshed and a browser is required."""


def unwrap(exc: BaseException) -> str:
    """Flatten ExceptionGroup/TaskGroup wrappers into the underlying cause(s)."""
    out = []
    stack = [exc]
    while stack:
        e = stack.pop()
        subs = getattr(e, "exceptions", None)
        if subs:
            stack.extend(subs)
            continue
        out.append(f"{type(e).__name__}: {e}")
    hints = [m for m in _sdk_msgs if "refresh" in m.lower() or "token" in m.lower()]
    return " | ".join(out[:3] + hints[-2:])


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
        d["obtained_at"] = time.time()
        self._save(d)

    def expiry_info(self) -> tuple[float | None, float | None]:
        """(seconds_remaining, days_remaining) for the saved access token, or (None, None)."""
        d = self._load()
        t, got = d.get("tokens"), d.get("obtained_at")
        if not t or not got or not t.get("expires_in"):
            return None, None
        left = got + float(t["expires_in"]) - time.time()
        return left, left / 86400

    async def get_client_info(self) -> OAuthClientInformationFull | None:
        d = self._load().get("client")
        return OAuthClientInformationFull.model_validate(d) if d else None

    async def set_client_info(self, info: OAuthClientInformationFull) -> None:
        d = self._load()
        d["client"] = info.model_dump(exclude_none=True, mode="json")
        self._save(d)


async def _redirect_handler(url: str) -> None:
    if not INTERACTIVE:
        raise LoginExpired(
            "Robinhood login expired and could not be refreshed; re-authorize with: rh_mcp.py login")
    print("\n=== OPEN THIS URL ON YOUR DESKTOP BROWSER ===\n")
    print(url)
    print("\nSign in and approve. The browser will land on a localhost page that fails to load.")
    print("Copy the FULL URL from the address bar and paste it below.\n")


async def _callback_handler() -> AuthorizationCodeResult:
    if not INTERACTIVE:
        raise LoginExpired(
            "Robinhood login expired and could not be refreshed; re-authorize with: rh_mcp.py login")
    pasted = await asyncio.get_event_loop().run_in_executor(None, input, "Paste redirected URL: ")
    cleaned = pasted.strip().strip("'\"<>` ")
    q = parse_qs(urlparse(cleaned).query)
    code = ((q.get("code") or [None])[0] or "").strip("'\" ")
    state = ((q.get("state") or [None])[0] or "").strip("'\" ") or None
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
    structured = getattr(result, "structured_content", None) or getattr(result, "structuredContent", None)
    if structured:
        return structured
    texts = [c.text for c in result.content if getattr(c, "type", "") == "text"]
    joined = "\n".join(texts)
    try:
        return json.loads(joined)
    except Exception:  # noqa: BLE001
        return joined


class RH:
    """Synchronous wrapper used by the agent. Raises LoginExpired when a browser is needed."""

    @staticmethod
    def _run(fn):
        _sdk_msgs.clear()
        try:
            return asyncio.run(_with_session(fn))
        except LoginExpired:
            raise
        except BaseException as e:  # noqa: BLE001
            flat = unwrap(e)
            if "LoginExpired" in flat or "No ?code=" in flat:
                raise LoginExpired(flat) from None
            raise RuntimeError(flat) from None

    def call(self, tool: str, **args):
        async def go(session):
            res = await session.call_tool(tool, args)
            # SDK 2.x renamed isError -> is_error; checking only the old name made every
            # server-side rejection look like success. Read both.
            if getattr(res, "is_error", None) or getattr(res, "isError", None):
                raise RuntimeError(f"{tool}: {_unwrap(res)}")
            return _unwrap(res)
        return self._run(go)

    def tools(self):
        async def go(session):
            return [t.name for t in (await session.list_tools()).tools]
        return self._run(go)

    def tool_schemas(self) -> dict:
        """{tool_name: set(argument names)} from the live server."""
        async def go(session):
            out = {}
            for t in (await session.list_tools()).tools:
                schema = getattr(t, "input_schema", None) or getattr(t, "inputSchema", None) or {}
                props = schema.get("properties", {}) or {}
                out[t.name] = set(props)
            return out
        return self._run(go)

    def schema_problems(self) -> list[str]:
        """Empty list when every required tool and argument exists."""
        live = self.tool_schemas()
        problems = []
        for tool, args in REQUIRED_TOOLS.items():
            if tool not in live:
                problems.append(f"missing tool {tool}")
                continue
            missing = [a for a in args if a not in live[tool]]
            if missing:
                problems.append(f"{tool} lost args {missing}")
        return problems

    @staticmethod
    def days_left() -> float | None:
        return FileTokenStorage().expiry_info()[1]


# ---------------------------------------------------------------- CLI
def _cmd_login():
    global INTERACTIVE
    INTERACTIVE = True
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


def _cmd_refresh():
    """Force a headless renewal by making a trivial call; the SDK refreshes if needed."""
    before = FileTokenStorage().expiry_info()[1]
    try:
        n = len(RH().tools())
    except LoginExpired as e:
        print(f"REFRESH FAILED — browser login required.\n{e}")
        raise SystemExit(1)
    after = FileTokenStorage().expiry_info()[1]
    print(f"OK ({n} tools). Days left: {before:.1f} -> {after:.1f}" if before and after else f"OK ({n} tools).")
    if before and after and after <= before + 0.01:
        print("NOTE: expiry did not move — the token was still valid, so no refresh was attempted.")


def _notify(title: str, message: str, priority: int = 0) -> None:
    """Pushover via stdlib so the check has no extra dependencies."""
    import urllib.parse
    import urllib.request
    tok, usr = os.environ.get("PUSHOVER_TOKEN"), os.environ.get("PUSHOVER_USER")
    if not tok or not usr:
        print(f"(no Pushover credentials in env; would have sent: {title} | {message})")
        return
    data = urllib.parse.urlencode({"token": tok, "user": usr, "title": title,
                                   "message": message, "priority": priority}).encode()
    try:
        urllib.request.urlopen("https://api.pushover.net/1/messages.json", data=data, timeout=15)
    except Exception as e:  # noqa: BLE001
        print(f"push failed: {e}")


def _cmd_check():
    """Daily pre-open check. Silent when healthy; pushes when you need to act."""
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    relogin = "Re-authorize from your phone:\nsudo -u orb /opt/orb-agent/venv/bin/python /opt/orb-agent/rh_mcp.py login\nthen: systemctl restart orb-agent"
    last = None
    for attempt in (1, 2):
        try:
            rh = RH()
            problems = rh.schema_problems()
            if problems:
                print(f"{stamp} TOOL SCHEMA CHANGED: {problems}")
                _notify("Robinhood tools changed — execution disabled today",
                        "The agent's order tools no longer match Robinhood's:\n"
                        + "\n".join(problems[:5]) + "\nAlerts continue; send this to be patched.", 1)
                raise SystemExit(2)
            n = len(REQUIRED_TOOLS)
            days = FileTokenStorage().expiry_info()[1]
            print(f"{stamp} check OK ({n}/{n} required tools verified), login {days:.1f}d left"
                  if days is not None else f"{stamp} check OK ({n}/{n} required tools verified)")
            if days is not None and days < WARN_WITHIN_DAYS:
                _notify("Robinhood login expiring",
                        f"About {days:.1f} day(s) left.\n{relogin}", 1)
            return
        except LoginExpired as e:
            print(f"{stamp} LOGIN EXPIRED: {e}")
            _notify("Robinhood login expired — act before the open",
                    f"Execution will be OFF today unless you re-authorize.\n{relogin}", 1)
            raise SystemExit(1)
        except Exception as e:  # noqa: BLE001
            last = e
            print(f"{stamp} check attempt {attempt} failed: {e}")
            if attempt == 1:
                time.sleep(30)
    _notify("Robinhood check failed",
            f"Couldn't reach Robinhood at 7:00 (not a login problem).\n{str(last)[:200]}\n"
            f"The 9:15 check will retry.", 0)
    raise SystemExit(1)


def _cmd_status():
    left, days = FileTokenStorage().expiry_info()
    if days is None:
        print("No saved token. Run: rh_mcp.py login")
        raise SystemExit(1)
    print(f"Robinhood login: {days:.1f} days left ({left/3600:.0f}h)")
    raise SystemExit(1 if days < WARN_WITHIN_DAYS else 0)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "probe"
    {"login": _cmd_login, "probe": _cmd_probe,
     "refresh": _cmd_refresh, "status": _cmd_status, "check": _cmd_check}.get(cmd, _cmd_probe)()
