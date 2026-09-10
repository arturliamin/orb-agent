#!/bin/bash
# One-shot installer for the ORB agent on a fresh Ubuntu VPS. Run as root:
#   bash install.sh
set -e
apt-get update -y && apt-get install -y python3 python3-venv git
id -u orb &>/dev/null || useradd -r -m -s /usr/sbin/nologin orb
mkdir -p /opt/orb-agent /var/lib/orb-agent
cp orb_agent.py /opt/orb-agent/
python3 -m venv /opt/orb-agent/venv
/opt/orb-agent/venv/bin/pip install -q alpaca-py pandas numpy requests mcp
cp rh_mcp.py /opt/orb-agent/ 2>/dev/null || true
chown -R orb:orb /opt/orb-agent /var/lib/orb-agent
cp orb-agent.service /etc/systemd/system/
if [ ! -f /etc/orb-agent.env ]; then
cat > /etc/orb-agent.env <<'ENV'
ALPACA_API_KEY=PASTE_KEY_ID
ALPACA_SECRET_KEY=PASTE_SECRET
PUSHOVER_TOKEN=PASTE_APP_TOKEN
PUSHOVER_USER=PASTE_USER_KEY
ACCOUNT_EQUITY=10000
ENV
chmod 600 /etc/orb-agent.env
echo ">>> Edit /etc/orb-agent.env with your keys, then run: systemctl enable --now orb-agent"
else
systemctl daemon-reload && systemctl enable --now orb-agent && systemctl restart orb-agent
echo ">>> Agent running. Logs: journalctl -u orb-agent -f"
fi
