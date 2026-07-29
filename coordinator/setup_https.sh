#!/usr/bin/env bash
# coordinator/setup_https.sh — give the coordinator a real HTTPS hostname, so that
# "Sign in with Google" works and node tokens stop crossing the internet in cleartext.
#
# Run ON THE COORDINATOR VM:
#     curl -O <this file>  &&  sudo bash setup_https.sh neuron.example.com
#     sudo bash setup_https.sh yourname.duckdns.org        # free subdomain also fine
#
# WHY THIS IS NEEDED AT ALL. Google refuses to register a redirect URI that is plain HTTP or a
# raw IP address -- only localhost is exempt -- so http://150.230.22.250:8001/auth/callback/google
# simply cannot be entered in the Google console. GitHub is laxer and works on the bare IP, but
# GitHub is a developer platform: requiring it means only developers can ever sign in. Everyone
# has a Google account. This script is what turns NEURON from developer-only into something a
# normal person can use, and it is the single largest usability step left.
#
# Caddy is used because it obtains and renews a Let's Encrypt certificate automatically, with no
# cron job, no certbot invocation and no renewal to forget. Cost: a domain (~EUR 10/year, or free
# via DuckDNS). Everything else here is free.
set -euo pipefail

DOMAIN="${1:-}"
if [ -z "$DOMAIN" ]; then
  echo "usage: sudo bash setup_https.sh <domain>"
  echo "  e.g. sudo bash setup_https.sh neuron.example.com"
  echo "       sudo bash setup_https.sh yourname.duckdns.org   (free)"
  exit 1
fi
[ "$(id -u)" -eq 0 ] || { echo "run with sudo"; exit 1; }

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

say "checking that $DOMAIN points at this machine"
myip="$(curl -fsS -m 10 https://api.ipify.org || echo '?')"
resolved="$(getent hosts "$DOMAIN" | awk '{print $1}' | head -1 || true)"
echo "   this VM resolves to : $myip"
echo "   $DOMAIN resolves to : ${resolved:-<nothing>}"
if [ "$resolved" != "$myip" ]; then
  cat <<EOF

   STOP. $DOMAIN does not point here yet, and Let's Encrypt will fail.
   Add a DNS A record:   $DOMAIN  ->  $myip
   (DuckDNS: set the IP on duckdns.org. Registrar: add an A record.)
   DNS can take a few minutes. Re-run this script once the two lines above match.
EOF
  exit 1
fi

say "opening ports 80 and 443"
# Oracle's Ubuntu images ship an iptables ruleset that REJECTs everything except 22, so a
# missing rule here looks exactly like a broken certificate request.
iptables -C INPUT -p tcp --dport 80 -j ACCEPT 2>/dev/null || iptables -I INPUT 5 -p tcp --dport 80 -j ACCEPT
iptables -C INPUT -p tcp --dport 443 -j ACCEPT 2>/dev/null || iptables -I INPUT 5 -p tcp --dport 443 -j ACCEPT
if command -v netfilter-persistent >/dev/null 2>&1; then
  netfilter-persistent save >/dev/null 2>&1 || true
else
  DEBIAN_FRONTEND=noninteractive apt-get install -y iptables-persistent >/dev/null 2>&1 || true
fi
echo "   done (local firewall)"
cat <<EOF

   ALSO REQUIRED, and not something this script can do: Oracle has a SECOND firewall in the
   cloud console. Add ingress rules for TCP 80 and TCP 443 from 0.0.0.0/0 under
   Networking -> Virtual Cloud Networks -> your VCN -> Security Lists -> Default.
   Without it the certificate request times out with no useful error.
EOF
read -r -p "   Press Enter once those ingress rules exist (or Ctrl-C to stop)... " _ || true

say "installing Caddy"
if ! command -v caddy >/dev/null 2>&1; then
  apt-get update -qq
  apt-get install -y debian-keyring debian-archive-keyring apt-transport-https curl >/dev/null
  curl -fsSL https://dl.cloudsmith.io/public/caddy/stable/gpg.key \
    | gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
  echo "deb [signed-by=/usr/share/keyrings/caddy-stable-archive-keyring.gpg] https://dl.cloudsmith.io/public/caddy/stable/deb/debian any-version main" \
    > /etc/apt/sources.list.d/caddy-stable.list
  apt-get update -qq && apt-get install -y caddy >/dev/null
fi
caddy version

say "configuring the reverse proxy"
cat > /etc/caddy/Caddyfile <<EOF
# NEURON coordinator. Caddy terminates TLS (certificate obtained and renewed automatically)
# and forwards to the coordinator, which keeps listening on plain HTTP on loopback only.
$DOMAIN {
    reverse_proxy 127.0.0.1:8001
}
EOF
systemctl restart caddy
sleep 6
systemctl is-active caddy >/dev/null && echo "   caddy is running"

say "verifying HTTPS"
if curl -fsS -m 25 "https://$DOMAIN/status" >/dev/null; then
  echo "   https://$DOMAIN/status  OK"
else
  echo "   FAILED. Check: journalctl -u caddy -n 40 --no-pager"
  echo "   Most common cause: the Oracle Security List ingress rules above are missing."
  exit 1
fi

say "pointing the coordinator at its new public address"
# PUBLIC_BASE_URL is what the coordinator sends to Google/GitHub as redirect_uri, and providers
# compare it verbatim -- so it must be the HTTPS name from now on, not the old IP.
#
# Write it to a DROP-IN, not the main unit. systemd applies drop-ins after the unit, so a value
# set in .service.d/*.conf silently overrides the same key in the .service file. Editing only the
# main unit (as this did) left the old http:// value winning, and the coordinator kept sending a
# redirect_uri that no longer matched the provider -- with no error until a login was attempted.
UNIT=/etc/systemd/system/neuron-coordinator.service
DROPIN_DIR="$UNIT.d"
mkdir -p "$DROPIN_DIR"
# strip the key wherever it may already live, so exactly one definition survives
sed -i '/^Environment=NEURON_PUBLIC_BASE_URL=/d' "$UNIT"
for f in "$DROPIN_DIR"/*.conf; do
  [ -e "$f" ] && sed -i '/^Environment=NEURON_PUBLIC_BASE_URL=/d' "$f"
done
printf '[Service]\nEnvironment=NEURON_PUBLIC_BASE_URL=https://%s\n' "$DOMAIN" \
  > "$DROPIN_DIR/public-url.conf"
chmod 644 "$DROPIN_DIR/public-url.conf"
systemctl daemon-reload && systemctl restart neuron-coordinator
sleep 4
echo "   effective value: $(systemctl show neuron-coordinator -p Environment | tr ' ' '\n' | grep PUBLIC_BASE_URL || echo '(NOT SET)')"

cat <<EOF

$(printf '\033[1m== done\033[0m')

Coordinator is live at:  https://$DOMAIN

Next, and this is the part that makes NEURON usable by people who are not developers:

  1. Google Cloud Console -> APIs & Services -> Credentials
     -> Create credentials -> OAuth client ID -> Web application
  2. Authorized redirect URI (paste exactly):
        https://$DOMAIN/auth/callback/google
  3. Add the id/secret to $UNIT :
        Environment=NEURON_GOOGLE_CLIENT_ID=...
        Environment=NEURON_GOOGLE_CLIENT_SECRET=...
     then:  sudo systemctl daemon-reload && sudo systemctl restart neuron-coordinator
  4. Check:  curl https://$DOMAIN/auth/providers

Every installed agent picks this up on its own -- no reinstall, and nothing for a user to
configure. They just see a "Log in with Google" button and click it.

Finally, update the agent default so new installs use HTTPS:
  agent/config.json  ->  "coordinator": "https://$DOMAIN"
EOF
